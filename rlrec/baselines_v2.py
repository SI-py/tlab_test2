"""Sparse item-neighbour baseline fitted only on the supplied training prefix."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import sparse


class ItemKNN:
    """Weighted cosine item neighbours with a recency/IDF query history.

    Training incidence is binary per user. A user's contribution is weighted by
    1/log(2 + distinct_items), reducing the effect of unusually active users.
    Cosine similarities are shrunk by weighted co-occurrence/(co-occurrence+
    shrinkage). Only ``neighbors`` entries per item are retained. Multiplication
    is done in item blocks, never as a dense item-by-item matrix.

    At inference the latest ``max_history`` known items receive exponential
    position recency weights and an IDF factor. Re-listens may contribute more
    than once. The evaluator controls seen-item masking for each dataset.
    Scores are ranking scores, not calibrated response probabilities.
    """

    def __init__(self, neighbors=100, shrinkage=10., half_life=10., max_history=64,
                 popularity_weight=.01, block_size=128):
        if neighbors < 1 or shrinkage < 0 or half_life <= 0 or max_history < 1 or block_size < 1:
            raise ValueError("Invalid neighbourhood or history configuration")
        self.neighbors = int(neighbors)
        self.shrinkage = float(shrinkage)
        self.half_life = float(half_life)
        self.max_history = int(max_history)
        self.popularity_weight = float(popularity_weight)
        self.block_size = int(block_size)

    def fit(self, train: pd.DataFrame, vocab_size: int):
        if train.empty or vocab_size < 4:
            raise ValueError("Need training rows and a real-item vocabulary")
        self.vocab_size = int(vocab_size)
        # Histories recover earlier prefix edges even when targets were sampled.
        # Sets remove repeated rows, same-day multiple targets, and re-listens.
        observed: dict[int, set[int]] = {}
        for row in train.itertuples(index=False):
            items = observed.setdefault(int(row.user_id), set())
            items.update(int(token) for token in row.history if 3 <= int(token) < vocab_size)
            if 3 <= int(row.target) < vocab_size:
                items.add(int(row.target))
        users = sorted(observed)
        row_parts, column_parts = [], []
        for index, uid in enumerate(users):
            items = np.asarray(sorted(observed[uid]), dtype=np.int32)
            if len(items):
                row_parts.append(np.full(len(items), index, dtype=np.int32))
                column_parts.append(items)
        if not column_parts:
            raise ValueError("No known item in training prefixes")
        row_indices = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
        incidence = sparse.csr_matrix((np.ones(len(columns), np.float32), (row_indices, columns)),
                                      shape=(len(users), vocab_size), dtype=np.float32)
        del observed, row_parts, column_parts, row_indices, columns
        counts = np.asarray(incidence.sum(axis=0)).reshape(-1)
        self.item_user_count = counts.astype(np.float32)
        self.idf = (1 + np.log((len(users)+1)/(counts+1))).astype(np.float32)
        self.idf[:3] = 0
        self.popularity = np.log1p(counts).astype(np.float32)
        self.popularity[:3] = 0
        if self.popularity.max() > 0:
            self.popularity /= self.popularity.max()
        activity = np.diff(incidence.indptr)
        user_weight = (1/np.log(2+activity)).astype(np.float32)
        weighted = incidence.multiply(np.sqrt(user_weight)[:, None]).tocsr()
        norm = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).reshape(-1))
        weighted_columns = weighted.tocsc()
        all_rows, all_columns, all_values = [], [], []
        for start in range(3, vocab_size, self.block_size):
            stop = min(start+self.block_size, vocab_size)
            cooccurrence = (weighted_columns[:, start:stop].T @ weighted).tocsr()
            for local in range(stop-start):
                item = start+local
                begin, end = cooccurrence.indptr[local:local+2]
                candidates = cooccurrence.indices[begin:end]
                common = cooccurrence.data[begin:end]
                keep = (candidates >= 3) & (candidates != item) & (common > 0)
                candidates, common = candidates[keep], common[keep]
                if not len(candidates) or norm[item] == 0:
                    continue
                denominator = norm[item]*norm[candidates]
                values = np.divide(common, denominator, out=np.zeros_like(common), where=denominator > 0)
                if self.shrinkage:
                    values *= common/(common+self.shrinkage)
                positive = values > 0
                candidates, values = candidates[positive], values[positive]
                if len(candidates) > self.neighbors:
                    threshold = np.partition(values, -self.neighbors)[-self.neighbors]
                    near = values >= threshold
                    candidates, values = candidates[near], values[near]
                order = np.lexsort((candidates, -values))[:self.neighbors]
                all_rows.extend([item]*len(order))
                all_columns.extend(candidates[order].tolist())
                all_values.extend(values[order].tolist())
        self.similarity = sparse.csr_matrix((np.asarray(all_values, np.float32),
            (np.asarray(all_rows, np.int32), np.asarray(all_columns, np.int32))),
            shape=(vocab_size, vocab_size), dtype=np.float32)
        self.training_manifest = {
            "model": "weighted cosine item-kNN", "train_rows": len(train), "train_users": len(users),
            "vocab_size": vocab_size, "known_items": int((counts[3:] > 0).sum()),
            "user_item_edges": int(incidence.nnz), "neighbour_edges": int(self.similarity.nnz),
            "neighbors": self.neighbors, "shrinkage": self.shrinkage,
            "half_life_positions": self.half_life, "max_history": self.max_history,
            "popularity_weight": self.popularity_weight,
            "incidence_rule": "Binary union of supplied training targets and their strictly earlier histories; no validation/test events",
            "weighting": "Inverse-log user activity in cosine; query item IDF; positional exponential recency",
        }
        return self

    def __call__(self, histories):
        if not hasattr(self, "similarity"):
            raise ValueError("Fit the item-kNN model first")
        rows, columns, weights = [], [], []
        for index, history in enumerate(histories):
            known = [int(token) for token in history if 3 <= int(token) < self.vocab_size][-self.max_history:]
            if not known:
                continue
            recency = np.exp2(-np.arange(len(known)-1, -1, -1)/self.half_life)
            value = recency*self.idf[known]
            value /= value.sum()
            rows.extend([index]*len(known))
            columns.extend(known)
            weights.extend(value.tolist())
        query = sparse.csr_matrix((np.asarray(weights, np.float32),
            (np.asarray(rows, np.int32), np.asarray(columns, np.int32))),
            shape=(len(histories), self.vocab_size), dtype=np.float32)
        scores = (query @ self.similarity).toarray()
        no_signal = np.max(scores, axis=1) <= 0
        scores += self.popularity_weight*self.popularity[None, :]
        scores[no_signal] = self.popularity
        scores[:, :3] = -1e8
        return scores

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        # Only trusted locally generated checkpoints are supported.
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise ValueError("Checkpoint does not contain an ItemKNN model")
        return value
