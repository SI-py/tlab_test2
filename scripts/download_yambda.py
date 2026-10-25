#!/usr/bin/env python3
"""Download the official Yambda-50M multi-event file and optional artist metadata."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/multi_event.parquet"
ARTIST_URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/artist_item_mapping.parquet"


def obtain(destination, url=URL):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using {destination}")
        return destination
    fd, temporary = tempfile.mkstemp(prefix="yambda-download-", dir=destination.parent)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "recsys-research"})
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, output, 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Saved {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/raw/yambda50m_multi.parquet")
    parser.add_argument("--artists", action="store_true", help="Also download the static artist mapping")
    args = parser.parse_args()
    obtain(args.output)
    if args.artists:
        obtain(args.output.parent / "yambda_artist_item_mapping.parquet", ARTIST_URL)


if __name__ == "__main__":
    main()
