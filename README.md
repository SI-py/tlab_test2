# RL для последовательных рекомендаций

[Статья с результатами](article.pdf). Таблицы и рисунки находятся в `results/`.

GPT-2 обучается предсказывать следующий объект, затем дообучается методом RLOO по модели отклика CatBoost. На KION и Yambda сравниваются RL, контроль без градиента награды, ItemKNN и переранжирование. Для нейросетевых моделей используются три запуска и два отложенных периода оценки.

## Запуск экспериментов

Нужен Python 3.11. CUDA используется автоматически, если доступна.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/download_data.py
python scripts/download_yambda.py --artists
python -m rlrec.research --config configs/kion.json --output artifacts/kion
python -m rlrec.research --config configs/yambda.json --output artifacts/yambda
```

Параметры моделей и разбиения данных заданы в `configs/`. Отдельные стадии запускаются через `--stage prepare|sft|reward|develop|evaluate`. Для нового эксперимента используйте новый каталог `--output`.

Рекомендации по истории из JSON-массива исходных item ID:

```bash
python scripts/recommend.py --run artifacts/kion \
  --history-json history.json --cutoff 2021-08-09 --output recommendations.csv
```

История содержит объекты в порядке потребления до `--cutoff`. Для Yambda время задаётся относительными секундами.

Сборка таблиц из обученных запусков: `python scripts/summarize_research.py --kion artifacts/kion --yambda artifacts/yambda`. Скрипты дополнительных разборов находятся в `scripts/`. Рисунки и PDF можно пересобрать из сохранённых результатов командами `python scripts/plot_article.py` и `python scripts/article.py`.
