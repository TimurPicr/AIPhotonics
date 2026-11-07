# Обучение и сравнение фазовых масок

```bash
python -m src.train --train-manifest runs/overfit9_balanced/datasets/train.json --eval-manifest runs/overfit9_balanced/datasets/probe.json --epochs 50 --batch-size 3 --out runs/hologram_9 --device cpu
```

Задайте манифесты, эпохи и новый каталог `--out`. Для эксперимента на самих
обучающих примерах замените `--eval-manifest ...` на `--evaluate-train`.
Настройки модели доступны через `--rounds` и `--plane-weights`. Веса loss
задаются через `--alpha` и `--beta`.

После каждой завершённой эпохи атомарно обновляется один файл `last.pt`.
Для оценки остановленного прогона используйте этот checkpoint.

На выходе сохраняются `training.jsonl` и `comparison/report.md`.
График среднего loss по эпохам сохраняется в `comparison/loss.png` и
встраивается в отчёт. Повторная оценка строит его по `training.jsonl` рядом
с checkpoint.
Постобработка один раз сравнивает сохранённые веса с Adam и Global MP-GS
по SSIM и Pearson.
В `comparison/summary.json` есть метрики по каждой плоскости и параметры
сравнения. По умолчанию Adam и Global MP-GS выполняют по 100 итераций на
четвёрку. Бюджеты задаются через `--adam-steps` и `--gs-iterations`. Adam
оптимизирует фазу с той же функцией потерь, что и сеть. Метрики считаются
после 8-битного квантования фазы и реконструкций на общих парах.

Повторная оценка загружает веса и параметры baseline из checkpoint.

```bash
python -m src.evaluate --checkpoint runs/hologram_9/last.pt --eval-manifest runs/overfit9_balanced/datasets/probe.json --out runs/hologram_9_probe
```

Полный сокращённый набор — `data/train5.5k/train/manifest.json`. В нём
1 000 синтетических, 250 видео и 125 случайных четвёрок.
Для обучения с отдельной проверкой используйте `train.json` и `val.json`
из `runs/train55k_10pct/datasets/`. Оба манифеста ссылаются на сокращённый набор.
