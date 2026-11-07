# Обучение и сравнение фазовых масок

Текущий набор в `data/`: `train.npy` — 8 000, `validation.npy` — 1 000,
`test.npy` — 1 000 четвёрок (80/10/10). Формат — `uint8 [N, 128, 128, 4]`,
порядок каналов соответствует расстояниям в `config.py`. Набор сокращён
случайной выборкой без возвращения с seed 42/42001/42002 соответственно.
Значения пикселей сохранены; точных совпадений изображений между частями нет.
Меньший набор может повлиять на качество обобщения.

Поддерживается прямое чтение `.npy` через memory mapping без создания PNG:

```bash
python -m src.train --train-manifest data/train.npy --eval-manifest data/validation.npy --epochs 50 --batch-size 4 --threads 6 --out outputs/train10k --device cpu
```

На CPU используется `channels_last`, коэффициенты Френеля кешируются;
модель, float32/complex64, loss, batch size и шаг оптимизатора сохранены.
По умолчанию обучение использует до 6 потоков; `--threads` задаёт их явно.
После каждой эпохи выполняется validation. В `best.pt` атомарно сохраняются
только лучшие веса: максимальный худший средний SSIM среди четырёх плоскостей,
при равенстве — средний Pearson. Метрики выбора учитывают 8-битное
квантование фазы и реконструкций; неопределённый Pearson учитывается как 0
с явным числом определённых пар. После обучения запускается сравнение с
Adam и Global MP-GS на всём `--eval-manifest`, которое занимает отдельное время.

Пример запуска с PNG-манифестами:

```bash
python -m src.train --train-manifest runs/overfit9_balanced/datasets/train.json --eval-manifest runs/overfit9_balanced/datasets/probe.json --epochs 50 --batch-size 3 --out runs/hologram_9 --device cpu
```

Задайте манифесты, эпохи и новый каталог `--out`. Для эксперимента на самих
обучающих примерах замените `--eval-manifest ...` на `--evaluate-train`.
Настройки модели доступны через `--rounds` и `--plane-weights`. Веса loss
задаются через `--alpha` и `--beta`.

Файл `best.pt` обновляется только при улучшении validation.
Для оценки остановленного прогона используйте этот checkpoint.
В `training.jsonl` пишутся train loss, validation loss и метрики каждой эпохи;
итоговое сравнение использует лучшую сохранённую модель.

При `--backend mlx` по умолчанию сохраняются только `last.pt` и
`training.jsonl`, без валидации численными методами. Чтобы включить
сравнение сети с Global MP-GS после каждой эпохи, добавьте
`--validate-every-epoch`; результаты сохраняются в `validation.json`.
Для отдельного сравнения после обучения используйте `src.evaluate`.

При backend Torch без `--validate-every-epoch` на выходе сохраняются
`training.jsonl` и `comparison/report.md`.
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
python -m src.evaluate --checkpoint runs/hologram_9/best.pt --eval-manifest runs/overfit9_balanced/datasets/probe.json --out runs/hologram_9_probe
```

Ранее использовавшийся сокращённый набор — `data/train5.5k/train/manifest.json`. В нём
1 000 синтетических, 250 видео и 125 случайных четвёрок.
Для обучения с отдельной проверкой используйте `train.json` и `val.json`
из `runs/train55k_10pct/datasets/`. Оба манифеста ссылаются на сокращённый набор.

## Динамические графики

Во втором терминале запустите отдельный монитор; обучение и инференс не меняются:

```bash
python -m src.monitor --run runs/hologram_9 --val-manifest path/to/val.json --test-manifest path/to/test.json --device cpu
```

Для NumPy-набора используйте `--run outputs/train10k
--val-manifest data/validation.npy --test-manifest data/test.npy`.

Окно Matplotlib обновляет loss, SSIM и CC (Pearson) для train/val/test при
появлении нового `best.pt`. Train берётся из checkpoint; все три loss
пересчитываются на его фиксированных весах через `reconstruction_loss`,
метрики — через `plane_metrics`, без квантования. Это оценочный train loss,
он отличается от среднего loss по шагам оптимизатора в `training.jsonl`.
CC усредняется по определённым парам, их число видно в окне и журнале.

Монитор ждёт первую эпоху, сохраняет оценки в `monitor.jsonl` внутри `--run`
и восстанавливает графики при повторном запуске. Если оценка медленнее
обучения, промежуточные checkpoint могут быть пропущены. Графики монитора
показывают оценки обновлений лучшей модели; все эпохи остаются в `training.jsonl`. `--interval` задаёт интервал проверки в секундах (по
умолчанию 5). Нужен графический backend Matplotlib; без окна используйте
`--once`, чтобы оценить текущий checkpoint и сохранить `monitor.png`.

Train/val/test должны иметь непересекающиеся исходники. Test, который
просматривается при настройке модели, следует считать диагностическим;
для финальной проверки нужен отдельный отложенный набор.
