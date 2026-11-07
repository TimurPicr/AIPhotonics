# Нейросетевой синтез многоплоскостного ДОЭ

Новая обучающая выборка: **55 000 целевых PNG = 13 750 четвёрок**
(40 000 синтетических, 5 000 COCO, 10 000 кадров Vimeo90K).
Генерация, источники, проверки и ленивая загрузка описаны в [DATASET55K.md](DATASET55K.md).

Четыре независимых PNG → один проход обученной U-Net → одна фазовая маска → одно-FFT Френель на четырёх расстояниях → PNG и Pearson/SSIM.

Обученный checkpoint — `checkpoints/best.pt`, шаг **1400**, SHA256:
`55ad50208b7545c1ea3f14485148fba089e6817d08eebfec9589eb42cb54412c`.

Технический конвейер проверен; качество восстановления ограничено, особенно на OOD.
Test: Pearson **0.14464**, SSIM **0.11864** (64 четвёрки).
OOD: Pearson **0.08278**, SSIM **0.02409** (16 четвёрок).
Каждая плоскость, категории, начальная сеть, Adam и время — в [RESULTS.md](RESULTS.md).

## Установка и проверка

Проверенная среда C: macOS arm64, Python 3.13, PyTorch 2.15.0, CPU, 2 потока.
Точные версии сохранены в `requirements-lock.txt`; `requirements.txt` задаёт основные зависимости.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

На CPU тест Metal пропускается. Обучение B проверено на Metal/Python 3.12.14;
версии, seed и параметры запусков находятся в `runs/*/config.json`.

## Инференс

Все входы — PNG режима `L`, 128×128. Порядок файлов соответствует расстояниям z.

```bash
python -m src.infer --images examples/examples_0000/1.png examples/examples_0000/2.png examples/examples_0000/3.png examples/examples_0000/4.png --checkpoint checkpoints/best.pt --out outputs/reproduce_example
```

Результат: `phase.png`, `recon_1.png` … `recon_4.png`, `metrics.json`.
Сначала сохраняется и перечитывается фаза, затем рассчитываются и перечитываются реконструкции.
Метрики считаются по PNG; JSON содержит checkpoint hash, протокол, native pixel pitches,
версию библиотек, seed, ошибки и раздельное время forward/полного пути.
Чистый нейросетевой режим выполняет один forward, без уточнения фазы.

## Фиксированные наборы и оценка

Данные уже включены: 64 val, 64 test, 16 OOD, 4 train_diagnostic и 8 examples.
В val/test по 16 четвёрок геометрии, символов, полутонов и смешанных целей.
Четыре изображения генерируются независимо. Внешние данные не используются.
Manifests фиксируют source IDs/seeds, SHA256 PNG, split, версии генерации и порядок z.
Хеши проверяются при загрузке; фиксированные исходники и PNG между наборами не пересекаются.
Train генерируется потоком с локальным RNG без сброса на каждом batch.
OOD удерживает треугольники, символы XYZ789 и частоты 12–20 против train 1–7.

```bash
python -m src.data --prepare-eval --val-size 64 --test-size 64 --ood-size 16
python -m src.evaluate --manifest data/test/manifest.json --checkpoint checkpoints/best.pt --out outputs/reproduce_test
python -m src.evaluate --manifest data/ood/manifest.json --checkpoint checkpoints/best.pt --out outputs/reproduce_ood
python -m src.evaluate --manifest examples/manifest.json --checkpoint checkpoints/best.pt --out outputs/reproduce_examples
```

`summary.json` содержит каждую плоскость, категории, среднее, худшее среднее плоскостей,
число валидных пар и ошибки с sample ID. Постоянная цель: Pearson `null`/диагностика;
постоянная реконструкция непостоянной цели: `null`/ошибка. SSIM: `data_range=1`,
`win_size=7`, без Gaussian weights, sample covariance, K1=0.01, K2=0.03.
JSON не допускает NaN/Infinity.

Финальные PNG и отчёты находятся в `outputs/{test,ood,examples}/`.
Коллажи примеров: `outputs/examples/examples_0000/collage.png` … `examples_0007/collage.png`.
Аудит повторного распространения всех test/OOD фаз: `outputs/final_verification.json`.
Одинаковая среда даёт одинаковые PNG/метрики; время и численные результаты разных платформ могут отличаться.

## Сравнение с baseline

Общая подвыборка — первые восемь val, по две каждой категории, зафиксирована до финального выбора модели.
Сравнение начальной сети, обученной сети и Adam выполняется на одинаковых целях и PNG-протоколе.

```bash
python -m src.compare --out outputs/reproduce_comparison --device cpu --threads 2 --steps 100 --seed 42
python -m src.baseline --images examples/examples_0000/1.png examples/examples_0000/2.png examples/examples_0000/3.png examples/examples_0000/4.png --steps 300 --seed 42 --out outputs/reproduce_baseline
```

Все фазы/PNG локального сравнения C — `outputs/comparison/final/`.
Baseline оптимизирует фазу Adam; сеть и baseline используют один оператор и нормировку.
Гибрид в финальную оценку не включён.

## Обучение и checkpoint

U-Net ширины 16 со skip-связями, 4→1 канал, выход `π*tanh(raw)`, interleave_factor=1.
Физический FFT фиксирован. Обучение без фазовых меток: 500 шагов MSE,
затем 1500 шагов MSE+0.1·(1−SSIM), Adam lr=1e-3, batch=8, seed=42.
Checkpoint выбран на шаге 1400 из бюджета 2000 по худшему среднему SSIM четырёх
плоскостей validation, затем среднему Pearson. Checkpoint видел 11 200 четвёрок;
полный бюджет выбранной ветки — 16 000 показов. Test/OOD не участвовали в выборе.

```bash
python -m src.train --steps 500 --batch-size 8 --seed 42 --device auto --checkpoint-dir checkpoints/repro --run-dir runs/repro
python -m src.train --resume checkpoints/repro/last.pt --steps 1500 --batch-size 8 --seed 42 --device auto --ssim-weight 0.1 --checkpoint-dir checkpoints/repro --run-dir runs/repro
```

`initial.pt` — согласованное начальное состояние; `best.pt` — выбранный результат;
`last.pt` — шаг 2000 с optimizer/RNG для продолжения. `pilot_best.pt` и `mse_best.pt` —
контрольные MSE-модели. Загрузка: `src.train.load_checkpoint(path, expected_protocol=current_protocol())`.
Несовместимые значения физики/нормировки вызывают ошибку.
Loss/архитектурные сравнения и CPU-проверка выбора — `runs/{mvp,ssim,interleave}/`.

## Физика, предположения и границы

Источник протокола — AGENTS.md/TASKS.md; PDF организаторов в репозитории отсутствуют.
Рабочее предположение: `amplitude = abs(U)`, нормировка отдельно для каждой
плоскости по максимуму с epsilon 1e-8; цели uint8/255. Абсолютная яркость между плоскостями
не сохраняется. Если полное условие требует интенсивность, нужно изменить общий config,
повторить обучение и оценку. Подбор усиления по целям не используется.

Физика хранится в `config.py`: λ=532e-9 м, шаг SLM=10e-6 м,
z=[0.0241,0.0253,0.0265,0.0277] м. Собственный одно-FFT Френель использует
центрированные координаты/FFT и norm=backward; глобальный фазовый множитель сохранён.
Цели сравниваются на native сетках каждой плоскости: dx_out≈[10.0166,10.5153,11.0141,11.5128] мкм.
Фаза кодируется `0→−π`, `255→+π`; все рабочие PNG — `L`, 128×128.

Идеи: U-Net/каналы глубины — Zeng 2026, обучение через восстановление — DeepCGH 2020,
PyTorch-оптика и Adam — Peng 2020, независимые 2D-цели — адаптация Yan 2025.
Источники и границы адаптации — [AGENTS.md §11](AGENTS.md). Чужой код не копировался.
Роли: A — физика/baseline, B — сеть/обучение, C — данные/интеграция/сдача.
Общий T0, реальные имена всех участников и дедлайн не предоставлены; первый коммит C —
2026-11-06 22:12 Europe/Moscow. Статусы и приёмка — [TASKS.md](TASKS.md), [B_STATUS.md](B_STATUS.md), [C_STATUS.md](C_STATUS.md).

Состав сдачи: исходники/tests, config, генератор и manifests/PNG, checkpoints,
логи обучения, `RESULTS.md`, результаты и коллажи. Исторический `outputs/c2_initial/`
относится к раннему initial checkpoint; для актуального сравнения используется `outputs/comparison/final/`.

## Архив сдачи

Из зафиксированной ревизии репозитория:

```bash
git archive --format=zip --output=/tmp/AIPhotonics-mvp.zip HEAD
```

Архив содержит исходники, данные, checkpoints, логи и результаты, без `.git` и `.venv`.
В извлечённом каталоге выполняются те же команды установки, тестов и инференса.

## Приёмка

C1–C3 завершены. Чистое извлечение архивированной ревизии прошло `pytest`:
**50 passed, 1 skipped** (Metal недоступен в среде C). Данные восстановились с
идентичными manifests; инференс и оценки test/OOD/examples дали **445 одинаковых PNG**,
метрики 89 запусков и одинаковые агрегаты/категории. Использовано существующее
окружение Python 3.13 с зафиксированными зависимостями; исходники импортировались
из извлечения. Подробный журнал — [outputs/release_verification.json](outputs/release_verification.json).
A подтвердил физику, B — загрузку/выбор checkpoint. Ограничения качества приведены в RESULTS.md.

Разбор расхождения с Yan 2025 и validation-диагностика: [ANALYSIS_YAN2025.md](ANALYSIS_YAN2025.md).
