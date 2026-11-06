# Многоплоскостной ДОЭ

Четыре независимые цели 128×128 → U-Net → фаза → одно-FFT Френель → четыре PNG.
План и контракты: [AGENTS.md](AGENTS.md), [TASKS.md](TASKS.md).

## Старт и соглашения

Первый коммит C: 2026-11-06 22:12 Europe/Moscow; общий T0/дедлайн команды
не предоставлены. Плановый срок MVP — 7–9 часов, предел — 24 часа от общего T0. C — текущий coding-agent; имена A/B не предоставлены.
Источник требований: AGENTS.md и TASKS.md; PDF организаторов в репозитории отсутствуют.
Предположения: `amplitude = abs(U)`, независимое деление каждой плоскости на её максимум
с epsilon 1e-8, вход uint8/255. При изменении протокола требуется повторное обучение.
Параметры физики задаёт A в config.py, общую нормировку — src.reconstruction.
Seed train/val/test/OOD: 42/42001/42002/42003. Best выбирает B по максимальному
худшему среднему SSIM плоскостей validation, затем среднему Pearson. Test не используется для выбора.
Проверка B `select_device()` подтверждена локально: CPU, конечный комплексный FFT/backward.

## Установка

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Системный Python 3.15 не используется: окружение проекта — Python 3.13.
Физика принадлежит A, сеть и checkpoints — B, данные/IO/метрики/интеграция — C.
Техническая готовность и качество обучения фиксируются раздельно. `checkpoints/initial.pt` —
необученная сеть для сквозной проверки; обученный `best.pt` пока не передан B.

## Данные и проверка C1

```bash
python -m src.data --prepare-eval --val-size 64 --test-size 64 --ood-size 16 --z-m 0.0241 0.0253 0.0265 0.0277
python -m pytest tests/test_io.py -q
```

Явные z позволяют подготовить данные до интеграции конфигурации A. Без `--z-m`
используется `config.CONFIG.z_m`; конфигурация A интегрирована.
Данные уже сохранены: validation/test/OOD — 64/64/16, train_diagnostic — 4,
examples — 8. Категории идут циклически: геометрия, символы, полутона, смешанные.
В val/test по 16 четвёрок каждой категории. Четыре цели генерируются независимо.
Manifests содержат split, seed, source IDs/seeds, порядок z, версии и SHA256 PNG.
Загрузка manifest проверяет хеши. Генератор проверяет отсутствие общих исходников
и идентичных PNG между фиксированными наборами. Внешние данные не используются.
OOD удерживает треугольники, символы XYZ789 и частоты 12–20 вместо train 1–7.
Полутона включают цели с максимумом ниже 1. Train — бесконечный RNG-поток;
`next(iter(make_train_loader(batch_size, seed)))` возвращает float32 [B,4,128,128].

`src.io.export_result(targets, phase, out, metadata=...)` — общий путь для A/B:
сохранение фазы → чтение фазы → общий оператор A → сохранение recon → чтение recon → метрики.
Pearson постоянной цели — null/диагностика; постоянная реконструкция непостоянной
цели — null/ошибка. Отчёт сохраняет число валидных пар и ошибок; SSIM всегда
считает `data_range=1`, `win_size=7`, остальные настройки и версия записаны в JSON.

## Инференс и оценка

```bash
python -m src.infer --images examples/examples_0000/1.png examples/examples_0000/2.png examples/examples_0000/3.png examples/examples_0000/4.png --checkpoint checkpoints/best.pt --out outputs/example
python -m src.evaluate --manifest data/test/manifest.json --checkpoint checkpoints/best.pt --out outputs/test
python -m src.evaluate --manifest data/ood/manifest.json --checkpoint checkpoints/best.pt --out outputs/ood
```

На пример: phase.png, recon_1.png … recon_4.png, metrics.json. Оценка набора
создаёт summary.json: категории, каждая плоскость, среднее и худшее среднее плоскостей,
число примеров/пар/ошибок. CPU по умолчанию; forward замеряется после двух прогревов,
отдельно от экспорта. Pipeline включает forward и экспорт; чтение входов и загрузка
checkpoint выполняются до замера. Метрики и изображения повторяемы; время меняется.
Checkpoint проверяется API B `load_checkpoint(..., expected_protocol=...)`;
текущий контракт B — `current_protocol()`, то есть `config.DEFAULT_CONFIG.snapshot()`.
Несовместимые значения физики/нормировки отклоняются; правка комментариев config.py
не меняет совместимость. `initial.pt` содержит актуальный протокол.

## Текущий результат

Проверено на macOS arm64, Python 3.13, CPU; комплексный FFT/backward доступен.
Точные версии установленных пакетов: requirements-lock.txt (`pip install -r requirements-lock.txt`).
Полный suite: 42 passed, 1 skipped (Metal недоступен в среде C); C1: 7 passed. Проверены PNG/фаза, RNG, manifests,
известные Pearson/SSIM, вырожденные пары, JSON и путь экспорта на искусственном операторе.
Проверка искусственного оператора тестирует только IO; физическую корректность проверяет A.
C2 проверен с реальным Френелем A и `initial.pt` B: два CLI-запуска дали побайтно
одинаковые phase/recon PNG и одинаковые метрики. Независимое восстановление из phase.png
совпало с сохранёнными реконструкциями. Результаты: `outputs/c2_initial/verification.json`.
Максимальная ошибка квантования фазы: 0.0123201 рад (не более π/255).
На 8 examples необученной сети: средний Pearson 0.0267693, SSIM 0.0289239,
худший средний SSIM плоскости 0.0231847; 32/32 валидных пар, ошибок нет.
Подробности по категориям/плоскостям: `outputs/c2_initial/examples/summary.json`.
Это техническая проверка начальной сети, качество обучения пока не оценено.
C3 ожидает обученный best.pt B и сравнение A3; test/OOD не использовались.

Идеи проекта: U-Net/каналы глубины — Zeng 2026, обучение через восстановление —
DeepCGH 2020, PyTorch-оптика и Adam baseline — Peng 2020; независимые синтетические
2D-цели C — адаптация подхода Yan 2025. Код генератора, IO и метрик написан здесь;
чужой код и готовые оптические операторы не копировались. Ссылки и границы адаптации — AGENTS.md §11.

Повторение технической проверки C2 без обученного checkpoint:

```bash
python -m src.infer --images examples/examples_0000/1.png examples/examples_0000/2.png examples/examples_0000/3.png examples/examples_0000/4.png --checkpoint checkpoints/initial.pt --out outputs/reproduce_initial
python -m src.evaluate --manifest examples/manifest.json --checkpoint checkpoints/initial.pt --out outputs/reproduce_examples
```

Общая подвыборка A3: `data/val/comparison_manifest.json`, первые 8 val (две четвёрки
каждой категории). Начальная сеть уже оценена: `outputs/comparison/initial/summary.json`;
mean Pearson=0.0335856, SSIM=0.0402451, worst-plane mean SSIM=0.0364030, ошибок 0/32.
После фиксации best.pt обученная сеть оценивается на тех же целях:

```bash
python -m src.evaluate --manifest data/val/comparison_manifest.json --checkpoint checkpoints/best.pt --out outputs/comparison/network
```
