# Передача роли C

- T0.1: src package, Python 3.13 `.venv`, зависимости установлены; точные версии — requirements-lock.txt.
- C1.1: `src.io.load_four_png`, `export_phase_png`, `load_phase_png`; строгие 128×128 L PNG.
- C1.2: `src.data.make_train_loader(batch_size, seed)`, бесконечный повторяемый поток float32 [B,4,128,128]; новый RNG на iterator, не на batch.
- C1.3: генератор manifests готов, ожидает публичную конфигурацию A для порядка z.
- C1.4: `src.evaluate.evaluate_four`, `aggregate_results`; постоянные реконструкции помечаются ошибками. `src.io.export_result(targets, phase, out, metadata=...)` перечитывает phase и reconstruction PNG.
- Проверки: `.venv/bin/python -m pytest tests/test_io.py -q` — 5 passed.
- B: экспорт C импортирует `src.reconstruction.{reconstruct_four,normalize_recons}` только при вызове. Для validation `aggregate_results(results)['worst_plane_mean']['ssim']` и `['mean']['pearson']`.
- A: общий экспорт доступен для CLI baseline; возвращает словарь, сохраняемый в metrics.json. Значение `metadata` — произвольный JSON-совместимый словарь.

C2 ожидает конфигурацию A и checkpoint с согласованным протоколом B. Test/OOD не оценены.
