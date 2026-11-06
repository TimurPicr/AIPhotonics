# Передача роли C

- T0.1: src package, Python 3.13 `.venv`, зависимости установлены; точные версии — requirements-lock.txt.
- C1.1: `src.io.load_four_png`, `export_phase_png`, `load_phase_png`; строгие 128×128 L PNG.
- C1.2: `src.data.make_train_loader(batch_size, seed)`, бесконечный повторяемый поток float32 [B,4,128,128]; новый RNG на iterator, не на batch.
- C1.3: сохранены data/{val,test,ood,train_diagnostic} (64/64/16/4) и examples (8); z из AGENTS.md переданы явно через prepare_split(..., z_m=...). Manifest содержит seeds, SHA256 и версии NumPy/Pillow; проверено отсутствие общих PNG/источников. CLI по умолчанию ожидает config.CONFIG.z_m от A.
- C1.4: `src.evaluate.evaluate_four`, `aggregate_results`; постоянные реконструкции помечаются ошибками. `src.io.export_result(targets, phase, out, metadata=...)` перечитывает phase и reconstruction PNG.
- Проверки: `.venv/bin/python -m pytest tests/test_io.py -q` — 7 passed; полный suite — 25 passed.
- B: экспорт C импортирует `src.reconstruction.{reconstruct_four,normalize_recons}` только при вызове. Для validation `aggregate_results(results)['worst_plane_mean']['ssim']` и `['mean']['pearson']`.
- A: общий экспорт доступен для CLI baseline; возвращает словарь, сохраняемый в metrics.json. Значение `metadata` — произвольный JSON-совместимый словарь.

C2: src.infer и CLI src.evaluate подготовлены; --help и compileall проверены. Реальный сквозной запуск ожидает конфигурацию/физику A и checkpoint с согласованным протоколом B. Первоначальный initial.pt B имеет protocol=None: B должен пересоздать его после config.py с текущим protocol. C3: test/OOD не оценены, обученный checkpoint и сравнение A3 ещё отсутствуют. README содержит команды и ограничения; готовность C3 не заявляется.

Проверка экспорта C1.4 выполнена с искусственным оператором только в тесте: проверяет перечитывание квантованной фазы и расчёт метрик по сохранённым recon PNG, не заменяет физическую проверку A1.4.
