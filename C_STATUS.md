# Передача роли C

- T0.1: src package, Python 3.13 `.venv`, зависимости установлены; точные версии — requirements-lock.txt.
- C1.1: `src.io.load_four_png`, `export_phase_png`, `load_phase_png`; строгие 128×128 L PNG.
- C1.2: `src.data.make_train_loader(batch_size, seed)`, бесконечный повторяемый поток float32 [B,4,128,128]; новый RNG на iterator, не на batch.
- C1.3: сохранены data/{val,test,ood,train_diagnostic} (64/64/16/4) и examples (8); z из AGENTS.md переданы явно через prepare_split(..., z_m=...). Manifest содержит seeds, SHA256 и версии NumPy/Pillow; проверено отсутствие общих PNG/источников. CLI по умолчанию ожидает config.CONFIG.z_m от A.
- C1.4: `src.evaluate.evaluate_four`, `aggregate_results`; постоянные реконструкции помечаются ошибками. `src.io.export_result(targets, phase, out, metadata=...)` перечитывает phase и reconstruction PNG.
- Проверки: `.venv/bin/python -m pytest tests/test_io.py -q` — 7 passed; полный suite — 25 passed.
- B: экспорт C импортирует `src.reconstruction.{reconstruct_four,normalize_recons}` только при вызове. Для validation `aggregate_results(results)['worst_plane_mean']['ssim']` и `['mean']['pearson']`.
- A: общий экспорт доступен для CLI baseline; возвращает словарь, сохраняемый в metrics.json. Значение `metadata` — произвольный JSON-совместимый словарь.

C2.1/C2.2 завершены: подключён current_protocol() B, CLI проверен с initial.pt и реальным Френелем A. Два запуска дали побайтно одинаковые PNG и одинаковые метрики. Тест независимо распространяет перечитанную phase.png, квантует recon и сверяет их с файлами. Несовместимый протокол intensity отклоняется.

- Полный suite: 41 passed, 1 skipped (Metal недоступен в среде C).
- Результаты C2: outputs/c2_initial/; verification.json содержит хеши повторных PNG и отдельно float-фазу/PNG-фазу/PNG-реконструкции.
- 8 examples, initial network: mean Pearson=0.0267693, SSIM=0.0289239, worst-plane mean SSIM=0.0231847, 32/32 валидных пар, ошибок нет.
- C3 ожидает обученный best.pt B и результаты сравнения A3. Test/OOD не оценены.
- README обновлён реальными командами и ограничениями. C3 не объявляется завершённым.
