# Передача роли B

**B1–B3 завершены; A3/C3 разблокированы.** Финальные веса — [checkpoints/best.pt](checkpoints/best.pt), SHA256 `55ad50208b7545c1ea3f14485148fba089e6817d08eebfec9589eb42cb54412c`.

- Модель: `PhaseUNet(base_channels=16, interleave_factor=1)`, четыре цели → одна фаза `π*tanh(raw)`. Interleave=4 проверен и отклонён по validation.
- Обучение на Metal: 500 шагов MSE, затем 1500 шагов `MSE + 0.1*(1-SSIM)`, всего 16 000 показов четвёрок. По фиксированному критерию выбран **шаг 1400**, а не последний шаг.
- Повторная CPU-проверка на 64 validation-четвёрках: средний **SSIM 0.12606**, **Pearson 0.16699**; худший средний SSIM плоскости **0.10873**. Все 256 пар валидны. Необученная сеть: SSIM 0.04731, Pearson 0.11487. Качество реконструкции остаётся низким; сильная точность не заявляется.
- Все **50 тестов A/B/C** прошли, включая Metal RNG, реальный Френель, PNG и checkpoint. CLI проверен и с обученными весами. Среда B: Python 3.12.14, PyTorch 2.15.0; остальные версии/протокол сохранены в конфигурациях запусков.
- `initial.pt` соответствует архитектуре и начальному состоянию выбранного обучения; `last.pt` содержит optimizer/RNG шага 2000 для продолжения. `pilot_best.pt` и `mse_best.pt` — контрольные MSE-модели бюджетов 500/1000.
- История/сравнения: [диагностика](runs/diagnostics/summary.json), [MSE-контроль](runs/mvp/summary.json), [interleave](runs/interleave/summary.json), [выбранный loss и CPU-проверка](runs/ssim/summary.json). Временные веса, PNG и установочный кеш удалены.
- A/C: использовать финальный `best.pt` и текущий `initial.pt`; сверять SHA256 в сравнении. Выбор сделан только по validation; test/OOD и итоговое сравнение с baseline выполняет C/A.

Оптика и нормировка берутся из A: `config.DEFAULT_CONFIG.snapshot()`. C использует тот же `current_protocol()` и экспорт с повторным чтением phase/reconstruction PNG.

Для загрузки: `from src.train import load_checkpoint, current_protocol`, затем `model, metadata = load_checkpoint(path, device="cpu", expected_protocol=current_protocol())`.

Воспроизведение выбранной схемы:

```bash
python -m src.train --steps 500 --batch-size 8 --seed 42 --device auto --checkpoint-dir checkpoints/repro --run-dir runs/repro
python -m src.train --resume checkpoints/repro/last.pt --steps 1500 --batch-size 8 --seed 42 --device auto --ssim-weight 0.1 --checkpoint-dir checkpoints/repro --run-dir runs/repro
```

При продолжении готового `last.pt` число `--steps` означает дополнительные обновления; явно указывать `--ssim-weight 0.1`.

Текущий режим: `R_k = |P_zk(exp(i*phi))|`, `L = mean((normalize(R_k)-T_k)^2) + 0.1*(1-SSIM)` по четырём z. SSIM использует окно 7 и sample covariance, как метрика C. Амплитуда остаётся рабочим предположением до проверки полного кейса.

Основа: [Zeng — U-Net](https://doi.org/10.3390/photonics13030252), [DeepCGH — обучение по восстановлению](https://doi.org/10.1364/OE.399624), [Peng — дифференцируемая оптика](https://doi.org/10.1145/3414685.3417802).
