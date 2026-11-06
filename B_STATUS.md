# Передача роли B

- B1.1–B1.4 готовы и отправлены; 20 тестов модели, loss, train-loop и checkpoint прошли.
- Среда B: Python 3.12.14, PyTorch 2.15.0. CPU и Metal проверены; полный градиент U-Net/четырёх Френелей работает на Metal. `--device auto` проверяет ускоритель и откатывается на CPU; RNG Metal сохраняется/восстанавливается.
- Загрузка: `from src.train import load_checkpoint, current_protocol`; `model, metadata = load_checkpoint(path, expected_protocol=current_protocol())`.
- **Согласование с A:** конфигурация — `config.DEFAULT_CONFIG`, расстояния — `.distances_m`, протокол — `.snapshot()`. Протокол checkpoint теперь содержит эти значения, а не хеш исходника; C следует использовать `current_protocol()` в `src.infer.protocol_snapshot`.
- `checkpoints/initial.pt` — необученная сеть для C2, с актуальным протоколом A. Она не является результатом обучения.
- Код validation использует `src.io.export_result` C и удаляет временные PNG. Критерий best: худший средний SSIM плоскостей, затем средний Pearson; тесты validation прошли.
- B2 завершён: smoke 20 шагов, MSE 0.12327→0.04501; overfit 200 шагов/4 четвёрки, MSE 0.10344→0.02438, PNG SSIM 0.04382→0.11759. Диагностика — `runs/diagnostics/`.
- B3.1: 500 шагов/4000 показов, 64 val; лучший шаг 450, средний SSIM 0.10452, Pearson 0.13996, худший средний SSIM 0.09209. Проверены 500 конечных обновлений и выбор best из логов. Пилот — `checkpoints/pilot_best.pt`, `runs/mvp/`.
- B3.2 ещё не завершён: O3 `--interleave-factor 4` проверяется при тех же 500 шагах, данных и MSE. Вход `[B,4,128,128]` переставляется в `[B,64,32,32]`, выход возвращается в одну фазу 128×128. Прежние checkpoints совместимы; 25 затронутых тестов прошли. Финальный `best.pt` пока не публикуется.
- C/A: `initial.pt` пересоздан самим train-loop из seed=42; перед итоговым сравнением пересчитать initial по окончательно выбранной архитектуре и SHA256 checkpoint.

Обучаем веса через `L = mean((normalize(abs(P_z(exp(i*phi)))) - T)**2)` по четырём z; режим интенсивности определяется общим протоколом.
Основа: [Zeng — U-Net](https://doi.org/10.3390/photonics13030252), [DeepCGH — обучение по восстановлению](https://doi.org/10.1364/OE.399624), [Peng — дифференцируемая оптика](https://doi.org/10.1145/3414685.3417802).
