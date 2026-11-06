# Передача роли B

- B1.1–B1.4 готовы и отправлены; 18 тестов модели, loss, train-loop и checkpoint прошли.
- Среда B: Python 3.12.14, PyTorch 2.15.0, CPU; комплексный FFT/backward работает.
- Загрузка: `from src.train import load_checkpoint, current_protocol`; `model, metadata = load_checkpoint(path, expected_protocol=current_protocol())`.
- **Согласование с A:** конфигурация — `config.DEFAULT_CONFIG`, расстояния — `.distances_m`, протокол — `.snapshot()`. Протокол checkpoint теперь содержит эти значения, а не хеш исходника; C следует использовать `current_protocol()` в `src.infer.protocol_snapshot`.
- `checkpoints/initial.pt` — необученная сеть для C2, с актуальным протоколом A. Она не является результатом обучения.
- Код validation использует `src.io.export_result` C и удаляет временные PNG. Критерий best: худший средний SSIM плоскостей, затем средний Pearson; тесты validation прошли.
- B2/B3 ожидают проверенный Френель A1.4 и baseline A2.2; физическое обучение пока не запускалось.

Обучаем веса через `L = mean((normalize(abs(P_z(exp(i*phi)))) - T)**2)` по четырём z; режим интенсивности определяется общим протоколом.
Основа: [Zeng — U-Net](https://doi.org/10.3390/photonics13030252), [DeepCGH — обучение по восстановлению](https://doi.org/10.1364/OE.399624), [Peng — дифференцируемая оптика](https://doi.org/10.1145/3414685.3417802).
