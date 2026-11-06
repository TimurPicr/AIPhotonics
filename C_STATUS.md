# Передача роли C

- T0.1, C1 и C2 завершены: PNG/фаза, поток независимых целей, фиксированные manifests, Pearson/SSIM, общий квантованный экспорт и CLI.
- Python 3.13, PyTorch 2.15.0, CPU macOS arm64; точные зависимости — requirements-lock.txt.
- Данные: 64 val, 64 test, 16 OOD, 4 train_diagnostic, 8 examples. Хеши проверены; фиксированные исходники/PNG между наборами не пересекаются.
- Финальный checkpoint B3.2: шаг 1400, SHA256 `55ad50208b7545c1ea3f14485148fba089e6817d08eebfec9589eb42cb54412c`. Протокол — `config.DEFAULT_CONFIG.snapshot()` через `src.train.current_protocol()`.
- A3 завершён: исходный отчёт A — outputs/comparison/a_summary.json. C повторил все три метода на одном локальном CPU: outputs/comparison/final/summary.json, сохранены фазы и реконструкции каждого метода. На 8 val mean SSIM initial/network/Adam = 0.05963/0.16193/0.14239, Pearson = 0.12182/0.17927/0.27771. Adam: 100 шагов, один старт, seed 42. Метрики/время разных платформ могут отличаться.
- C3.1 завершён. Test: 64 четвёрки, 256 пар, Pearson 0.14464, SSIM 0.11864, worst-plane mean SSIM 0.10198. OOD: 16 четвёрок, 64 пары, Pearson 0.08278, SSIM 0.02409, worst-plane mean SSIM 0.01974. Все 320 Pearson определены, ошибок нет.
- Выбор модели/экспериментов завершён до test/OOD; после финальной оценки веса не менялись.
- Все test/OOD PNG повторно прочитаны, распространены из phase.png и сверены с recon PNG; метрики и агрегаты пересчитаны. Аудит: outputs/final_verification.json.
- 8 финальных examples и коллажи — outputs/examples/. Исторический outputs/c2_initial относится к checkpoint до B3.1; актуальные initial-результаты — outputs/comparison/final/initial_network/.
- C3.2 завершён: README, RESULTS.md, данные/checkpoints/логи/PNG включены в архивируемую ревизию. C3.3 — проверка чистого извлечения архива. Полный suite: 50 passed, 1 skipped (Metal недоступен в среде C).

Общий экспорт для A/B: `src.io.export_result(targets, phase, out, metadata=...)`. Для validation: `aggregate_results(results)['worst_plane_mean']['ssim']`, затем `['mean']['pearson']`. Постоянные реконструкции непостоянных целей записываются как ошибки; JSON не допускает NaN/Infinity.
