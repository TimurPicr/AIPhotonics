# TIMUR-FORWARD: обучение на достижимых изображениях

Ветка: `timur`. База: `85e1267` (основной MVP и его анализ).

## Постановка

Меняется генерация целей: сохранённая предсказанная phase.png → четыре реконструкции Френеля → четыре PNG. Модель по-прежнему получает четыре изображения и предсказывает одну фазу. Архитектура, амплитудный протокол, нормировка и физический loss сохранены; эталонная фаза служит проверке достижимости, не новому supervised loss.

96 уникальных масок исходного best.pt взяты из outputs/test (64), outputs/ood (16), outputs/examples (8) и outputs/comparison/final/network (8). Это маски одной обученной сети, не произвольные фазы. Исходный checkpoint SHA256: `55ad50208b7545c1ea3f14485148fba089e6817d08eebfec9589eb42cb54412c`.

До генерации изображений маски разделены на train/val/test = 64/16/16, seed=42. В каждой исходной категории 16/4/4 маски. Повторы фаз и исходных sample IDs удаляются до split; исходные маски не пересекаются между наборами. Категория означает происхождение маски, не визуальный класс полученной реконструкции. Аугментаций нет.

## Запуск

```bash
python -m src.forward_data
python -m src.train --steps 500 --batch-size 8 --seed 42 --device cpu --train-manifest data/forward/train/manifest.json --val-manifest data/forward/val/manifest.json --checkpoint-dir checkpoints/timur_forward --run-dir runs/timur_forward
python -m src.train --resume checkpoints/timur_forward/last.pt --steps 1500 --batch-size 8 --seed 42 --device cpu --ssim-weight 0.1 --train-manifest data/forward/train/manifest.json --val-manifest data/forward/val/manifest.json --checkpoint-dir checkpoints/timur_forward --run-dir runs/timur_forward
```

Обучение с нуля из seed 42, batch=8, Adam lr=1e-3, бюджет 2000 шагов / 16 000 показов. Первые 500 шагов — MSE, следующие 1500 — MSE+0.1*(1−SSIM). Отбор checkpoint только по новым 16 val: худший средний SSIM плоскостей, затем Pearson. Исходные checkpoints не перезаписываются.

Сравниваются на одинаковом новом test: исходный best.pt, новая сеть, ближайшая train-четвёрка с её фазой и исходная фаза test-маски (контроль достижимости). Все метрики — общий PNG roundtrip. Исторические метрики старого test приводятся отдельно: набор целей изменился, разность с ними не является чистым эффектом обучения.

## Проверки и ограничения

- Тесты проверяют точное восстановление целей из исходной фазы, повторяемость генерации, отсутствие общих родителей/хешей между splits и повторяемость train-потока.
- CLI проверен на двух реальных train-step и продолжении из last.pt.
- Обобщение здесь проверяется только на 16 новых масках той же сети-генератора. Это пилот с одним seed; вывод на произвольные изображения по нему не делается.
- Полное основное обучение и итоговые метрики выполняются далее; test не используется для подбора.
