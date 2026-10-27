# Применение и воспроизведение детектора CSI

[Аналитический отчёт](../../research/motion_model_report.md) описывает задачу, методику и результаты. Этот документ содержит параметры использования и технические проверки.

## Применение

```python
import numpy as np
import torch
from experiments.final_motion.pipeline import MotionDetector

torch.set_num_threads(1)
detector = MotionDetector("experiments/motion_results/deployment.pt", device="cpu")
# amplitudes: неотрицательный float32 [N, 52, 400], сетка 100 Гц,
# поднесущие L-LTF [6:32] + [33:59], до любой нормализации.
probability = detector.predict(amplitudes)
movement = probability >= detector.threshold
```

Командный интерфейс принимает `.npy` с таким же представлением амплитуд:

```powershell
python -m experiments.final_motion.pipeline windows.npy --output predictions.npz
```

Веса, параметры нормализации и порог загружаются вместе. `.pt` также содержит архитектуру, предобработку, seed, частоту, размер окна, поднесущие, индексы обучающих окон, идентификаторы записей, параметры обучения и контрольные суммы кода. SHA-256 итогового файла — `c90de68eeabb6661c7eb956da5ea3371bdb5aa85b2be909fee8e93f1e5f09e6e`.

Полный набор результатов находится в [experiments/motion_results](../motion_results): 126 валидационных и 10 контрольных файлов весов в `checkpoints/`, прогнозы в `predictions/`, потери по 20 эпохам в `curves/`. В `.npz` поля `indices`, `truth` и `probability` связывают прогноз с исходным окном. [test_metrics.csv](../motion_results/test_metrics.csv) и [seed_summary.csv](../motion_results/seed_summary.csv) содержат результаты всех контрольных seed.

### Целостность данных и воспроизведение

Исходные CSV содержат 1 569 769 пакетов, из которых четыре поврежденных исключаются. Временная шкала включает шесть переполнений uint32. Сравнение разрывов в целых микросекундах устраняет 12 пограничных случаев, когда дробное представление интервала ровно 50 мс приводило к ложному превышению порога. Поэтому текущий набор содержит 3052 окна вместо прежних 3045; начала 3029 окон совпадают. Разница между старыми и текущими метриками включает изменение состава окон и не является чистой оценкой эффекта предобработки. Девять пробных обучений на прежнем наборе находятся [в отдельном архиве](../motion_pilot_float_timing/README.md) и не входят в текущие оценки.

Независимое восстановление всех 3052 окон из ZIP через расчет амплитуды `hypot` отличается от подготовленного массива не более чем на 3.8146973e-06 в единицах устройства. [source_audit.csv](../motion_results/source_audit.csv) содержит сверку по записям, [data_manifest.json](../motion_results/data_manifest.json) — контрольные суммы архивов и массива. [materials.csv](../motion_results/materials.csv) связывает исходные материалы с их контрольными суммами. Предыдущая сетка включает 256 обучений и 656 строк метрик; повторные расчеты подтверждают ее метрики и сохраненные прогнозы четырех архитектур.

Проверка текущих артефактов охватывает инференс 136 файлов весов, 176 строк метрик, статистики нормализации только по обучающим окнам, отсутствие пересечения записей между обучением и соответствующим контролем, неперекрывающиеся окна, метки и конечность весов. Максимальное расхождение повторных вероятностей на GPU — 0, между итоговой моделью на CPU и GPU — 6.377697e-06. Решения на 3DO совпадают. Условия замеров CPU и GPU доступны в [validation.json](../motion_results/validation.json).

Последовательность команд из корня проекта:

```powershell
python -m experiments.final_motion.prepare_data
python -m experiments.final_motion.audit_data
python -m experiments.final_motion.verify_signal
python -m experiments.final_motion.train
python -m experiments.final_motion.validate
python -m experiments.final_motion.analyze
python -m experiments.final_motion.write_report
python -m experiments.final_motion.verify_artifacts
```

`train` проверяет неизменность плана и кода и пропускает готовые запуски. Изменение параметров требует другой папки результатов. Для подготовки нужны `wallhack1.8k.zip` и `3DO.zip` в `datasets/`; для загрузки готовой модели архивы не требуются. Восстанавливаемый кэш `amplitudes.npy` хранится локально и исключен из Git. В репозиторий входят код, веса, прогнозы, таблицы и готовые PNG.
