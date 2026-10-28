# Аугментации CSI-моделей и перенос на Atheros

Эксперимент проверяет четыре политики аугментации и повторно обученный контроль на одной архитектуре TCN. Все 75 обучений используют только Wallhack. Days и OPERAnet участвуют только в оценке замороженных моделей. Подробные результаты, графики, ограничения и ссылки на веса находятся в [аналитическом отчёте](../../research/augmentation_report.md).

Дополнительно проверены LogReg, LogReg + PCA-8 с теми же политиками и LightGBM без аугментаций: 165 классических моделей. Условия этой серии отдельно зафиксированы в [classical_protocol.md](classical_protocol.md). Для нормализованной классической цепочки общее усиление сокращается; применение его после нормализации не использовалось как подмена аппаратного эффекта.

## Воспроизведение

Нужны зависимости основного проекта и `experiments/robustness/requirements.txt`, CUDA для предусмотренного обучения и подготовленные данные предыдущего исследования:

- `experiments/motion_results/amplitudes.npy` и `windows.csv`;
- `tmp/robustness_cache/days.npy` и `opera.npy`;
- `experiments/robustness/results/days_windows.csv` и `opera_windows.csv`;
- прежние прогнозы для сопоставления с LogReg и исходной TCN.

```powershell
python -m experiments.augmentation.verify transforms
python -m experiments.augmentation.run train
python -m experiments.augmentation.run evaluate
python -m experiments.augmentation.analyze
python -m experiments.augmentation.verify final
python -m experiments.augmentation.classical train
python -m experiments.augmentation.classical evaluate
python -m experiments.augmentation.analyze_classical
```

План, код и входные данные фиксируются контрольными суммами до обучения. При повторном запуске готовые обучения используются только при совпадении плана и обучающих индексов. Изменение политик требует отдельной папки результатов. Выбор `combined` зафиксирован по development-доменам Wallhack до внешних оценок, см. [locked_selection.json](results/locked_selection.json). Он не означает разрешение использовать эту модель во всех средах: проверка сохранения качества на OPERAnet не пройдена.

Все CSV и JSON записываются с окончаниями строк LF. Большие восстановимые массивы очищенного сигнала находятся в игнорируемой папке `tmp/augmentation`. Код построения графиков не входит в публикацию; численные результаты и готовые PNG сохраняются.

Материалы классической серии находятся в `results/classical/`: 165 предикторов `.joblib`, метаданные, 231 файл прогнозов, таблицы метрик, проверка сохранения и отдельный выбор LogReg по development-данным. Параметры LogReg и PCA обучаются только на Wallhack. Внешние оценки не участвуют в выборе политики. Для инференса классических `.joblib` используется та же команда `experiments.augmentation.infer`.

## Модели и прогноз

`results/checkpoints/` содержит 75 предикторов `.joblib`, те же веса и статистики в `.pt` и метаданные разбиения `.json`. Имена начинаются с `cv_*`, `domain_holdout` или `all_wallhack`. Последняя группа предназначена для прямого переноса после обучения на всех 44 записях Wallhack.

```powershell
python -m experiments.augmentation.infer windows.npy --model experiments/augmentation/results/checkpoints/all_wallhack__combined__s42.joblib --output predictions.npz --device cpu
```

Вход — неотрицательная амплитуда `float32 [N,52,400]`, сетка 100 Гц, до очистки и нормализации. Согласование тонов и времени для Atheros и Intel должно повторять [предыдущий протокол](../robustness/protocol.md). Аугментации при инференсе не применяются. Команда принимает и соответствующий `.pt`.

`results/predictions/` содержит 105 файлов с метками, оценками и идентификаторами окон. `curves/` содержит обучение по 20 эпохам. В `summary.csv` находятся средние, SD и диапазоны, в `paired_intervals.csv` — парные интервалы разностей, в `preservation.csv` и `assessment.json` — проверка заранее заданных критериев. Все материалы перечислены с контрольными суммами в `artifact_manifest.csv`.
