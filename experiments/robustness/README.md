# PCA, классические модели и перенос между устройствами

Эксперимент сопоставляет TCN с логистической регрессией, LightGBM и пороговым детектором, проверяет PCA и перенос на Days и OPERAnet. Полная сетка архитектур не запускается. Отбор выполняется на development-доменах Wallhack; целевые среды используются для внешней проверки.

Основные выводы, метрики, графики и ссылки на итоговые веса приведены в [отчёте](../../research/robustness_report.md). Условия сравнения зафиксированы в [protocol.md](protocol.md). Версии реализаций и параметры моделей указаны в `results/locked_selection.json` и `results/provenance.json`.

## Воспроизведение

Нужны исходные архивы `datasets/day*.zip`, `datasets/wificsi1.zip`, `datasets/wificsi2.zip`, исправленный кэш `experiments/motion_results/amplitudes.npy` и существующие веса TCN. Кэш исходного эксперимента восстанавливается командой `python -m experiments.final_motion.prepare_data`. Не следует запускать `train_grid.py` для этой проверки.

Дополнительные зависимости перечислены в `requirements.txt`; версии среды сохранены в `results/environment.json`. Все команды выполняются из корня репозитория.

```powershell
python -m pip install -r requirements.txt
python -m pip install -r experiments/robustness/requirements.txt
python -m experiments.robustness.datasets days
python -m experiments.robustness.datasets opera
python -m experiments.robustness.verify
python -m experiments.robustness.run source
python -m experiments.robustness.run external
python -m experiments.robustness.verify --final
python -m experiments.robustness.analyze
```

Существующие модели и прогнозы повторно используются при совпадении идентификаторов запусков и состава записей. При изменении чтения данных или параметров эксперимента результаты нужно сохранять в новый каталог: продолжение старого запуска не предназначено для сравнения разных версий кода.

`datasets.py` проверяет все записи, сохраняет инвентаризацию и готовит ограниченную выборку окон. Разделение OPERAnet выполняется по экспериментам с объединением двух приёмников в одну группу. Обучение на Room1 использует фиксированные 20 эпох; Room2 не участвует в подборе эпох, порога или PCA.

## Сохранённые модели и прогнозы

В `results/checkpoints/*.joblib` находятся готовые предикторы: модель, средние, масштабы, PCA и порог. Для TCN дополнительно сохранены `.pt` с тензорами весов и преобразованиями. JSON рядом с каждым предиктором содержит индексы train/validation, перечень записей, конфигурацию и долю объяснённой дисперсии PCA.

Файлы `all_wallhack__*` обучены на исходном источнике; именно они используются для прямого переноса. Файлы `opera_room1__*` обучены на двух приёмниках первой комнаты OPERAnet. Файлы `domain_holdout__*` предназначены для проверки исключённого домена Wallhack. Подменять эти назначения при сопоставлении метрик нельзя.

Прогноз сохранённого предиктора по уже подготовленным амплитудам формы `[N,52,400]`:

```powershell
python -m experiments.robustness.infer tmp/robustness_cache/opera.npy --model experiments/robustness/results/checkpoints/opera_room1__tcn__s42.joblib --output tmp/opera_predictions.npz
```

Амплитуды должны быть согласованы с антенным каналом, временной сеткой и адаптером поднесущих из протокола. Команда не принимает произвольный сырой CSI или изображение спектрограммы.

`results/predictions/` содержит оценки и решения для каждого окна с исходной меткой, именем записи и временными границами. `results/curves/` содержит кривые обучения. Сводные CSV сохраняют матрицы ошибок и метрики по средам. `results/verification.json` фиксирует проверки декодирования, разбиений и воспроизводимости прогнозов после загрузки весов. Контрольные суммы файлов находятся в `results/artifact_manifest.csv`.

Большие кэши и распакованные исходники остаются в `tmp/robustness_cache` и `tmp/robustness_probe`; каталог `tmp` исключён из Git. Для публикации результатов достаточно кода, протокола, весов, метрик, прогнозов и готовых графиков. Графики хранятся в `research/robustness_figures`, их исходные числовые данные — в `results`.
