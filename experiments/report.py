"""Составление отчётов по сохранённым результатам экспериментов с CSI.

Обучение вызывает ``build_report()`` после записи плана, запусков и метрик.
Отчёты можно обновить отдельно, в том числе до завершения всей сетки.
Для этого достаточно стандартной библиотеки Python.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("tinycnn", "tcn", "cnngru", "transformer")
PREPROCESSORS = ("logamp", "relative", "relative_diff", "lp20_centered")
ARCHITECTURE_NAMES = {
    "tinycnn": "TinyCNN",
    "tcn": "Causal TCN",
    "cnngru": "CNN-GRU",
    "transformer": "Tiny Transformer",
}
PARAMETERS = {"tinycnn": 26849, "tcn": 46081, "cnngru": 43681, "transformer": 96897}
SUCCESS = {"complete", "completed", "success", "ok"}
FAILURE = {"failed", "error", "interrupted"}
RUNNING = {"running", "started", "in_progress"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (list, tuple)):
        value = " + ".join(map(str, value))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _integer(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return str(int(number)) if math.isfinite(number) else "—"


def _count(value: Any) -> Any:
    return len(value) if isinstance(value, (list, tuple)) else value


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in ("scenario", "preprocessing", "architecture", "seed"))


def _index_unique(rows: list[dict[str, Any]], *, domains: bool = False) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = _key(row) + ((str(row.get("target_domain", "")),) if domains else ())
        if key in indexed:
            raise ValueError(f"В таблице результатов повторяется строка: {key}")
        indexed[key] = row
    return indexed


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _status(run: dict[str, Any] | None, metric: dict[str, Any] | None) -> str:
    if run is None:
        return "метрики записаны; журнала нет" if metric is not None else "ожидает"
    status = str(run.get("status", "")).lower()
    if status in SUCCESS:
        return "готово" if metric is not None else "готово; метрики отсутствуют"
    if status in FAILURE:
        return "ошибка / прервано"
    if status in RUNNING:
        return "выполняется"
    return status or "ожидает"


def build_report(root: str | Path | None = None) -> dict[str, Any]:
    """Составить два Markdown-отчёта по результатам в корне проекта ``root``.

    В начале обучения таблицы CSV могут ещё отсутствовать; план обязателен.
    Все запланированные запуски остаются в отчёте, пропущенные значения
    обозначаются прочерком.
    """
    root = Path(root).resolve() if root is not None else ROOT
    public = root / "experiments" / "results"
    plan = json.loads((public / "plan.json").read_text(encoding="utf-8-sig"))
    scenarios = plan["scenarios"]
    architectures = _list(plan.get("architectures", ARCHITECTURES))
    preprocessors = _list(plan.get("preprocessors", PREPROCESSORS))
    seeds = _list(plan.get("seeds", plan.get("seed", 42)))
    epochs = plan.get("epochs", 20)
    batch = plan.get("batch", plan.get("batch_size", 64))
    runs = _read_csv(public / "runs.csv")
    results = _read_csv(public / "results.csv")
    recordings = _read_csv(public / "recordings.csv")
    manifest_path = public / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    run_index = _index_unique(runs)
    metric_index = _index_unique(results, domains=True)
    scenario_index = {str(scenario["id"]): scenario for scenario in scenarios}
    if len(scenario_index) != len(scenarios):
        raise ValueError("Идентификаторы сценариев должны быть уникальными")

    expected = []
    expected_metric_count = 0
    for scenario in scenarios:
        target_domains = _list(scenario["test_domains"])
        if len(target_domains) != len(set(map(str, target_domains))) or not target_domains:
            raise ValueError(f"Нужен непустой список разных тестовых доменов: {scenario['id']}")
        targets = target_domains + (["ALL"] if len(target_domains) > 1 else [])
        for preprocessing in preprocessors:
            for architecture in architectures:
                for seed in seeds:
                    key = (str(scenario["id"]), str(preprocessing), str(architecture), str(seed))
                    expected.append((key, scenario, targets))
                    expected_metric_count += len(targets)

    expected_keys = {key for key, _, _ in expected}
    extra_runs = set(run_index) - expected_keys
    extra_metrics = {key[:4] for key in metric_index} - expected_keys
    if extra_runs or extra_metrics:
        raise ValueError(f"В результатах есть конфигурации, которых нет в плане: {extra_runs | extra_metrics}")
    expected_metric_keys = {key + (str(target),) for key, _, targets in expected for target in targets}
    if set(metric_index) - expected_metric_keys:
        raise ValueError("В результатах есть тестовые домены, которых нет в плане")

    complete = sum(str(run.get("status", "")).lower() in SUCCESS for run in runs)
    failed = sum(str(run.get("status", "")).lower() in FAILURE for run in runs)
    running = sum(str(run.get("status", "")).lower() in RUNNING for run in runs)
    pending = len(expected) - complete - failed - running
    complete_with_all_metrics = sum(
        bool(run_index.get(key))
        and str(run_index[key].get("status", "")).lower() in SUCCESS
        and all(key + (str(target),) in metric_index for target in targets)
        for key, _, targets in expected
    )
    finished = complete_with_all_metrics == len(expected)
    now = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y %H:%M МСК")
    timestamp = f"Обновлено {now}."
    progress = (
        f"Завершены все {complete} обучений. Сохранены {len(results)} строк метрик по тестовым доменам."
        if finished else
        f"Завершено {complete} из {len(expected)} обучений; выполняется {running}, "
        f"прервано {failed}, ожидает запуска {pending}. "
        f"В таблицах пока есть {len(results)} из {expected_metric_count} строк метрик. "
        "Прочерк означает, что результат ещё не записан."
    )

    preprocessor_rows = [
        ["`logamp`", "`log(1 + A)`", "Сохраняет средний уровень амплитуды и её изменения"],
        ["`relative`", "`R = A / max(mean_time(A), 1) − 1`", "Среднее по времени считается для каждой поднесущей текущего окна; дисперсия сохраняется"],
        ["`relative_diff`", "`R[t] − R[t−1]`", "Первая разность `relative`; первый отсчёт — ноль"],
        ["`lp20_centered`", "ФНЧ Баттерворта 4-го порядка, 20 Гц → вычитание среднего окна", "Начальное состояние фильтра задаётся по первому отсчёту и сбрасывается перед каждым окном"],
    ]
    architecture_rows = [
        [ARCHITECTURE_NAMES.get(name, name), f"`{name}`", PARAMETERS.get(name, "—")]
        for name in architectures
    ]
    scenario_rows = []
    for scenario in scenarios:
        scenario_rows.append([
            f"`{scenario['id']}`", scenario.get("family", "—"),
            f"{scenario['train_dataset']}: {_cell(scenario['train_domains'])}",
            f"{scenario['test_dataset']}: {_cell(scenario['test_domains'])}",
            _count(scenario.get("train_recordings")), _count(scenario.get("test_recordings")),
            _integer(scenario.get("train_windows")), _integer(scenario.get("test_windows")),
        ])

    main_rows = []
    domain_rows = []
    summary_values: dict[tuple[str, str], list[dict[str, Any]]] = {}
    missing_pooled = []
    for key, scenario, targets in expected:
        scenario_id, preprocessing, architecture, seed = key
        run = run_index.get(key)
        pooled_target = "ALL" if len(_list(scenario["test_domains"])) > 1 else str(_list(scenario["test_domains"])[0])
        pooled = metric_index.get(key + (pooled_target,))
        run_id = (run or {}).get("run_id", f"{scenario_id}__{preprocessing}__{architecture}__s{seed}")
        if run and str(run.get("status", "")).lower() in SUCCESS and pooled is None:
            missing_pooled.append(run_id)
        main_rows.append([
            f"`{run_id}`", f"`{preprocessing}`", ARCHITECTURE_NAMES.get(architecture, architecture),
            f"{scenario['train_dataset']}: {_cell(scenario['train_domains'])}",
            f"{scenario['test_dataset']}: {_cell(scenario['test_domains'])}",
            f"{_count(scenario.get('train_recordings'))} / {_count(scenario.get('test_recordings'))}",
            f"{_integer(scenario.get('train_windows'))} / {_integer(scenario.get('test_windows'))}",
            _number((pooled or {}).get("balanced_accuracy")), _number((pooled or {}).get("f1")),
            _number((pooled or {}).get("roc_auc")), _number((pooled or {}).get("average_precision")),
            _number((run or {}).get("train_seconds"), 1), _status(run, pooled),
        ])
        if pooled is not None and run and str(run.get("status", "")).lower() in SUCCESS:
            summary_values.setdefault((preprocessing, architecture), []).append(pooled)
        for target in targets:
            metric = metric_index.get(key + (str(target),))
            metric = metric or {}
            domain_rows.append([
                f"`{run_id}`", scenario.get("family", "—"), f"`{preprocessing}`",
                ARCHITECTURE_NAMES.get(architecture, architecture), scenario["test_dataset"], target,
                _integer(metric.get("n_windows")), _integer(metric.get("n_positive")), _integer(metric.get("n_negative")),
                _integer(metric.get("tp")), _integer(metric.get("tn")), _integer(metric.get("fp")), _integer(metric.get("fn")),
                _number(metric.get("accuracy")), _number(metric.get("balanced_accuracy")),
                _number(metric.get("precision")), _number(metric.get("recall")), _number(metric.get("f1")),
                _number(metric.get("roc_auc")), _number(metric.get("average_precision")),
                _number(metric.get("threshold"), 2), "готово" if metric else _status(run, None),
            ])

    summary_rows = []
    for preprocessing in preprocessors:
        for architecture in architectures:
            values = summary_values.get((str(preprocessing), str(architecture)), [])
            averages = []
            for field in ("balanced_accuracy", "f1", "roc_auc", "average_precision"):
                numbers = []
                for value in values:
                    try:
                        number = float(value[field])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        numbers.append(number)
                averages.append(_number(mean(numbers)) if numbers else "—")
            summary_rows.append([
                f"`{preprocessing}`", ARCHITECTURE_NAMES.get(architecture, architecture),
                f"{len(values)}/{len(scenarios) * len(seeds)}", *averages,
            ])

    findings = []
    if finished:
        ranked = sorted(
            [(mean(float(value['balanced_accuracy']) for value in values), preprocessing, architecture)
             for (preprocessing, architecture), values in summary_values.items()], reverse=True)
        first, second = ranked[:2]
        findings = [
            f"Лучший средний результат по {len(scenarios)} сценариям показала связка `{first[1]}` и "
            f"{ARCHITECTURE_NAMES[first[2]]}: balanced accuracy {first[0]:.4f}. "
            f"У `{second[1]}` и {ARCHITECTURE_NAMES[second[2]]} результат почти такой же — {second[0]:.4f}. "
            "Для дальнейших экспериментов стоит сохранить оба варианта. Обучение проводилось с одним seed, "
            "поэтому пока неизвестно, насколько устойчива эта разница.", "",
            "Следующий шаг — повторить обучения с несколькими seed и проверить выбранные модели на новом домене. "
            "В него стоит добавить неподвижного человека и несколько независимых фоновых записей. "
            "Это позволит проверить, распознаёт ли модель само движение и сохраняется ли результат за пределами текущих записей.", "",
        ]

    figures = []
    if (root / "research" / "figures" / "experiment_grid_balanced_accuracy.png").is_file():
        figures += [
            "![Balanced accuracy при разных разбиениях, предобработках и архитектурах](figures/experiment_grid_balanced_accuracy.png)",
            "", "На каждой панели показана одна группа разбиений. Значения усреднены по её сценариям; "
            "для проверки на нескольких доменах используется общая метрика `ALL`.", "",
        ]
    if (root / "research" / "figures" / "experiment_domain_transfer.png").is_file():
        figures += [
            "![Перенос между доменами Wallhack и днями 3DO](figures/experiment_domain_transfer.png)",
            "", "Матрицы показывают, как меняется результат при переносе между доменами. "
            "Для этого примера заранее выбрана связка `relative` и TinyCNN. "
            "Сравнение всех моделей приведено в таблице выше.", "",
        ]

    document = [
        "# Эксперименты с CSI", "", timestamp, "",
        "Для задачи «есть движение / нет движения» проверены четыре архитектуры и четыре варианта предобработки. "
        f"Каждое сочетание обучалось на {len(scenarios)} разбиениях данных. " + progress, "",
        "### Данные", "",
        (f"В обучении используются {len(recordings)} полных записей и {manifest.get('shape', ['—'])[0]} окон. "
         "Число записей по наборам: "
         + "; ".join(f"{dataset} — {count}" for dataset, count in sorted(Counter(row.get('dataset', '') for row in recordings).items()))
         + ".") if recordings else "Подготовленные записи пока не появились в таблице данных.", "",
        "В Wallhack к движению относятся ходьба и ходьба с махами руками, к фону — пустое помещение. "
        "Исходная метка 0 остаётся нулём, метки 1 и 2 объединяются в класс 1. "
        "В 3DO взяты только полные записи фона `b*` и ходьбы `w*`, с метками 0 и 1. "
        "Записи сидения и лежания `s*` и `l*` исключены: в них встречаются движения, а точных временных меток для них нет.", "",
        "Таким образом, текущие результаты описывают различение ходьбы и пустой сцены. "
        "Неподвижный человек в отрицательном классе не представлен. "
        "Остальные четыре локальных набора — UT-HAR, NTU-Fi HAR, NTU-Fi HumanID и Widar — "
        "не содержат подходящего размеченного фона. Выбор наборов описан в [README](../README.md).", "",
        "### Подготовка данных", "",
        "CSI читается из исходных CSV. Комплексные отсчёты восстанавливаются как "
        "`real=data[2k+1]`, `imag=data[2k]`; готовые массивы 3DO с ошибочным объединением I/Q не используются. "
        "Из L-LTF берутся 52 поднесущие `[6:32] + [33:59]`, затем вычисляется амплитуда `A=|H|`.", "",
        "При чтении исправляются переполнения таймера uint32. Повреждённые пакеты и пакеты с `rx_state≠0` пропускаются. "
        "Интервал `Δt≤0` или `Δt>50 мс` начинает новый сегмент. Внутри сегмента данные интерполируются "
        "на сетку 100 Гц и делятся на неперекрывающиеся окна по 4 секунды. Одно окно содержит 400 отсчётов; "
        "модель получает массив `[52,400]`.", "",
        _table(["Предобработка", "Преобразование амплитуды", "Как применяется"], preprocessor_rows), "",
        "После преобразования каждый канал стандартизируется. Среднее и стандартное отклонение считаются "
        "по времени и всем окнам обучающей выборки; нижняя граница стандартного отклонения — `1e−4`. "
        "К тестовым данным применяются эти же параметры. Среднее внутри `relative` и центрирования "
        "вычисляется отдельно для текущего окна. Фильтр `lp20_centered` работает причинно, "
        "но модель всё равно выдаёт результат после получения полного четырёхсекундного окна.", "",
        "### Разделение на обучение и тест", "",
        "Данные разделяются по полным исходным записям и доменам. Окна одной записи всегда остаются "
        "в одной части. Во время обучения перемешивается только порядок окон в уже собранной обучающей выборке.", "",
        "У Wallhack четыре домена: `LOS/BQ`, `LOS/PIFA`, `NLOS/BQ`, `NLOS/PIFA`. "
        "Они описывают сочетания антенны и условий распространения сигнала. Для них проверены схемы "
        "«один домен → остальные три» и «три домена → четвёртый». "
        "У 3DO три домена — дни `d1`, `d2`, `d3`, поэтому используются схемы «один день → остальные два» "
        "и «два дня → третий». Перенос между датасетами проверяется в обе стороны.", "",
        f"Всего получается {len(scenarios)} сценариев. Для каждого выполнены {len(preprocessors)} варианта "
        f"предобработки × {len(architectures)} архитектуры × {len(seeds)} seed. "
        "Точные списки записей сохранены в [plan.json](../experiments/results/plan.json), "
        "а источник и границы каждого окна — в [windows.csv](../experiments/results/windows.csv).", "",
        _table(["Сценарий", "Схема", "Обучение", "Тест", "Записей для обучения", "Записей для теста", "Окон для обучения", "Окон для теста"], scenario_rows), "",
        "### Обучение моделей", "",
        _table(["Архитектура", "Идентификатор", "Обучаемых параметров"], architecture_rows), "",
        f"Каждая модель обучалась {epochs} эпох с размером батча {batch} и seed {_cell(seeds)}. "
        f"Использовался {plan.get('optimizer', 'AdamW')} с weight decay `1e−4`. "
        "Начальная скорость обучения — `1e−3` для TinyCNN, TCN и CNN-GRU, `3e−4` для Transformer. "
        "По косинусному расписанию она снижалась до 1% начального значения. Dropout равен 0.2, "
        "норма градиента ограничена значением 1.0.", "",
        "Функция потерь — `BCEWithLogitsLoss`. Вес положительного класса рассчитывается как "
        "`число фоновых окон / число окон движения` только на обучающих данных; отдельного "
        "балансирующего семплера нет. Для оценки берутся веса последней эпохи и порог 0.5.", "",
        "Отдельная валидационная выборка и ранняя остановка в этой сетке не использовались. "
        "В каждом домене Wallhack есть только одна фоновая запись: если отдать её на валидацию целиком, "
        "в обучении не останется фона. Поэтому число эпох и порог были выбраны заранее "
        "и не подстраивались по тестовым результатам.", "",
        f"Обучения выполнялись на {plan.get('gpu', 'устройстве, не указанном в плане')} "
        "с CUDA autocast float16, GradScaler и разрешённым TF32. "
        "Один и тот же seed помогает повторить эксперимент, но не гарантирует побитового совпадения "
        "на другой версии GPU или CUDA.", "",
        "### Сравнение результатов", "",
        *findings,
        "В таблице каждый сценарий имеет одинаковый вес. Если в нём несколько тестовых доменов, "
        "сначала считается метрика по всем их предсказаниям (`ALL`), затем она усредняется с остальными сценариями. "
        "Это усреднение по сценариям, а не по всем окнам датасетов сразу.", "",
        _table(["Предобработка", "Архитектура", "Готово сценариев", "Средняя balanced accuracy", "Средний F1", "Средний ROC-AUC", "Средняя AP"], summary_rows), "",
        *figures,
        "### Результаты каждого обучения", "",
        "Одна строка соответствует одному обучению. В столбцах с числом записей и окон сначала указана "
        "обучающая часть, затем тестовая. Для нескольких тестовых доменов показана общая метрика `ALL`. "
        "Отдельные домены и матрицы ошибок приведены в [дополнительной таблице](experiment_results_by_domain.md).", "",
        "F1, precision и recall относятся к движению. AP означает average precision; "
        "при нулевом знаменателе precision, recall и F1 приняты равными нулю. "
        "Время в таблице включает только цикл обучения модели.", "",
        _table(["Запуск", "Предобработка", "Архитектура", "Обучение", "Тест", "Записей обучение / тест", "Окон обучение / тест", "Balanced accuracy", "F1", "ROC-AUC", "AP", "Обучение, с", "Статус"], main_rows), "",
        "### Что ещё нужно проверить", "",
        "Высокие оценки, включая округлённые значения 1.000, получены на конкретных записях фона и ходьбы. "
        "Окна одной сессии зависимы, поэтому большое число окон не заменяет новые независимые записи. "
        "По этим экспериментам пока нельзя оценить обнаружение начала движения, остановку человека "
        "или задержку непрерывного детектора. Доля ложных срабатываний по окнам также не даёт число тревог в час.", "",
        "Текущая сетка помогает выбрать кандидатов для следующего эксперимента. "
        "После выбора по этим таблицам окончательную оценку нужно проводить на новых данных. "
        "Повторы с несколькими seed и дополнительные фоновые сессии позволят оценить разброс результатов.", "",
        "### Файлы и повторный запуск", "",
        "Исходные значения находятся в [runs.csv](../experiments/results/runs.csv) и [results.csv](../experiments/results/results.csv). "
        "Список записей — в [recordings.csv](../experiments/results/recordings.csv), параметры подготовки — "
        "в [data_manifest.json](../experiments/results/data_manifest.json). "
        "В `runs.csv` также сохранены время обучения, первая и последняя ошибка, число параметров и путь к весам.", "",
        "Код находится в `experiments/`, веса и обработанные массивы — в `experiments/results/`. "
        "Команды подготовки, обучения и проверки описаны в [README](../README.md). "
        "Чтобы обновить этот отчёт по сохранённым результатам, выполните из корня проекта "
        "`python experiments/report.py`.",
    ]
    if missing_pooled:
        document += ["", "У следующих завершённых запусков отсутствуют общие метрики: " + ", ".join(f"`{name}`" for name in missing_pooled) + "."]

    detailed = [
        "# Результаты по тестовым доменам", "", timestamp, "",
        "Здесь собраны метрики каждого обучения отдельно по доменам. "
        "Данные, предобработка и схемы разделения описаны в [основном отчёте](experiment_results.md). "
        + progress, "",
        "Строка `ALL` объединяет предсказания нескольких тестовых доменов одного запуска. "
        "Она не добавляет отдельного обучения. Порог классификации — 0.5; "
        "`+` обозначает движение, `−` — фон.", "",
        "TP — верно найденное движение, TN — верно найденный фон, FP — ложное срабатывание, "
        "FN — пропущенное движение. Precision, recall и F1 рассчитаны для класса движения. "
        "AP — average precision; её величина зависит от доли положительных примеров. "
        "Если модель не предсказала ни одного положительного примера, precision принимается равным нулю.", "",
        _table(["Запуск", "Схема", "Предобработка", "Архитектура", "Тестовый датасет", "Тестовый домен", "Окон", "+", "−", "TP", "TN", "FP", "FN", "Accuracy", "Balanced accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP", "Порог", "Статус"], domain_rows), "",
        "Полные значения сохранены в [results.csv](../experiments/results/results.csv). "
        "Надёжные доверительные интервалы пока не рассчитаны: соседние окна одной записи зависимы. "
        "В частности, в каждом домене Wallhack фон представлен одной сессией.",
    ]
    main_path = root / "research" / "experiment_results.md"
    domain_path = root / "research" / "experiment_results_by_domain.md"
    _write_atomic(main_path, "\n".join(document))
    _write_atomic(domain_path, "\n".join(detailed))
    return {
        "planned": len(expected), "complete": complete, "running": running,
        "failed": failed, "pending": pending, "complete_with_all_metrics": complete_with_all_metrics,
        "metric_rows": len(results), "expected_metric_rows": expected_metric_count,
        "finished": finished, "main_report": str(main_path), "domain_report": str(domain_path),
    }


if __name__ == "__main__":
    summary = build_report()
    print(f"Отчёты обновлены: завершено {summary['complete']} из {summary['planned']} обучений, "
          f"сохранено {summary['metric_rows']} строк метрик.")
    print(f"Основной отчёт: {summary['main_report']}")
    print(f"Результаты по доменам: {summary['domain_report']}")
