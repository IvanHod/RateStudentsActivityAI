"""Безопасное применение разницы баллов после ручного ревью валидации."""

import argparse
import json
import os
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

# Переменные должны быть доступны до импорта графа, который создаёт клиент языковой модели.
load_dotenv(".env")

from student_score_ai.sheets import apply_score_adjustments, get_score_values
from student_score_ai.validation_policy import VALIDATION_POLICY


DATE_PATTERN: Final = re.compile(r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)")
COMPLETION_FILE: Final = "completed.json"
RESULTS_FILE: Final = f"students_validated_v{VALIDATION_POLICY.version}.json"
PREVIEW_FILE: Final = f"sheet_updates_v{VALIDATION_POLICY.version}_preview.json"
SCORE_LEDGER_FILE: Final = "score_application_ledger.json"
REBUILD_PREVIEW_FILE: Final = (
    f"sheet_updates_v{VALIDATION_POLICY.version}_rebuild_preview.json"
)
SCORE_BASELINE_FILE: Final = "score_rebuild_baseline.json"


def main() -> None:
    """Создать preview, сверить его с таблицей либо применить рассчитанную разницу."""
    args = _parse_arguments()
    work_dir = args.work_dir
    if args.rebuild or args.rebuild_preview:
        full_plan = _build_full_score_plan(work_dir)
        rebuild_preview_path = work_dir / REBUILD_PREVIEW_FILE

        # Полный план сохраняется отдельно от дельты, чтобы оператор мог проверить именно восстановление.
        rebuild_preview_path.write_text(
            json.dumps(full_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _print_full_plan(full_plan, rebuild_preview_path)
        if args.rebuild:
            _apply_full_rebuild(work_dir, full_plan)
        return

    plan = _build_score_plan(work_dir)
    preview_path = work_dir / PREVIEW_FILE

    # Preview всегда создаётся до внешнего изменения, чтобы оператор видел точные последствия.
    preview_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_plan(plan, preview_path)
    if args.compare_sheet:
        _print_sheet_comparison(plan, work_dir)
        return
    if not args.apply:
        return

    ledger_path = work_dir / SCORE_LEDGER_FILE
    ledger = _read_score_ledger(ledger_path)
    existing_application = _find_application(ledger, VALIDATION_POLICY.version)
    if existing_application is not None:
        state = existing_application["state"]
        raise RuntimeError(
            f"Для v{VALIDATION_POLICY.version} уже есть запись в журнале "
            f"со статусом «{state}»: {ledger_path}. Повторное начисление заблокировано."
        )

    # Конфигурационная ошибка не должна оставлять ложный pending-маркер без обращения к таблице.
    settings = _get_sheet_settings()

    # Pending сохраняется раньше внешнего запроса, чтобы неизвестный результат таймаута не задвоился.
    application: dict[str, object] = {
        "validation_version": VALIDATION_POLICY.version,
        "state": "pending",
        "created_at": datetime.now().isoformat(),
        "plan": plan,
        "mention_adjustments": plan["delta_mentions"],
    }
    ledger["applications"].append(application)
    _write_score_ledger(ledger_path, ledger)

    # Внешний запрос выполняется только после сохранения pending-маркера для защиты от повтора.
    changes = apply_score_adjustments(
        service_account_path=settings["service_account_path"],
        sheet_id=settings["sheet_id"],
        worksheet_name=settings["worksheet_name"],
        adjustments=_score_adjustments(
            plan["delta_mentions"],
            settings["points_per_mention"],
        ),
        lecture_date=_lecture_monday(work_dir.name),
        start_row=settings["score_start_row"],
    )
    application.update(
        {
            "state": "applied",
            "applied_at": datetime.now().isoformat(),
            "sheet_updates": changes,
        }
    )
    _write_score_ledger(ledger_path, ledger)
    print(f"Начисления применены и записаны в журнал: {ledger_path}")


def _parse_arguments() -> argparse.Namespace:
    """Разобрать параметры запуска команды ревью.

    Returns:
        Аргументы с рабочей папкой лекции и признаком внешнего применения.
    """
    parser = argparse.ArgumentParser(
        description="Сверить текущую валидацию с начисленными баллами и применить только разницу."
    )
    parser.add_argument(
        "work_dir",
        type=Path,
        help="Папка результата лекции с результатом текущей валидации",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help="Записать рассчитанную разницу в Google Sheets",
    )
    action.add_argument(
        "--compare-sheet",
        action="store_true",
        help="Прочитать текущие ячейки Google Sheets и показать результат после корректировки",
    )
    action.add_argument(
        "--rebuild-preview",
        action="store_true",
        help="Подготовить полный план начислений без учёта старой истории",
    )
    action.add_argument(
        "--rebuild",
        action="store_true",
        help="Полностью восстановить начисления текущей версии после очистки таблицы",
    )
    return parser.parse_args()


def _build_score_plan(work_dir: Path) -> dict[str, object]:
    """Рассчитать новые упоминания, не дублируя уже начисленные ранее.

    Args:
        work_dir: папка артефактов одной лекции.

    Returns:
        Сериализуемый план с кандидатами, прежними начислениями и дельтой.

    Raises:
        FileNotFoundError: если отсутствует результат текущей валидации.
    """
    # Валидация и маркер прошлого начисления являются единственным источником расчёта дельты.
    results = _read_json_list(work_dir / RESULTS_FILE, "results")
    baseline = _read_score_baseline(work_dir / SCORE_BASELINE_FILE)
    completion_updates = _read_completed_updates(work_dir / COMPLETION_FILE)
    ledger = _read_score_ledger(work_dir / SCORE_LEDGER_FILE)
    selected_results = [
        result
        for result in results
        if result.get("matched_name")
        and isinstance(result.get("confidence"), (int, float))
        and result["confidence"] > VALIDATION_POLICY.score_confidence_threshold
    ]
    selected_mentions = Counter(
        str(result["matched_name"])
        for result in selected_results
    )
    if baseline is None:
        previous_mentions = Counter(
            str(update["student"])
            for update in completion_updates
            for _ in range(int(update["mentions"]))
        )
        previous_mentions.update(_applied_mention_adjustments(ledger))
    else:
        if baseline["state"] != "applied":
            raise RuntimeError(
                "Полное восстановление имеет статус pending; сначала сверьте таблицу "
                "и завершите разбор этой операции"
            )
        previous_mentions = Counter(baseline["selected_mentions"])
        previous_mentions.update(
            _applied_mention_adjustments(
                ledger,
                after_version=int(baseline["validation_version"]),
            )
        )
    delta_mentions = {
        name: selected_mentions.get(name, 0) - previous_mentions.get(name, 0)
        for name in sorted(set(selected_mentions) | set(previous_mentions))
        if selected_mentions.get(name, 0) != previous_mentions.get(name, 0)
    }
    return {
        "validation_version": VALIDATION_POLICY.version,
        "confidence_threshold_exclusive": VALIDATION_POLICY.score_confidence_threshold,
        "selected_results": selected_results,
        "selected_mentions": dict(sorted(selected_mentions.items())),
        "previous_applied_mentions": dict(sorted(previous_mentions.items())),
        "delta_mentions": delta_mentions,
    }


def _build_full_score_plan(work_dir: Path) -> dict[str, object]:
    """Подготовить полные начисления текущей версии без опоры на старую историю.

    Args:
        work_dir: папка артефактов одной лекции.

    Returns:
        Сериализуемый полный план с подходящими результатами и упоминаниями.
    """
    # Полный план используется только после подтверждённой очистки таблицы, не как обычная дельта.
    results = _read_json_list(work_dir / RESULTS_FILE, "results")
    selected_results = [
        result
        for result in results
        if result.get("matched_name")
        and isinstance(result.get("confidence"), (int, float))
        and result["confidence"] > VALIDATION_POLICY.score_confidence_threshold
    ]
    selected_mentions = Counter(
        str(result["matched_name"])
        for result in selected_results
    )
    return {
        "validation_version": VALIDATION_POLICY.version,
        "confidence_threshold_exclusive": VALIDATION_POLICY.score_confidence_threshold,
        "selected_results": selected_results,
        "selected_mentions": dict(sorted(selected_mentions.items())),
    }


def _apply_full_rebuild(work_dir: Path, full_plan: dict[str, object]) -> None:
    """Записать полные баллы текущей версии после явного сброса прежних начислений.

    Args:
        work_dir: папка артефактов одной лекции.
        full_plan: полный план начислений без предыдущей истории.

    Returns:
        None.

    Raises:
        RuntimeError: если для этой лекции уже начато либо завершено восстановление.
    """
    baseline_path = work_dir / SCORE_BASELINE_FILE
    existing_baseline = _read_score_baseline(baseline_path)
    if existing_baseline is not None:
        raise RuntimeError(
            f"Для лекции уже существует точка восстановления: {baseline_path}. "
            "Повторное полное начисление заблокировано."
        )

    # Конфигурационная ошибка не должна создавать ложную точку восстановления.
    settings = _get_sheet_settings()
    baseline: dict[str, object] = {
        "validation_version": VALIDATION_POLICY.version,
        "state": "pending",
        "created_at": datetime.now().isoformat(),
        "selected_mentions": full_plan["selected_mentions"],
        "plan": full_plan,
    }
    _write_score_baseline(baseline_path, baseline)

    # Полные упоминания преобразуются в баллы только после сохранения pending-маркера.
    changes = apply_score_adjustments(
        service_account_path=settings["service_account_path"],
        sheet_id=settings["sheet_id"],
        worksheet_name=settings["worksheet_name"],
        adjustments=_score_adjustments(
            full_plan["selected_mentions"],
            settings["points_per_mention"],
        ),
        lecture_date=_lecture_monday(work_dir.name),
        start_row=settings["score_start_row"],
    )
    baseline.update(
        {
            "state": "applied",
            "applied_at": datetime.now().isoformat(),
            "sheet_updates": changes,
        }
    )
    _write_score_baseline(baseline_path, baseline)
    print(f"Полные начисления восстановлены: {baseline_path}")


def _score_adjustments(
    delta_mentions: object,
    points_per_mention: str | int | float,
) -> dict[str, float]:
    """Преобразовать дельту упоминаний в допускаемые Google Sheets изменения баллов.

    Args:
        delta_mentions: отображение ФИО в изменение количества упоминаний.
        points_per_mention: стоимость одного упоминания в баллах.

    Returns:
        Ненулевые корректировки баллов по ФИО.

    Raises:
        ValueError: если план содержит нечисловую дельту.
    """
    # Явное преобразование отделяет чистый план ревью от побочного эффекта записи в таблицу.
    if not isinstance(delta_mentions, dict):
        raise ValueError("План начисления не содержит корректную дельту упоминаний")
    adjustments: dict[str, float] = {}
    for name, mentions in delta_mentions.items():
        if not isinstance(name, str) or isinstance(mentions, bool):
            raise ValueError("План начисления содержит некорректную дельту упоминаний")
        try:
            adjustment = float(mentions) * float(points_per_mention)
        except (TypeError, ValueError) as error:
            raise ValueError("План начисления содержит нечисловую дельту") from error
        if adjustment:
            adjustments[name] = adjustment
    return adjustments


def _print_sheet_comparison(plan: dict[str, object], work_dir: Path) -> None:
    """Вывести сравнение плана с текущими значениями соответствующей колонки Google Sheets.

    Args:
        plan: рассчитанный локальный план корректировок.
        work_dir: папка результата лекции для определения даты недели.

    Returns:
        None.
    """
    # Сравнение читает только нужные строки и не создаёт запись или pending-маркер.
    settings = _get_sheet_settings()
    adjustments = _score_adjustments(
        plan["delta_mentions"],
        settings["points_per_mention"],
    )
    current_scores = get_score_values(
        service_account_path=settings["service_account_path"],
        sheet_id=settings["sheet_id"],
        worksheet_name=settings["worksheet_name"],
        students=adjustments,
        lecture_date=_lecture_monday(work_dir.name),
        start_row=settings["score_start_row"],
    )
    if not adjustments:
        print("Сравнение с таблицей: корректировок нет")
        return

    # Текущий и прогнозный баллы показываются рядом, чтобы оператор видел эффект до --apply.
    print("Сравнение с Google Sheets:")
    for student, adjustment in sorted(adjustments.items()):
        current_score = current_scores.get(student)
        if current_score is None:
            print(f"- {student}: строка не найдена, корректировка {adjustment:+g}")
            continue
        print(
            f"- {student}: сейчас {current_score:g}, "
            f"корректировка {adjustment:+g}, "
            f"станет {current_score + adjustment:g}"
        )


def _read_json_list(path: Path, key: str | None) -> list[dict[str, Any]]:
    """Прочитать ожидаемый список JSON-объектов из файла результата.

    Args:
        path: путь к JSON-файлу.
        key: ключ контейнера либо ``None`` для списка в корне файла.

    Returns:
        Список JSON-объектов.

    Raises:
        ValueError: если файл не соответствует ожидаемой структуре.
    """
    # Строгая структура предотвращает расчёт дельты по повреждённому или чужому артефакту.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        values = payload.get(key) if isinstance(payload, dict) else None
    else:
        values = payload
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"Некорректная структура файла: {path}")
    return values


def _read_completed_updates(path: Path) -> list[dict[str, Any]]:
    """Прочитать прежние начисления из completed.json, если они существуют.

    Args:
        path: путь к маркеру обычного завершения лекции.

    Returns:
        Ранее применённые начисления либо пустой список для новой лекции.
    """
    # В review-режиме новая лекция ещё не имеет completed.json, но её preview должен быть доступен.
    if not path.exists():
        return []
    return _read_json_list(path, "sheet_updates")


def _read_score_ledger(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Прочитать журнал ручных применений или создать его пустое представление.

    Args:
        path: путь к журналу начислений в папке лекции.

    Returns:
        Журнал с последовательностью попыток применения.

    Raises:
        ValueError: если существующий журнал имеет неверную структуру.
    """
    # Отсутствующий журнал означает, что до этого применялись только баллы из completed.json.
    if not path.exists():
        return {"applications": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    applications = payload.get("applications") if isinstance(payload, dict) else None
    if not isinstance(applications, list) or not all(
        isinstance(application, dict)
        for application in applications
    ):
        raise ValueError(f"Некорректная структура журнала начислений: {path}")
    return {"applications": applications}


def _read_score_baseline(path: Path) -> dict[str, Any] | None:
    """Прочитать точку полного восстановления баллов, если она существует.

    Args:
        path: путь к журналу полного восстановления одной лекции.

    Returns:
        Проверенная точка восстановления либо ``None`` до первого полного запуска.

    Raises:
        ValueError: если существующая точка восстановления имеет неверную структуру.
    """
    # Базовая точка заменяет историю v3 только после явного полного восстановления таблицы.
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Некорректная точка восстановления: {path}")
    version = payload.get("validation_version")
    state = payload.get("state")
    selected_mentions = payload.get("selected_mentions")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or state not in {"pending", "applied"}
        or not isinstance(selected_mentions, dict)
    ):
        raise ValueError(f"Некорректная точка восстановления: {path}")
    if not all(
        isinstance(name, str)
        and isinstance(mentions, int)
        and not isinstance(mentions, bool)
        and mentions >= 0
        for name, mentions in selected_mentions.items()
    ):
        raise ValueError(f"Некорректные упоминания в точке восстановления: {path}")
    return payload


def _write_score_ledger(path: Path, ledger: dict[str, list[dict[str, Any]]]) -> None:
    """Атомарно сохранить журнал ручных начислений.

    Args:
        path: итоговый путь к журналу.
        ledger: сериализуемый журнал с попытками применения.

    Returns:
        None.
    """
    # Временный файл не позволит потерять pending-маркер при остановке процесса в момент записи.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_score_baseline(path: Path, baseline: dict[str, object]) -> None:
    """Атомарно сохранить точку полного восстановления баллов.

    Args:
        path: итоговый путь к точке восстановления.
        baseline: сериализуемая запись состояния полного начисления.

    Returns:
        None.
    """
    # Временный файл защищает pending-маркер от повреждения при остановке процесса.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _find_application(
    ledger: dict[str, list[dict[str, Any]]],
    validation_version: int,
) -> dict[str, Any] | None:
    """Найти единственную попытку начисления для версии валидации.

    Args:
        ledger: журнал ручных начислений.
        validation_version: номер проверенной версии.

    Returns:
        Запись применения либо ``None``, если версия ещё не обрабатывалась.

    Raises:
        ValueError: если журнал содержит несколько записей для одной версии.
    """
    # Единственность версии делает повторный запуск безопасным и заметным оператору.
    applications = [
        application
        for application in ledger["applications"]
        if application.get("validation_version") == validation_version
    ]
    if len(applications) > 1:
        raise ValueError(
            f"В журнале начислений несколько записей для v{validation_version}"
        )
    return applications[0] if applications else None


def _applied_mention_adjustments(
    ledger: dict[str, list[dict[str, Any]]],
    after_version: int | None = None,
) -> Counter[str]:
    """Получить уже применённые корректировки количества упоминаний из журнала.

    Args:
        ledger: журнал ручных начислений.
        after_version: учитывать только версии строго новее указанной базовой версии.

    Returns:
        Суммарная дельта упоминаний по ФИО.

    Raises:
        ValueError: если запись applied не содержит ожидаемой дельты упоминаний.
    """
    # В расчёт следующей версии попадают только подтверждённые операции, но не pending-попытки.
    adjustments: Counter[str] = Counter()
    for application in ledger["applications"]:
        if application.get("state") != "applied":
            continue
        application_version = application.get("validation_version")
        if isinstance(application_version, bool) or not isinstance(application_version, int):
            raise ValueError("Applied-запись журнала не содержит корректную версию")
        if after_version is not None and application_version <= after_version:
            continue
        application_adjustments = application.get("mention_adjustments")
        if not isinstance(application_adjustments, dict):
            raise ValueError(
                "Applied-запись журнала не содержит корректную дельту упоминаний"
            )
        for name, mentions in application_adjustments.items():
            if not isinstance(name, str) or isinstance(mentions, bool):
                raise ValueError(
                    "Applied-запись журнала содержит некорректную дельту упоминаний"
                )
            try:
                adjustments[name] += int(mentions)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Applied-запись журнала содержит нечисловую дельту упоминаний"
                ) from error
    return adjustments


def _get_sheet_settings() -> dict[str, str | int | float]:
    """Получить параметры доступа и начисления из окружения.

    Returns:
        Настройки Google Sheets, необходимые для одной пакетной записи.

    Raises:
        RuntimeError: если обязательная настройка не задана.
    """
    # Внешние параметры читаются только перед применением, чтобы dry-run не обращался к Google Sheets.
    required = {
        "SPREADSHEET_ID_2026": os.getenv("SPREADSHEET_ID_2026"),
        "GOOGLE_SERVICE_ACCOUNT_FILE": os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Не заданы переменные окружения: {', '.join(missing)}")
    return {
        "sheet_id": required["SPREADSHEET_ID_2026"] or "",
        "service_account_path": required["GOOGLE_SERVICE_ACCOUNT_FILE"] or "",
        "worksheet_name": os.getenv("WORKSHEET_NAME_2026", "Семестр II"),
        "points_per_mention": float(os.getenv("POINTS_PER_MENTION", "0.5")),
        "score_start_row": int(os.getenv("STUDENT_START_ROW", "4")),
    }


def _lecture_monday(work_dir_name: str) -> date:
    """Получить понедельник недели лекции из имени рабочей папки.

    Args:
        work_dir_name: имя рабочей папки, содержащее дату лекции.

    Returns:
        Понедельник недели, в которой прошла лекция.

    Raises:
        RuntimeError: если дата отсутствует или имеет неверный формат.
    """
    # Используется та же недельная колонка, что и обычный конвейер начисления.
    match = DATE_PATTERN.search(work_dir_name)
    if match is None:
        raise RuntimeError(f"В имени папки не найдена дата: {work_dir_name}")
    try:
        lecture_date = datetime.strptime(match.group(1), "%d.%m.%Y").date()
    except ValueError as error:
        raise RuntimeError(f"Некорректная дата в имени папки: {work_dir_name}") from error
    return lecture_date.fromordinal(lecture_date.toordinal() - lecture_date.weekday())


def _print_plan(plan: dict[str, object], preview_path: Path) -> None:
    """Вывести оператору короткий итог рассчитанной дельты.

    Args:
        plan: сериализуемый план начисления.
        preview_path: путь записанного preview-файла.

    Returns:
        None.
    """
    # Короткий вывод снижает риск запустить --apply, не заметив кандидатов и число упоминаний.
    delta_mentions = plan["delta_mentions"]
    print(f"Preview сохранён: {preview_path}")
    print(f"Порог: confidence > {VALIDATION_POLICY.score_confidence_threshold}")
    print(f"Корректировки упоминаний: {delta_mentions}")


def _print_full_plan(full_plan: dict[str, object], preview_path: Path) -> None:
    """Вывести оператору итог полного восстановления после очистки таблицы.

    Args:
        full_plan: план начислений без вычитания прежней истории.
        preview_path: путь записанного preview-файла полного восстановления.

    Returns:
        None.
    """
    # Отдельный вывод не позволяет спутать полные начисления с обычной дельтой.
    print(f"Preview полного восстановления сохранён: {preview_path}")
    print(f"Порог: confidence > {VALIDATION_POLICY.score_confidence_threshold}")
    print(f"Полные упоминания: {full_plan['selected_mentions']}")


if __name__ == "__main__":
    main()
