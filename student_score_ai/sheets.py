# app/sheets.py
"""Чтение студентов и начисление баллов в Google Sheets."""

from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from google.oauth2.service_account import Credentials
from google.auth.transport.requests import AuthorizedSession


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets"
]
SHEETS_API_URL = "https://sheets.googleapis.com/v4/spreadsheets"


def get_sheets_service(service_account_path: str) -> Any:
    """Создать авторизованный клиент Google Sheets.

    Args:
        service_account_path: путь к JSON-ключу service account.

    Returns:
        Клиент Google Sheets API v4.
    """
    # AuthorizedSession обходит нестабильное TLS-поведение httplib2 на текущем окружении.
    credentials = Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
    return AuthorizedSession(credentials)


def get_students(
    service_account_path: str,
    sheet_id: str,
    worksheet_name: str,
    start_row: int = 4,
) -> list[str]:
    """Получить список студентов из колонки A, начиная с заданной строки.

    Args:
        service_account_path: путь к JSON-ключу service account.
        sheet_id: идентификатор Google-таблицы.
        worksheet_name: название листа таблицы.
        start_row: первая строка со студентом.

    Returns:
        Непустые ФИО студентов без повторов, в порядке таблицы.
    """
    # Диапазон узкий, чтобы не читать лишние данные и не зависеть от заголовков таблицы.
    range_name = f"'{worksheet_name}'!A{start_row}:A"
    service = get_sheets_service(service_account_path)
    response = _get_values(service, sheet_id, range_name)
    return list(
        dict.fromkeys(
            row[0].strip()
            for row in response.get("values", [])
            if row and row[0].strip()
        )
    )


def update_scores(
    service_account_path: str,
    sheet_id: str,
    worksheet_name: str,
    students: Iterable[str],
    lecture_date: date,
    points_per_mention: float = 0.5,
    start_row: int = 4,
) -> list[dict[str, str | int | float]]:
    """Прибавить баллы указанным студентам в строках той же таблицы.

    Args:
        service_account_path: путь к JSON-ключу service account.
        sheet_id: идентификатор Google-таблицы.
        worksheet_name: название листа таблицы.
        students: ФИО студентов, которым начисляются баллы.
        lecture_date: дата лекции из имени файла.
        points_per_mention: сколько баллов добавить за одно упоминание.
        start_row: первая строка со студентом.

    Returns:
        Сведения о выполненных изменениях.
    """
    # Счётчик сохраняет все упоминания: каждое из них добавляет 0,5 балла.
    mentions = Counter(student.strip() for student in students if student.strip())
    if not mentions:
        return []

    service = get_sheets_service(service_account_path)
    names_range = f"'{worksheet_name}'!A{start_row}:A"
    names = _get_values(service, sheet_id, names_range).get("values", [])
    score_column = _find_lecture_column(service, sheet_id, worksheet_name, lecture_date)
    scores_range = f"'{worksheet_name}'!{score_column}{start_row}:{score_column}"
    scores = _get_values(service, sheet_id, scores_range).get("values", [])

    # Пакетное обновление уменьшает число сетевых запросов и не оставляет частично обновлённые строки.
    changes: list[dict[str, object]] = []
    updated: list[dict[str, str | int | float]] = []
    for index, row in enumerate(names, start=start_row):
        name = row[0].strip() if row else ""
        mention_count = mentions.get(name, 0)
        if not mention_count:
            continue
        score_index = index - start_row
        current_value = (
            scores[score_index][0]
            if score_index < len(scores) and scores[score_index]
            else "0"
        )
        current_score = _parse_score(current_value)
        points_added = mention_count * points_per_mention
        new_score = current_score + points_added
        changes.append(
            {"range": f"'{worksheet_name}'!{score_column}{index}", "values": [[new_score]]}
        )
        updated.append(
            {
                "student": name,
                "mentions": mention_count,
                "old": current_score,
                "added": points_added,
                "new": new_score,
            }
        )

    if changes:
        response = service.post(
            f"{SHEETS_API_URL}/{sheet_id}/values:batchUpdate",
            params={"valueInputOption": "USER_ENTERED"},
            json={"data": changes},
            timeout=60,
        )
        response.raise_for_status()
    return updated


def _find_lecture_column(
    service: Any,
    sheet_id: str,
    worksheet_name: str,
    lecture_date: date,
) -> str:
    """Найти столбец, в третьей строке которого стоит дата лекции.

    Args:
        values_api: ресурс значений Google Sheets API.
        sheet_id: идентификатор Google-таблицы.
        worksheet_name: название листа таблицы.
        lecture_date: дата, которую нужно найти.

    Returns:
        Буквенное имя подходящего столбца.

    Raises:
        RuntimeError: если дата отсутствует или встречается несколько раз.
    """
    # Форматированное значение сохраняет вид даты, который видит преподаватель в строке заголовков.
    response = _get_values(
        service,
        sheet_id,
        f"'{worksheet_name}'!3:3",
        valueRenderOption="FORMATTED_VALUE",
    )
    values = response.get("values", [[]])[0]
    matching_columns = [
        _column_name(index)
        for index, value in enumerate(values, start=1)
        if _parse_sheet_date(value) == lecture_date
    ]
    if len(matching_columns) != 1:
        formatted_date = lecture_date.strftime("%d.%m.%Y")
        raise RuntimeError(
            f"Для даты {formatted_date} в строке 3 найдено столбцов: {len(matching_columns)}"
        )
    return matching_columns[0]


def _get_values(
    service: Any,
    sheet_id: str,
    range_name: str,
    **params: str,
) -> dict[str, Any]:
    """Получить значения диапазона Google Sheets через авторизованную сессию.

    Args:
        service: авторизованная HTTP-сессия Google.
        sheet_id: идентификатор Google-таблицы.
        range_name: диапазон в A1-нотации.
        **params: дополнительные параметры Google Sheets API.

    Returns:
        JSON-ответ API.
    """
    # URL-кодирование параметров выполняет requests, сохраняя кириллицу в названии листа.
    response = service.get(
        f"{SHEETS_API_URL}/{sheet_id}/values/{range_name}",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _parse_sheet_date(value: object) -> date | None:
    """Распознать дату из отображаемого значения ячейки Google Sheets.

    Args:
        value: отображаемое содержимое ячейки.

    Returns:
        Дату или ``None``, если ячейка не содержит поддерживаемую дату.
    """
    # Несколько форматов покрывают типичные русские настройки таблиц без привязки к локали API.
    text = str(value).strip()
    for format_string in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue
    return None


def _column_name(index: int) -> str:
    """Преобразовать индекс столбца Google Sheets в буквенное обозначение.

    Args:
        index: индекс столбца, начиная с единицы.

    Returns:
        Буквенное имя столбца, например ``A`` или ``AA``.
    """
    # Алфавитная нумерация нужна, потому что API принимает диапазоны в A1-нотации.
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _parse_score(value: object) -> float:
    """Преобразовать содержимое ячейки с баллами в число.

    Args:
        value: значение ячейки Google Sheets.

    Returns:
        Количество уже начисленных баллов.
    """
    # Пустые и нечисловые ячейки считаются нулём, чтобы первая запись не падала с ошибкой.
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return 0.0
