# app/sheets.py
import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets"
]


def get_worksheet(service_account_path: str, sheet_id: str, worksheet_name: str):
    creds = Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(sheet_id)
    return spreadsheet.worksheet(worksheet_name)


def update_scores(service_account_path: str, sheet_id: str, worksheet_name: str, students: list[str], points_to_add: int):
    ws = get_worksheet(service_account_path, sheet_id, worksheet_name)

    all_values = ws.get_all_values()
    headers = all_values[0]

    fio_col = headers.index("ФИО") + 1
    score_col = headers.index("Баллы") + 1

    rows = all_values[1:]

    updated = []

    for i, row in enumerate(rows, start=2):
        fio = row[fio_col - 1].strip()
        if fio in students:
            current_score = row[score_col - 1].strip()
            current_score = int(current_score) if current_score.isdigit() else 0
            new_score = current_score + points_to_add
            ws.update_cell(i, score_col, new_score)
            updated.append({"student": fio, "old": current_score, "new": new_score})

    return updated