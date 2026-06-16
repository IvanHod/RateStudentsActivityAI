# app/main.py
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('.env')

from student_score_ai.graph import build_graph


def main():
    path = Path('D:\\Бакалавры\\2025-2026\\Запись лекций\\Лекция 25 - максимальный поток 16.04.2025')
    app = build_graph()

    result = app.invoke({
        "video_path": path / "лекция_25_поток_1.mp4",
        "students_path": path.parent.parent / "students.txt",
        "sheet_id": os.getenv("SPREADSHEET_ID"),
        "worksheet_name": "Семестр II",
        "points_to_add": 1,
        "logs": []
    })

    print("=== DONE ===")
    for log in result.get("logs", []):
        print(log)

    print("\nValidated students:")
    for s in result.get("validated_students", []):
        print(s)

    print("\nSheet updates:")
    for u in result.get("sheet_updates", []):
        print(u)


if __name__ == "__main__":
    main()