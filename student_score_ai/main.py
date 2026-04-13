# app/main.py
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('.env')

from student_score_ai.graph import build_graph


def main():
    app = build_graph()

    result = app.invoke({
        "video_path": Path(__file__).parent / "data/лекция_23_поток_1.mp4",
        "students_path": Path(__file__).parent / "data/students.txt",
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