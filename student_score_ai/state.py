from pathlib import Path
from datetime import date
from typing import TypedDict

from student_score_ai.extract_names import NameListWithTime
from student_score_ai.validate_names import ValidationResult


class PipelineState(TypedDict, total=False):
    work_dir: Path
    video_path: Path
    students: list[str]
    audio_path: Path
    chunk_paths: list[Path]
    transcript_paths: list[Path]
    full_transcript: str

    extracted_names: NameListWithTime
    validated_students: ValidationResult

    sheet_id: str
    service_account_path: str
    worksheet_name: str
    lecture_date: date
    points_per_mention: float
    score_start_row: int
    sheet_updates: list[dict[str, str | int | float]]

    logs: list[str]
