from pathlib import Path
from typing import TypedDict

from student_score_ai.extract_names import NameListWithTime
from student_score_ai.validate_names import ValidationResult


class PipelineState(TypedDict, total=False):
    video_path: Path
    students_path: Path
    audio_path: Path
    chunk_paths: list[Path]
    transcript_paths: list[Path]
    full_transcript: str

    extracted_names: NameListWithTime
    validated_students: ValidationResult

    sheet_id: str
    worksheet_name: str
    points_to_add: int

    logs: list[str]