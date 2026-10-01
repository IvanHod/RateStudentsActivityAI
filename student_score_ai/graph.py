import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
from student_score_ai.state import PipelineState
from student_score_ai.preprocess import video_to_audio
from student_score_ai.transcribe import split_audio_to_chunks, transcribe_chunks
from student_score_ai.extract_names import extract_student_names, NameListWithTime
from student_score_ai.validate_names import (
    ValidationResult,
    validate_student_names_batch_with_gpt,
)
from student_score_ai.validation_policy import VALIDATION_POLICY
from student_score_ai.sheets import update_scores


def node_video_to_audio(state: PipelineState) -> PipelineState:
    """Извлечь аудиодорожку в рабочую папку конкретного видео."""
    # Рабочая папка изолирует артефакты параллельных и последовательных видео.
    path_input: Path = state["video_path"]
    path_output = state["work_dir"] / "audio" / f"{path_input.stem}.wav"
    state["audio_path"] = path_output
    if not path_output.exists():
        print(f"Шаг 2/5: извлечение аудио — {path_input.name}")
        video_to_audio(path_input, path_output)
        state.setdefault("logs", []).append(f"Audio saved: {path_output}")
    else:
        print(f"Шаг 2/5: аудио из кэша — {path_output.name}")

    return state


def node_transcribe(state: PipelineState) -> PipelineState:
    """Разделить аудио и создать транскрипты через Whisper."""
    # Все производные файлы помещаются рядом с прочими результатами одного видео.
    path_input: Path = state["audio_path"]
    chunks_folder = state["work_dir"] / "audio_chunks"
    if len(list(chunks_folder.glob('*.wav'))) == 0:
        print("Шаг 2/5: разделение аудио на чанки")
        split_audio_to_chunks(path_input, output_dir=chunks_folder)
    state["chunk_paths"] = list(chunks_folder.glob('*.wav'))

    state.setdefault("logs", []).append(f"Splitted to chunks: '{chunks_folder}'")

    output_dir = state["work_dir"] / "transcripts"
    missing_chunk_paths = [
        chunk_path
        for chunk_path in state["chunk_paths"]
        if not (output_dir / f"{chunk_path.stem}.txt").exists()
    ]
    if missing_chunk_paths:
        print(
            f"Шаг 2/5: Whisper, осталось чанков: "
            f"{len(missing_chunk_paths)}/{len(state['chunk_paths'])}"
        )
        transcribe_chunks(missing_chunk_paths, output_dir=output_dir)
    else:
        print(f"Шаг 2/5: транскрипты из кэша — {len(list(output_dir.glob('*.txt')))} файлов")
    state["transcript_paths"] = list(output_dir.glob('*.txt'))  # list[Path]

    state["full_transcript"] = ''  # str
    for path in sorted(output_dir.glob('*.txt'), key=lambda p: int(p.stem[-4:])):
        with open(path, "r", encoding="utf-8") as file:
            state["full_transcript"] += file.read() + '\n'

    with open(state["work_dir"] / 'transcripts.txt', "w", encoding="utf-8") as file:
        file.write(state["full_transcript"])


    state["logs"].append(f"Transcribed chunks: {output_dir}")

    return state


def node_extract_names(state: PipelineState) -> PipelineState:
    """Извлечь идентификационные фрагменты и сохранить результат текущей версии."""
    # Версионный кэш запускает улучшенный извлекатель и не перезаписывает артефакты прошлых проверок.
    path_output = state["work_dir"] / f"fio_list_v{VALIDATION_POLICY.version}.json"
    if not path_output.exists():
        print("Шаг 3/5: извлечение ФИО")
        names: NameListWithTime = extract_student_names(state["transcript_paths"])
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(names.model_dump_json(indent=2))

        state.setdefault("logs", []).append(f"Extracted names: {names.model_dump_json()}")
    else:
        print("Шаг 3/5: список ФИО из кэша")

    with open(path_output, "r", encoding="utf-8") as file:
        state["extracted_names"] = NameListWithTime.model_validate_json(file.read())

    return state


def node_validate_names(state: PipelineState) -> PipelineState:
    """Сопоставить извлечённые имена со списком студентов."""
    # Кэш в рабочей папке позволяет возобновить запуск без повторной валидации.
    path_output = state["work_dir"] / f"students_validated_v{VALIDATION_POLICY.version}.json"

    if not path_output.exists():
        print("Шаг 4/5: сверка ФИО со списком студентов")
        validated: ValidationResult = validate_student_names_batch_with_gpt(
            state["extracted_names"],
            state["students"],
            state["transcript_paths"],
        )
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(validated.model_dump_json(indent=2))

        state.setdefault("logs", []).append(f"Validated names: {validated.model_dump_json()}")
    else:
        print("Шаг 4/5: результаты сверки из кэша")


    with open(path_output, "r", encoding="utf-8") as file:
        state["validated_students"] = ValidationResult.model_validate_json(file.read())
    _write_validation_comparison(
        state["work_dir"],
        state["extracted_names"],
        state["validated_students"],
    )

    return state


def _write_validation_comparison(
    work_dir: Path,
    extracted_names: NameListWithTime,
    validation_result: ValidationResult,
) -> None:
    """Записать сопоставимое сравнение результатов разных версий.

    Args:
        work_dir: папка артефактов одной лекции.
        extracted_names: канонические фрагменты и минуты текущей версии извлечения.
        validation_result: результат новой контекстной проверки.

    Returns:
        None.
    """
    # Старая и новая версии извлекают разный набор фрагментов, поэтому сверять их по индексу нельзя.
    results_by_version = _load_validation_versions(work_dir)
    results_by_version[f"v{VALIDATION_POLICY.version}"] = validation_result

    # Совпадающие исходные фрагменты сопоставляются по тексту и минуте, а новые остаются пустыми в старых версиях.
    comparison_rows: list[dict[str, object]] = []
    for index, source in enumerate(extracted_names.names):
        row: dict[str, object] = {
            "index": index,
            "source": {"raw_name": source.name, "minute": source.minute_start},
        }
        for version, result in results_by_version.items():
            matching_result = next(
                (
                    item
                    for item in result.results
                    if item.raw_name == source.name
                    and item.minute == source.minute_start
                ),
                None,
            )
            row[version] = (
                matching_result.model_dump() if matching_result is not None else None
            )
        comparison_rows.append(row)

    # Отдельный файл не меняет исходные результаты и остаётся воспроизводимым при повторном запуске.
    comparison_path = work_dir / f"students_validation_comparison_v{VALIDATION_POLICY.version}.json"
    comparison_path.write_text(
        json.dumps({"results": comparison_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_validation_versions(work_dir: Path) -> dict[str, ValidationResult]:
    """Загрузить все доступные прошлые версии валидации из рабочей папки.

    Args:
        work_dir: папка артефактов одной лекции.

    Returns:
        Результаты, сопоставленные с метками версий.
    """
    # Автопоиск сохраняет сравнение полезным при v5 и последующих версиях без нового кода.
    versions: dict[str, ValidationResult] = {}
    for path in sorted(work_dir.glob("students_validated*.json")):
        if path.name == "students_validated.json":
            version = "v1"
        else:
            match = re.fullmatch(r"students_validated_v(\d+)\.json", path.name)
            if match is None:
                continue
            version = f"v{match.group(1)}"
        versions[version] = ValidationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    return versions


def node_update_scores(state: PipelineState) -> PipelineState:
    """Начислить баллы уверенно распознанным студентам."""
    # Режим ревью позволяет проверить v4 до любого необратимого изменения ведомости.
    if VALIDATION_POLICY.review_only:
        print(
            "Шаг 5/5: начисление отключено — "
            f"v{VALIDATION_POLICY.version} ожидает ручного подтверждения"
        )
        state["sheet_updates"] = []
        state.setdefault("logs", []).append("Sheet updates skipped: validation review mode")
        return state

    score_cache_path = state["work_dir"] / f"sheet_updates_v{VALIDATION_POLICY.version}.json"
    # Локальный кэш не даёт повторно начислить баллы при перезапуске незавершённой лекции.
    if score_cache_path.exists():
        print("Шаг 5/5: начисления из кэша")
        state["sheet_updates"] = json.loads(score_cache_path.read_text(encoding="utf-8"))
        return state

    # Порог позволяет учитывать проверенные вручную вероятные совпадения согласно политике курса.
    valid_students = [
        result.matched_name
        for result in state["validated_students"].results
        if (
            result.matched_name
            and result.confidence > VALIDATION_POLICY.score_confidence_threshold
        )
    ]
    print(f"Шаг 5/5: начисление для подтверждённых упоминаний — {len(valid_students)}")
    state["sheet_updates"] = update_scores(
        service_account_path=state["service_account_path"],
        sheet_id=state["sheet_id"],
        worksheet_name=state["worksheet_name"],
        students=valid_students,
        lecture_date=state["lecture_date"],
        points_per_mention=state["points_per_mention"],
        start_row=state["score_start_row"],
    )
    state.setdefault("logs", []).append(f"Sheet updates: {state['sheet_updates']}")
    score_cache_path.write_text(
        json.dumps(state["sheet_updates"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state


def build_graph() -> Any:
    """Собрать граф обработки одного видео."""
    # Явные узлы делают отдельные этапы кэшируемыми и возобновляемыми.
    graph = StateGraph(PipelineState)

    graph.add_node("video_to_audio", node_video_to_audio)
    graph.add_node("transcribe", node_transcribe)
    graph.add_node("extract_names", node_extract_names)
    graph.add_node("validate_names", node_validate_names)
    graph.add_node("update_scores", node_update_scores)

    graph.set_entry_point("video_to_audio")
    graph.add_edge("video_to_audio", "transcribe")
    graph.add_edge("transcribe", "extract_names")
    graph.add_edge("extract_names", "validate_names")
    graph.add_edge("validate_names", "update_scores")
    graph.add_edge("update_scores", END)

    return graph.compile()
