import json
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
from student_score_ai.state import PipelineState
from student_score_ai.preprocess import video_to_audio
from student_score_ai.transcribe import split_audio_to_chunks, transcribe_chunks
from student_score_ai.extract_names import extract_student_names, NameListWithTime
from student_score_ai.validate_names import validate_student_names_with_gpt, ValidationResult
from student_score_ai.sheets import update_scores


PIPELINE_CACHE_VERSION = 2


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
    if not output_dir.exists() or len(list(output_dir.glob('*.txt'))) == 0:
        print(f"Шаг 2/5: Whisper, чанков: {len(state['chunk_paths'])}")
        transcribe_chunks(state["chunk_paths"], output_dir=output_dir)
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
    """Извлечь ФИО из транскриптов и сохранить результат."""
    # Кэш JSON предотвращает повторные платные обращения к модели после перезапуска.
    path_output = state["work_dir"] / 'fio_list.json'
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
    path_output = state["work_dir"] / f"students_validated_v{PIPELINE_CACHE_VERSION}.json"

    if not path_output.exists():
        print("Шаг 4/5: сверка ФИО со списком студентов")
        validated: ValidationResult = validate_student_names_with_gpt(
            state["extracted_names"],
            state["students"],
        )
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(validated.model_dump_json(indent=2))

        state.setdefault("logs", []).append(f"Validated names: {validated.model_dump_json()}")
    else:
        print("Шаг 4/5: результаты сверки из кэша")


    with open(path_output, "r", encoding="utf-8") as file:
        state["validated_students"] = ValidationResult.model_validate_json(file.read())

    return state


def node_update_scores(state: PipelineState) -> PipelineState:
    """Начислить баллы уверенно распознанным студентам."""
    score_cache_path = state["work_dir"] / f"sheet_updates_v{PIPELINE_CACHE_VERSION}.json"
    # Локальный кэш не даёт повторно начислить баллы при перезапуске незавершённой лекции.
    if score_cache_path.exists():
        print("Шаг 5/5: начисления из кэша")
        state["sheet_updates"] = json.loads(score_cache_path.read_text(encoding="utf-8"))
        return state

    # Только статус valid исключает начисление баллов по сомнительным совпадениям.
    valid_students = [
        result.matched_name
        for result in state["validated_students"].results
        if result.status == "valid" and result.matched_name
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
