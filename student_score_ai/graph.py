from pathlib import Path

from langgraph.graph import StateGraph, END
from student_score_ai.state import PipelineState
from student_score_ai.preprocess import video_to_audio
from student_score_ai.transcribe import split_audio_to_chunks, transcribe_chunks
from student_score_ai.extract_names import extract_student_names, NameListWithTime
from student_score_ai.validate_names import validate_student_names_with_gpt, ValidationResult

# from student_score_ai.sheets import update_scores


def node_video_to_audio(state: PipelineState) -> PipelineState:
    path_input: Path = state["video_path"]

    state["audio_path"] = path_output = path_input.parent / f'cache_{path_input.stem}' / f"{path_input.stem}.wav"
    if not path_output.exists():
        video_to_audio(path_input, path_output)
        state.setdefault("logs", []).append(f"Audio saved: {path_output}")

    return state


def node_transcribe(state: PipelineState) -> PipelineState:
    """Транскрибировать через whisper"""
    path_input: Path = state["audio_path"]
    chunks_folder = path_input.parent / f"audio_{path_input.stem}"
    if len(list(chunks_folder.glob('*.wav'))) == 0:
        split_audio_to_chunks(path_input, output_dir=chunks_folder)
    state["chunk_paths"] = list(chunks_folder.glob('*.wav'))

    state.setdefault("logs", []).append(f"Splitted to chunks: '{chunks_folder}'")

    output_dir = path_input.parent / f"transcripts_{path_input.stem}"
    if not output_dir.exists() or len(list(output_dir.glob('*.txt'))) == 0:
        transcribe_chunks(state["chunk_paths"], output_dir=output_dir)
    state["transcript_paths"] = list(output_dir.glob('*.txt'))  # list[Path]

    state["full_transcript"] = ''  # str
    for path in sorted(output_dir.glob('*.txt'), key=lambda p: int(p.stem[-4:])):
        with open(path, "r", encoding="utf-8") as file:
            state["full_transcript"] += file.read() + '\n'

    with open(path_input.parent / 'transcripts.txt', "w", encoding="utf-8") as file:
        file.write(state["full_transcript"])


    state["logs"].append(f"Transcribed chunks: {output_dir}")

    return state


def node_extract_names(state: PipelineState) -> PipelineState:
    path_output: Path = state["audio_path"].parent / 'fio_list.json'
    if not path_output.exists():
        names: NameListWithTime = extract_student_names(state["transcript_paths"])
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(names.model_dump_json(indent=2))

        state.setdefault("logs", []).append(f"Extracted names: {names.model_dump_json()}")

    with open(path_output, "r", encoding="utf-8") as file:
        state["extracted_names"] = NameListWithTime.model_validate_json(file.read())

    return state


def node_validate_names(state: PipelineState) -> PipelineState:
    path_output: Path = state["audio_path"].parent / 'students_validated.json'

    if not path_output.exists():
        validated: ValidationResult = validate_student_names_with_gpt(
            state["extracted_names"],
            state["students_path"],
        )
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(validated.model_dump_json(indent=2))

        state.setdefault("logs", []).append(f"Validated names: {validated.model_dump_json()}")


    with open(path_output, "r", encoding="utf-8") as file:
        state["validated_students"] = ValidationResult.model_validate_json(file.read())

    return state


def node_update_scores(state: PipelineState) -> PipelineState:
    # todo
    return state


def build_graph():
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