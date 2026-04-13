import json
from pathlib import Path

from langgraph.graph import StateGraph, END
from student_score_ai.state import PipelineState
from student_score_ai.preprocess import video_to_audio
from student_score_ai.transcribe import split_audio_to_chunks, transcribe_chunks
from student_score_ai.extract_names import extract_student_names, NameListWithTime
from student_score_ai.validate_names import validate_student_names_with_gpt, ValidationResult

# from student_score_ai.sheets import update_scores

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(state: PipelineState, node_name: str) -> Path:
    video_name = Path(state["video_path"]).name
    return CACHE_DIR / f"{video_name}.{node_name}.json"


def _serialize_cache_payload(state: PipelineState, keys: list[str]) -> dict:
    payload: dict = {}

    for key in keys:
        if key not in state:
            continue

        value = state[key]
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            payload[key] = value

    return payload


def _load_cache_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as cache_file:
        return json.load(cache_file)


def _restore_cached_values(state: PipelineState, payload: dict) -> PipelineState:
    if "audio_path" in payload:
        state["audio_path"] = Path(payload["audio_path"])
    if "chunk_paths" in payload:
        state["chunk_paths"] = [Path(item) for item in payload["chunk_paths"]]
    if "transcript_paths" in payload:
        state["transcript_paths"] = payload["transcript_paths"]
    if "full_transcript" in payload:
        state["full_transcript"] = payload["full_transcript"]
    if "extracted_names" in payload:
        state["extracted_names"] = payload["extracted_names"]
    if "validated_students" in payload:
        state["validated_students"] = payload["validated_students"]
    if "sheet_updates" in payload:
        state["sheet_updates"] = payload["sheet_updates"]
    return state


def _cached_node(
    state: PipelineState,
    node_name: str,
    runner,
    cache_keys: list[str],
) -> PipelineState:
    cache_file = _cache_path(state, node_name)
    state.setdefault("logs", [])

    if cache_file.exists():
        payload = _load_cache_payload(cache_file)
        _restore_cached_values(state, payload)
        state["logs"].append(f"{node_name}: loaded from cache {cache_file.name}")
        return state

    state = runner(state)
    payload = _serialize_cache_payload(state, cache_keys)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    state["logs"].append(f"{node_name}: cached to {cache_file.name}")
    return state


def node_video_to_audio(state: PipelineState) -> PipelineState:
    path_input: Path = state["video_path"]

    state["audio_path"] = path_output = path_input.parent / 'cache' / f"{path_input.stem}.wav"
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
            state["full_transcript"] += file.read()

    state["logs"].append(f"Transcribed chunks: {output_dir}")

    return state


def node_extract_names(state: PipelineState) -> PipelineState:
    path_output: Path = state["audio_path"].parent / 'fio_list.json'
    if not path_output.exists():
        names: NameListWithTime = extract_student_names(state["transcript_paths"])
        with open(path_output, "w", encoding="utf-8") as file:
            file.write(names.model_dump_json())

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
            file.write(validated.model_dump_json())

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