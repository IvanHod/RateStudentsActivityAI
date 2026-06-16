from typing import Iterable

from pathlib import Path
from pydub import AudioSegment

import whisper


def split_audio_to_chunks(path_audio: Path, chunk_ms: int = 60_000, output_dir: Path | None = None) -> list[Path]:
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "audio" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(path_audio)
    chunk_paths = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = output_dir / f"chunk_{i:04d}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths


def transcribe_chunks(
    chunk_paths: Iterable[Path],
    output_dir: Path | None = None,
    model_name: str = "medium",
    language: str = "ru",
    task: str = "transcribe",
) -> tuple[list[str], str]:
    """
    Локальная транскрибация чанков через Whisper.

    Args:
        chunk_paths: список путей к аудио-чанкам
        output_dir: куда сохранять txt-файлы
        model_name: tiny/base/small/medium/large/turbo
        language: язык аудио, например 'ru'
        task: 'transcribe' или 'translate'

    Returns:
        transcript_paths: список путей к сохранённым txt
        full_text: объединённый текст всех чанков
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_name, device='cuda')

    transcript_paths: list[str] = []
    full_text_parts: list[str] = []

    for chunk_path in chunk_paths:
        result = model.transcribe(
            str(chunk_path),
            language=language,
            task=task,
            fp16=False,  # безопаснее для CPU
            verbose=False,
        )

        text = result["text"].strip()

        txt_path = output_dir / f"{chunk_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        transcript_paths.append(str(txt_path))
        full_text_parts.append(text)

    full_text = "\n".join(full_text_parts)
    return transcript_paths, full_text