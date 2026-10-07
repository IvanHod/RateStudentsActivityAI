#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jiwer
import numpy as np
import whisper


REFERENCE = (
    "Да. Не факт, в зависимости от того как ты сделаешь, но если складывать - то да. "
    "Если складывать - то да. Хотя от компилятора тоже зависит. "
    "Смотрите, неявное преобразование типов. Например, вы пишете int какая то переменная "
    "var равно 3.14 что будет лежать в var? Все согласны? Почему единица? "
    "Где вы там единицу вообще нашли?"
)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
WHISPER_SAMPLE_RATE = 16_000
SUBJECT_TERMS_PROMPT = (
    "алгоритмы, структуры данных, компилятор, преобразование типов, "
    "типы данных, int, переменная, var"
)


@dataclass(frozen=True)
class VadSettings:
    """Store VAD thresholds used to exclude non-speech before transcription."""

    enabled: bool
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int


@dataclass(frozen=True)
class VadStatistics:
    """Describe the amount of speech retained by VAD for one input file."""

    segments: int
    speech_seconds: float


def normalize_text(text: str) -> str:
    """
    Нормализация для честного сравнения ASR:
    - lowercase;
    - ё -> е;
    - Unicode normalization;
    - 3,14 и 3.14 становятся "3 14";
    - удаление пунктуации;
    - схлопывание пробелов.

    Не преобразует числа в слова, чтобы не вносить дополнительную эвристику.
    """
    text = unicodedata.normalize("NFKC", text.lower())
    text = text.replace("ё", "е")

    # Десятичный разделитель считаем несущественной пунктуацией.
    text = re.sub(r"(?<=\d)[.,](?=\d)", " ", text)

    # Всё, кроме букв/цифр/underscore и пробелов -> пробел.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collect_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Неподдерживаемый формат: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Не найден путь: {input_path}")

    files = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"В {input_path} не найдено аудиофайлов: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return files


def load_vad_model() -> Any:
    """Load Silero VAD or explain how to install the optional runtime dependency."""
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен silero-vad. Выполните `uv sync` в корне проекта."
        ) from exc

    # Load the detector once because model initialization is expensive for every WAV.
    return load_silero_vad()


def merge_speech_intervals(
    timestamps: Sequence[dict[str, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping padded VAD intervals to avoid duplicating audio samples."""
    if not timestamps:
        return []

    # Padding may make adjacent VAD intervals overlap, so normalize them before joining.
    intervals: list[tuple[int, int]] = []
    for timestamp in timestamps:
        start = timestamp["start"]
        end = timestamp["end"]

        if intervals and start <= intervals[-1][1]:
            previous_start, previous_end = intervals[-1]
            intervals[-1] = (previous_start, max(previous_end, end))
        else:
            intervals.append((start, end))

    return intervals


def extract_speech_audio(
    audio_path: Path,
    *,
    start: float,
    end: float | None,
    vad_model: Any,
    vad_settings: VadSettings,
) -> tuple[np.ndarray, VadStatistics]:
    """Return only VAD-detected speech from the requested interval of an audio file."""
    try:
        import torch
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "Для VAD требуются torch и silero-vad. Выполните `uv sync`."
        ) from exc

    # Decode with Whisper so VAD and ASR receive the same mono 16 kHz waveform.
    audio = whisper.load_audio(str(audio_path))
    start_sample = int(start * WHISPER_SAMPLE_RATE)
    end_sample = (
        len(audio)
        if end is None
        else min(int(end * WHISPER_SAMPLE_RATE), len(audio))
    )
    selected_audio = audio[start_sample:end_sample]

    if selected_audio.size == 0:
        raise ValueError("В выбранном интервале нет аудиосэмплов.")

    # Detect speech before ASR so silent or noisy regions cannot trigger decoding loops.
    timestamps = get_speech_timestamps(
        torch.from_numpy(selected_audio),
        vad_model,
        threshold=vad_settings.threshold,
        sampling_rate=WHISPER_SAMPLE_RATE,
        min_speech_duration_ms=vad_settings.min_speech_duration_ms,
        min_silence_duration_ms=vad_settings.min_silence_duration_ms,
        speech_pad_ms=vad_settings.speech_pad_ms,
        return_seconds=False,
    )
    intervals = merge_speech_intervals(timestamps)

    if not intervals:
        raise ValueError("VAD не обнаружил речь в выбранном интервале.")

    # Concatenation prevents Whisper from decoding long silences while retaining word edges.
    speech_audio = np.concatenate(
        [
            selected_audio[segment_start:segment_end]
            for segment_start, segment_end in intervals
        ]
    )
    speech_seconds = sum(segment_end - segment_start for segment_start, segment_end in intervals)

    return speech_audio, VadStatistics(
        segments=len(intervals),
        speech_seconds=speech_seconds / WHISPER_SAMPLE_RATE,
    )


def transcribe_file(
    model: whisper.Whisper,
    audio_path: Path,
    *,
    start: float,
    end: float | None,
    beam_size: int,
    initial_prompt: str | None,
    vad_model: Any | None,
    vad_settings: VadSettings,
) -> tuple[str, str, float, VadStatistics | None]:
    """Transcribe one selected audio interval with stable Whisper settings."""

    # VAD receives the selected waveform and removes non-speech before Whisper decodes it.
    vad_statistics: VadStatistics | None = None
    if vad_settings.enabled:
        if vad_model is None:
            raise ValueError("Модель VAD не загружена.")

        audio_input, vad_statistics = extract_speech_audio(
            audio_path,
            start=start,
            end=end,
            vad_model=vad_model,
            vad_settings=vad_settings,
        )
    else:
        # Preserve the old timestamp-based path to make VAD-on and VAD-off comparable.
        audio_input = str(audio_path)
        if end is None:
            clip_timestamps = f"{start}"
        else:
            clip_timestamps = f"{start},{end}"

    # Keep decoder settings identical across both modes so VAD is the only changed factor.
    transcription_options: dict[str, Any] = {
        "language": "ru",
        "task": "transcribe",
        "initial_prompt": initial_prompt,
        "beam_size": beam_size,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "verbose": False,
    }
    if not vad_settings.enabled:
        transcription_options["clip_timestamps"] = clip_timestamps

    result = model.transcribe(audio_input, **transcription_options)

    text = result["text"].strip()
    language = result.get("language", "ru")

    # Обычный whisper не возвращает language_probability
    language_probability = 1.0

    return text, language, language_probability, vad_statistics


def evaluate(reference: str, hypothesis: str) -> dict:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    word_result = jiwer.process_words(ref_norm, hyp_norm)
    char_result = jiwer.process_characters(ref_norm, hyp_norm)

    ref_words = len(ref_norm.split())
    word_errors = (
            word_result.substitutions
            + word_result.deletions
            + word_result.insertions
    )

    return {
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
        "wer": word_result.wer,
        "cer": char_result.cer,
        "word_errors": word_errors,
        "substitutions": word_result.substitutions,
        "deletions": word_result.deletions,
        "insertions": word_result.insertions,
        "hits": word_result.hits,
        "reference_words": ref_words,
        "alignment": jiwer.visualize_alignment(word_result),
    }


def main() -> None:
    """Compare Whisper transcriptions of audio variants against one reference text."""
    parser = argparse.ArgumentParser(
        description="Сравнение качества аудиоверсий через OpenAI Whisper + WER/CER."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Аудиофайл или папка с несколькими вариантами аудио.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("whisper_test/v1/whisper_comparison.csv"),
        help="Итоговая CSV-таблица.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("whisper_test/v1/whisper_comparison_details.txt"),
        help="Подробный TXT с выравниванием ошибок.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Модель OpenAI Whisper, по умолчанию medium.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Устройство: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Номер CUDA GPU, если --device cuda.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size Whisper. Для честного сравнения одинаков для всех файлов.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0,
        help="Начало оцениваемого фрагмента в секундах. По умолчанию 0.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Конец фрагмента в секундах. Если не задан — до конца.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Необязательно: TXT-файл с эталоном вместо встроенного текста.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--initial-prompt",
        default=None,
        help="Короткий список ожидаемых терминов. По умолчанию подсказка отключена.",
    )
    prompt_group.add_argument(
        "--use-subject-terms-prompt",
        action="store_true",
        help="Использовать встроенный словарь терминов по алгоритмам и программированию.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить Silero VAD и распознавать весь выбранный аудиоинтервал.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.6,
        help="Порог уверенности Silero VAD: 0..1; выше — строже. По умолчанию 0.6.",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="Минимальная длительность речи для VAD в мс. По умолчанию 250.",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="Пауза, разделяющая речь, для VAD в мс. По умолчанию 500.",
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=200,
        help="Запас до и после речи для VAD в мс. По умолчанию 200.",
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start должен быть >= 0")
    if args.end is not None and args.end <= args.start:
        parser.error("--end должен быть больше --start")
    if not 0 < args.vad_threshold < 1:
        parser.error("--vad-threshold должен быть больше 0 и меньше 1")
    if min(
        args.vad_min_speech_ms,
        args.vad_min_silence_ms,
        args.vad_speech_pad_ms,
    ) < 0:
        parser.error("Параметры длительности VAD должны быть >= 0")

    # Keep prompting opt-in because the evaluated recordings showed prompt repetition.
    initial_prompt = (
        SUBJECT_TERMS_PROMPT if args.use_subject_terms_prompt else args.initial_prompt
    )
    vad_settings = VadSettings(
        enabled=not args.no_vad,
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
    )

    reference = REFERENCE
    if args.reference_file is not None:
        reference = args.reference_file.read_text(encoding="utf-8").strip()

    files = collect_audio_files(args.input)

    print(f"Модель: {args.model}")
    print(f"Файлов: {len(files)}")
    print(f"Фрагмент: {args.start:.2f} с -> {'конец' if args.end is None else f'{args.end:.2f} с'}")
    print(f"Initial prompt: {initial_prompt or 'отключён'}")
    print(f"VAD: {'включён' if vad_settings.enabled else 'отключён'}")
    print("Загружаю Whisper...")

    device = args.device

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        device = f"cuda:{args.device_index}"

    # Initialize VAD once to keep every audio variant evaluated by the same detector.
    vad_model = load_vad_model() if vad_settings.enabled else None
    model = whisper.load_model(args.model, device=device)

    rows = []
    details = []

    for i, audio_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {audio_path.name}")

        try:
            text, detected_language, language_probability, vad_statistics = transcribe_file(
                model,
                audio_path,
                start=args.start,
                end=args.end,
                beam_size=args.beam_size,
                initial_prompt=initial_prompt,
                vad_model=vad_model,
                vad_settings=vad_settings,
            )
            metrics = evaluate(reference, text)

            row = {
                "file": audio_path.name,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "word_errors": metrics["word_errors"],
                "substitutions": metrics["substitutions"],
                "deletions": metrics["deletions"],
                "insertions": metrics["insertions"],
                "hits": metrics["hits"],
                "reference_words": metrics["reference_words"],
                "detected_language": detected_language,
                "language_probability": language_probability,
                "vad_segments": vad_statistics.segments if vad_statistics else "",
                "vad_speech_seconds": (
                    vad_statistics.speech_seconds if vad_statistics else ""
                ),
                "transcription": text,
                "normalized_transcription": metrics["hypothesis_normalized"],
                "error": "",
            }

            details.append(
                "\n".join(
                    [
                        "=" * 100,
                        f"FILE: {audio_path.name}",
                        f"WER: {metrics['wer']:.4f} ({metrics['wer'] * 100:.2f}%)",
                        f"CER: {metrics['cer']:.4f} ({metrics['cer'] * 100:.2f}%)",
                        (
                            "Ошибки: "
                            f"S={metrics['substitutions']} "
                            f"D={metrics['deletions']} "
                            f"I={metrics['insertions']} "
                            f"H={metrics['hits']}"
                        ),
                        (
                            "VAD: "
                            f"сегментов={vad_statistics.segments}; "
                            f"речи={vad_statistics.speech_seconds:.2f} с"
                            if vad_statistics
                            else "VAD: отключён"
                        ),
                        "",
                        "WHISPER:",
                        text,
                        "",
                        "NORMALIZED:",
                        metrics["hypothesis_normalized"],
                        "",
                        "ALIGNMENT:",
                        metrics["alignment"],
                        ]
                )
            )

        except Exception as exc:
            row = {
                "file": audio_path.name,
                "wer": float("inf"),
                "cer": float("inf"),
                "word_errors": "",
                "substitutions": "",
                "deletions": "",
                "insertions": "",
                "hits": "",
                "reference_words": "",
                "detected_language": "",
                "language_probability": "",
                "vad_segments": "",
                "vad_speech_seconds": "",
                "transcription": "",
                "normalized_transcription": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            details.append(
                "\n".join(
                    [
                        "=" * 100,
                        f"FILE: {audio_path.name}",
                        f"ERROR: {type(exc).__name__}: {exc}",
                        ]
                )
            )

        rows.append(row)

    # Лучший результат сверху.
    rows.sort(key=lambda r: (r["wer"], r["cer"]))

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "file",
        "wer",
        "cer",
        "word_errors",
        "substitutions",
        "deletions",
        "insertions",
        "hits",
        "reference_words",
        "detected_language",
        "language_probability",
        "vad_segments",
        "vad_speech_seconds",
        "transcription",
        "normalized_transcription",
        "error",
    ]

    # utf-8-sig + ; удобнее открывается Excel/LibreOffice в русской локали.
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    with args.details.open("w", encoding="utf-8") as f:
        # Record inference settings so each comparison can be reproduced later.
        f.write("CONFIGURATION:\n")
        f.write(f"model: {args.model}\n")
        f.write("language: ru\n")
        f.write(f"beam_size: {args.beam_size}\n")
        f.write(f"start: {args.start}\n")
        f.write(f"end: {args.end if args.end is not None else 'end of file'}\n")
        f.write(f"initial_prompt: {initial_prompt or 'disabled'}\n\n")
        f.write(f"vad_enabled: {vad_settings.enabled}\n")
        f.write(f"vad_threshold: {vad_settings.threshold}\n")
        f.write(f"vad_min_speech_ms: {vad_settings.min_speech_duration_ms}\n")
        f.write(f"vad_min_silence_ms: {vad_settings.min_silence_duration_ms}\n")
        f.write(f"vad_speech_pad_ms: {vad_settings.speech_pad_ms}\n\n")
        f.write("REFERENCE:\n")
        f.write(reference)
        f.write("\n\nNORMALIZED REFERENCE:\n")
        f.write(normalize_text(reference))
        f.write("\n\n")
        f.write("\n\n".join(details))

    print("\nРезультаты:")
    for row in rows:
        if row["error"]:
            print(f"{row['rank']:>2}. {row['file']}: ERROR — {row['error']}")
        else:
            print(
                f"{row['rank']:>2}. {row['file']}: "
                f"WER={row['wer'] * 100:.2f}% | "
                f"CER={row['cer'] * 100:.2f}% | "
                f"errors={row['word_errors']}"
            )

    print(f"\nCSV: {args.output.resolve()}")
    print(f"Details: {args.details.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        sys.exit(130)
