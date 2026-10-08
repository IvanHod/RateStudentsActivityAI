#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import httpx
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
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SPEECH_MS = 100
DEFAULT_VAD_MIN_SILENCE_MS = 500
DEFAULT_VAD_SPEECH_PAD_MS = 200
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 900.0
DEFAULT_GIGAAM_MAX_SEGMENT_SECONDS = 24.0
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


@dataclass(frozen=True)
class TranscriptionSettings:
    """Store decoding inputs shared by every ASR backend."""

    language: str
    initial_prompt: str | None
    beam_size: int


@dataclass(frozen=True)
class TranscriptionResult:
    """Describe the normalized response returned by an ASR backend."""

    text: str
    language: str
    language_probability: float


class TranscriptionBackend(Protocol):
    """Define the adapter contract required to evaluate an ASR provider."""

    name: str
    model_name: str

    def transcribe(
        self,
        audio: np.ndarray,
        settings: TranscriptionSettings,
    ) -> TranscriptionResult:
        """Transcribe a mono 16 kHz waveform with provider-specific transport."""


class WhisperBackend:
    """Adapt local OpenAI Whisper to the common ASR backend contract."""

    name = "whisper"

    def __init__(self, model: whisper.Whisper, model_name: str) -> None:
        """Store the loaded Whisper model and its user-facing identifier."""
        self._model = model
        self.model_name = model_name

    def transcribe(
        self,
        audio: np.ndarray,
        settings: TranscriptionSettings,
    ) -> TranscriptionResult:
        """Run local Whisper with deterministic settings used by prior experiments."""
        # Stable decoding makes backend and audio comparisons repeatable.
        options: dict[str, Any] = {
            "language": settings.language,
            "task": "transcribe",
            "initial_prompt": settings.initial_prompt,
            "beam_size": settings.beam_size,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "verbose": False,
        }
        result = self._model.transcribe(audio, **options)
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", settings.language),
            language_probability=1.0,
        )


class OllamaBackend:
    """Adapt Ollama's OpenAI-compatible audio transcription endpoint."""

    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout_seconds: float) -> None:
        """Store the remote endpoint and ASR model selected for the experiment."""
        self._base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._timeout_seconds = timeout_seconds

    def transcribe(
        self,
        audio: np.ndarray,
        settings: TranscriptionSettings,
    ) -> TranscriptionResult:
        """Upload a temporary WAV to Ollama and normalize its JSON response."""
        # A WAV file gives HTTP ASR backends the exact VAD-filtered samples Whisper sees.
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            write_wav(audio_file.name, audio)
            response = self._send_transcription_request(audio_file.name, settings)

        # API validation turns malformed provider responses into actionable benchmark errors.
        try:
            payload = response.json()
            text = payload["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Ollama вернула ответ без строкового поля 'text': "
                f"{response.text[:500]}"
            ) from exc

        if not isinstance(text, str):
            raise RuntimeError("Ollama вернула поле 'text' не строкового типа.")

        return TranscriptionResult(
            text=normalize_ollama_text(text),
            language=settings.language,
            language_probability=1.0,
        )

    def _send_transcription_request(
        self,
        audio_path: str,
        settings: TranscriptionSettings,
    ) -> httpx.Response:
        """Send one multipart transcription request to the configured Ollama server."""
        # Multipart is required by the OpenAI-compatible /v1/audio/transcriptions API.
        form_data: dict[str, str] = {
            "model": self.model_name,
            "language": settings.language,
            "response_format": "json",
        }
        if settings.initial_prompt:
            form_data["prompt"] = settings.initial_prompt

        try:
            with open(audio_path, "rb") as audio_file:
                response = httpx.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    data=form_data,
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    timeout=self._timeout_seconds,
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Не удалось выполнить Ollama ASR-запрос. "
                f"Проверьте --ollama-base-url ({self._base_url}), модель "
                f"'{self.model_name}' и доступность /v1/audio/transcriptions."
            ) from exc

        return response


class GigaAMBackend:
    """Adapt a local GigaAM short-form model to the common ASR contract."""

    name = "gigaam"

    def __init__(
        self,
        model: Any,
        model_name: str,
        max_segment_seconds: float,
    ) -> None:
        """Store a loaded GigaAM model and its safe short-form segment limit."""
        self._model = model
        self.model_name = model_name
        self._max_segment_samples = int(
            max_segment_seconds * WHISPER_SAMPLE_RATE
        )

    def transcribe(
        self,
        audio: np.ndarray,
        settings: TranscriptionSettings,
    ) -> TranscriptionResult:
        """Transcribe VAD-ready audio in chunks accepted by GigaAM's short API."""
        del settings

        # GigaAM rejects recordings longer than 25 seconds, while VAD may join speech
        # fragments into a longer waveform. Splitting at 24 seconds preserves a margin.
        chunks = split_audio_for_gigaam(audio, self._max_segment_samples)
        text = " ".join(self._transcribe_chunk(chunk) for chunk in chunks).strip()
        return TranscriptionResult(
            text=text,
            language="ru",
            language_probability=1.0,
        )

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Persist one safe waveform chunk for GigaAM's path-based API."""
        # The official API accepts a WAV path, so a temporary lossless file prevents
        # a second decoding path from changing the VAD-filtered signal.
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            write_wav(audio_file.name, audio)
            result = self._model.transcribe(audio_file.name)

        text = getattr(result, "text", result)
        if not isinstance(text, str):
            raise RuntimeError("GigaAM вернула транскрипцию не строкового типа.")
        return text.strip()


def normalize_ollama_text(text: str) -> str:
    """Remove optional ASR metadata wrappers emitted by compatible Ollama models."""
    # Qwen3-ASR returns language metadata before the actual <asr_text> payload.
    if "<asr_text>" in text:
        text = text.split("<asr_text>", maxsplit=1)[1]
    return text.replace("</asr_text>", "").strip()


def write_wav(path: str, audio: np.ndarray) -> None:
    """Write a mono float waveform as a PCM WAV file for HTTP ASR backends."""
    # Clipping prevents integer overflow while preserving Whisper's normalized amplitude.
    pcm_audio = np.clip(audio, -1.0, 1.0)
    pcm_audio = (pcm_audio * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(np.dtype(np.int16).itemsize)
        wav_file.setframerate(WHISPER_SAMPLE_RATE)
        wav_file.writeframes(pcm_audio.tobytes())


def split_audio_for_gigaam(
    audio: np.ndarray,
    max_segment_samples: int,
) -> tuple[np.ndarray, ...]:
    """Split a waveform into non-empty chunks below GigaAM's short-form limit."""
    if max_segment_samples <= 0:
        raise ValueError("Лимит сегмента GigaAM должен быть больше нуля.")
    if audio.size == 0:
        raise ValueError("Нельзя отправить GigaAM пустой аудиофрагмент.")

    # Exact sample boundaries avoid resampling and guarantee every part remains safe.
    return tuple(
        audio[start : start + max_segment_samples]
        for start in range(0, len(audio), max_segment_samples)
    )


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


def collect_audio_files(input_paths: Sequence[Path]) -> list[Path]:
    """Return unique supported audio files from the requested files and folders."""
    # Several explicit files make it possible to reproduce a benchmark exactly.
    files: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Неподдерживаемый формат: {input_path.suffix}")
            files.add(input_path)
            continue

        if not input_path.is_dir():
            raise FileNotFoundError(f"Не найден путь: {input_path}")

        # Directory input remains convenient when every audio file is in scope.
        files.update(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not files:
        raise FileNotFoundError(
            "Не найдены аудиофайлы: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )
    return sorted(files)


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
    audio: np.ndarray,
    *,
    vad_model: Any,
    vad_settings: VadSettings,
) -> tuple[np.ndarray, VadStatistics]:
    """Return only VAD-detected speech from an already selected audio interval."""
    try:
        import torch
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "Для VAD требуются torch и silero-vad. Выполните `uv sync`."
        ) from exc

    # Detect speech before ASR so silent or noisy regions cannot trigger decoding loops.
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
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
            audio[segment_start:segment_end]
            for segment_start, segment_end in intervals
        ]
    )
    speech_seconds = sum(segment_end - segment_start for segment_start, segment_end in intervals)

    return speech_audio, VadStatistics(
        segments=len(intervals),
        speech_seconds=speech_seconds / WHISPER_SAMPLE_RATE,
    )


def load_audio_interval(
    audio_path: Path,
    *,
    start: float,
    end: float | None,
) -> np.ndarray:
    """Load one mono 16 kHz interval so every backend receives the same signal."""
    # Explicit slicing makes APIs without Whisper's clip_timestamps feature comparable.
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
    return selected_audio


def prepare_audio(
    audio_path: Path,
    *,
    start: float,
    end: float | None,
    vad_model: Any | None,
    vad_settings: VadSettings,
) -> tuple[np.ndarray, VadStatistics | None]:
    """Return one selected waveform and optional VAD statistics for any backend."""
    # Shared preparation prevents different providers from receiving different intervals.
    selected_audio = load_audio_interval(audio_path, start=start, end=end)
    if not vad_settings.enabled:
        return selected_audio, None

    if vad_model is None:
        raise ValueError("Модель VAD не загружена.")

    return extract_speech_audio(
        selected_audio,
        vad_model=vad_model,
        vad_settings=vad_settings,
    )


def transcribe_file(
    backend: TranscriptionBackend,
    audio_path: Path,
    *,
    start: float,
    end: float | None,
    transcription_settings: TranscriptionSettings,
    vad_model: Any | None,
    vad_settings: VadSettings,
) -> tuple[str, str, float, VadStatistics | None]:
    """Transcribe one audio interval through the selected ASR backend."""
    # Audio is prepared once so VAD remains an independent variable across providers.
    audio, vad_statistics = prepare_audio(
        audio_path,
        start=start,
        end=end,
        vad_model=vad_model,
        vad_settings=vad_settings,
    )
    result = backend.transcribe(audio, transcription_settings)
    return result.text, result.language, result.language_probability, vad_statistics


def evaluate(reference: str, hypothesis: str) -> dict[str, str | float | int]:
    """Calculate normalized word and character recognition metrics."""
    # Identical normalization keeps punctuation from affecting ASR quality metrics.
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Both granularities are needed because WER alone hides character-level changes.
    word_result = jiwer.process_words(ref_norm, hyp_norm)
    char_result = jiwer.process_characters(ref_norm, hyp_norm)

    # Error components reveal whether a setting loses speech or hallucinates additions.
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


def resolve_whisper_device(device: str, device_index: int) -> str:
    """Resolve a requested local Whisper device to its execution identifier."""
    # CUDA auto-detection is relevant only when this process loads local Whisper weights.
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return f"cuda:{device_index}"
    return device


def create_backend(
    *,
    backend_name: str,
    model_name: str,
    device: str,
    device_index: int,
    ollama_base_url: str,
    ollama_timeout_seconds: float,
    gigaam_max_segment_seconds: float,
) -> TranscriptionBackend:
    """Create the requested ASR adapter without changing benchmark orchestration."""
    # Provider-specific initialization remains isolated from VAD and metric calculation.
    if backend_name == WhisperBackend.name:
        whisper_device = resolve_whisper_device(device, device_index)
        model = whisper.load_model(model_name, device=whisper_device)
        return WhisperBackend(model, model_name)
    if backend_name == OllamaBackend.name:
        return OllamaBackend(
            base_url=ollama_base_url,
            model_name=model_name,
            timeout_seconds=ollama_timeout_seconds,
        )
    if backend_name == GigaAMBackend.name:
        try:
            import gigaam
        except ImportError as exc:
            raise RuntimeError(
                "Не установлен GigaAM. Выполните `uv sync` в корне проекта."
            ) from exc

        # CPU is explicit here because the benchmark environment has no CUDA device.
        model = gigaam.load_model(model_name, fp16_encoder=False, device="cpu")
        return GigaAMBackend(
            model=model,
            model_name=model_name,
            max_segment_seconds=gigaam_max_segment_seconds,
        )
    raise ValueError(f"Неподдерживаемый ASR-бэкенд: {backend_name}")


def main() -> None:
    """Compare audio variants through a selected ASR backend against one reference."""
    parser = argparse.ArgumentParser(
        description="Сравнение качества аудиоверсий через ASR-бэкенды + WER/CER."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Аудиофайлы и/или папки с несколькими вариантами аудио.",
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
        "--backend",
        default=WhisperBackend.name,
        choices=(WhisperBackend.name, OllamaBackend.name, GigaAMBackend.name),
        help="ASR-бэкенд. По умолчанию whisper.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Имя модели выбранного бэкенда. По умолчанию medium.",
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
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=(
            "Базовый URL Ollama API. "
            f"По умолчанию {DEFAULT_OLLAMA_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
        default=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        help=(
            "Таймаут одного Ollama-запроса в секундах. "
            f"По умолчанию {DEFAULT_OLLAMA_TIMEOUT_SECONDS:.0f}."
        ),
    )
    parser.add_argument(
        "--gigaam-max-segment-seconds",
        type=float,
        default=DEFAULT_GIGAAM_MAX_SEGMENT_SECONDS,
        help=(
            "Максимальная длительность WAV-части для GigaAM (< 25 с). "
            f"По умолчанию {DEFAULT_GIGAAM_MAX_SEGMENT_SECONDS:.0f}."
        ),
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
        default=DEFAULT_VAD_THRESHOLD,
        help=(
            "Порог уверенности Silero VAD: 0..1; выше — строже. "
            f"По умолчанию {DEFAULT_VAD_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=DEFAULT_VAD_MIN_SPEECH_MS,
        help=(
            "Минимальная длительность речи для VAD в мс. "
            f"По умолчанию {DEFAULT_VAD_MIN_SPEECH_MS}."
        ),
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=DEFAULT_VAD_MIN_SILENCE_MS,
        help=(
            "Пауза, разделяющая речь, для VAD в мс. "
            f"По умолчанию {DEFAULT_VAD_MIN_SILENCE_MS}."
        ),
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=DEFAULT_VAD_SPEECH_PAD_MS,
        help=(
            "Запас до и после речи для VAD в мс. "
            f"По умолчанию {DEFAULT_VAD_SPEECH_PAD_MS}."
        ),
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
    if args.ollama_timeout_seconds <= 0:
        parser.error("--ollama-timeout-seconds должен быть больше 0")
    if not 0 < args.gigaam_max_segment_seconds < 25:
        parser.error("--gigaam-max-segment-seconds должен быть больше 0 и меньше 25")

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

    transcription_settings = TranscriptionSettings(
        language="ru",
        initial_prompt=initial_prompt,
        beam_size=args.beam_size,
    )
    reference = REFERENCE
    if args.reference_file is not None:
        reference = args.reference_file.read_text(encoding="utf-8").strip()

    files = collect_audio_files(args.input)

    print(f"Бэкенд: {args.backend}")
    print(f"Модель: {args.model}")
    print(f"Файлов: {len(files)}")
    print(f"Фрагмент: {args.start:.2f} с -> {'конец' if args.end is None else f'{args.end:.2f} с'}")
    print(f"Initial prompt: {initial_prompt or 'отключён'}")
    print(f"VAD: {'включён' if vad_settings.enabled else 'отключён'}")
    if args.backend == OllamaBackend.name:
        print(f"Ollama API: {args.ollama_base_url}")
    if args.backend == GigaAMBackend.name:
        print(
            "Лимит части GigaAM: "
            f"{args.gigaam_max_segment_seconds:.2f} с"
        )

    # Initialize shared services once so every variant sees identical dependencies.
    vad_model = load_vad_model() if vad_settings.enabled else None
    print(f"Инициализирую {args.backend}...")
    backend = create_backend(
        backend_name=args.backend,
        model_name=args.model,
        device=args.device,
        device_index=args.device_index,
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_seconds=args.ollama_timeout_seconds,
        gigaam_max_segment_seconds=args.gigaam_max_segment_seconds,
    )

    rows = []
    details = []

    for i, audio_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {audio_path.name}")

        try:
            text, detected_language, language_probability, vad_statistics = transcribe_file(
                backend,
                audio_path,
                start=args.start,
                end=args.end,
                transcription_settings=transcription_settings,
                vad_model=vad_model,
                vad_settings=vad_settings,
            )
            metrics = evaluate(reference, text)

            row = {
                "backend": backend.name,
                "model": backend.model_name,
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
                        "TRANSCRIPTION:",
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
                "backend": args.backend,
                "model": args.model,
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
        "backend",
        "model",
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
        f.write(f"backend: {backend.name}\n")
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
        if args.backend == OllamaBackend.name:
            f.write(f"ollama_base_url: {args.ollama_base_url}\n")
            f.write(f"ollama_timeout_seconds: {args.ollama_timeout_seconds}\n\n")
        if args.backend == GigaAMBackend.name:
            f.write(
                "gigaam_max_segment_seconds: "
                f"{args.gigaam_max_segment_seconds}\n\n"
            )
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
