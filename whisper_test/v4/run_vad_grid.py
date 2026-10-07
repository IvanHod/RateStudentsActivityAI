"""Run the v4 VAD sensitivity grid on a fixed, reproducible audio benchmark."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


VAD_THRESHOLDS = (0.4, 0.5, 0.6)
SPEECH_PADDING_MS = (200, 400, 600)
DEFAULT_MIN_SPEECH_MS = 100
DEFAULT_MIN_SILENCE_MS = 500


@dataclass(frozen=True)
class VadConfiguration:
    """Store one independent VAD setting combination in the v4 grid."""

    threshold: float
    speech_pad_ms: int


def project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    # The runner is stored in whisper_test/v4, so two parents lead to the root.
    return Path(__file__).resolve().parents[2]


def default_input_files(root: Path) -> tuple[Path, ...]:
    """Return the seven audio variants used for the v3-to-v4 comparison."""
    # Keeping the v3 set fixed makes changes attributable to VAD parameters only.
    audio_directory = root / "whisper_test" / "v1" / "audio_test"
    return (
        root / "whisper_test" / "chunk_0049_original.wav",
        audio_directory / "chunk_0049_target_continuous.wav",
        audio_directory / "chunk_0049_target_from_13s.wav",
        audio_directory / "chunk_0049_target_spectral_subtraction.wav",
        audio_directory / "chunk_0049_target_strong_denoise.wav",
        audio_directory / "chunk_0049_target_voice_focus.wav",
        audio_directory / "chunk_0049_whisper_enhanced.wav",
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse runner options while preserving the reproducible default benchmark."""
    # Optional explicit paths let future experiments use several annotated fragments.
    parser = argparse.ArgumentParser(
        description="Запуск сетки параметров Silero VAD для Whisper."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help="Файлы или папки с аудио; без аргументов используется набор v3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Папка с CSV, деталями транскрипции и общей сводкой.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Модель OpenAI Whisper. По умолчанию medium.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Устройство выполнения Whisper. По умолчанию cpu.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Номер CUDA GPU при --device cuda.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=DEFAULT_MIN_SPEECH_MS,
        help="Минимальная длительность речи VAD в мс. По умолчанию 100.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=DEFAULT_MIN_SILENCE_MS,
        help="Пауза, разделяющая речь, в мс. По умолчанию 500.",
    )
    return parser.parse_args(argv)


def vad_configurations() -> tuple[VadConfiguration, ...]:
    """Return every threshold and padding combination planned for v4."""
    # The grid isolates sensitivity from boundary preservation without changing speech length.
    return tuple(
        VadConfiguration(threshold=threshold, speech_pad_ms=speech_pad_ms)
        for threshold in VAD_THRESHOLDS
        for speech_pad_ms in SPEECH_PADDING_MS
    )


def configuration_name(configuration: VadConfiguration) -> str:
    """Build a stable filename stem for one VAD configuration."""
    return (
        f"threshold_{configuration.threshold:.1f}_"
        f"pad_{configuration.speech_pad_ms}ms"
    )


def build_command(
    *,
    root: Path,
    input_files: Sequence[Path],
    output_dir: Path,
    arguments: argparse.Namespace,
    configuration: VadConfiguration,
) -> list[str]:
    """Build one comparator command with only VAD sensitivity settings varied."""
    # Separate artifacts prevent a later configuration from overwriting earlier evidence.
    stem = configuration_name(configuration)
    return [
        sys.executable,
        str(root / "compare_noize_removing.py"),
        *(str(input_file) for input_file in input_files),
        "--output",
        str(output_dir / f"{stem}.csv"),
        "--details",
        str(output_dir / f"{stem}.txt"),
        "--model",
        arguments.model,
        "--device",
        arguments.device,
        "--device-index",
        str(arguments.device_index),
        "--vad-threshold",
        str(configuration.threshold),
        "--vad-min-speech-ms",
        str(arguments.min_speech_ms),
        "--vad-min-silence-ms",
        str(arguments.min_silence_ms),
        "--vad-speech-pad-ms",
        str(configuration.speech_pad_ms),
    ]


def read_best_result(csv_path: Path) -> dict[str, str]:
    """Read the rank-one row from a completed comparator CSV file."""
    # The comparator sorts rows by WER and CER, so the first row is the configuration winner.
    with csv_path.open(encoding="utf-8-sig", newline="") as result_file:
        reader = csv.DictReader(result_file, delimiter=";")
        return next(reader)


def write_summary(
    output_dir: Path,
    results: Sequence[tuple[VadConfiguration, dict[str, str]]],
) -> None:
    """Write a compact cross-configuration table for selecting a VAD profile."""
    # A compact summary supports parameter choice without manually opening nine CSV files.
    summary_path = output_dir / "summary.csv"
    fieldnames = (
        "threshold",
        "speech_pad_ms",
        "best_file",
        "best_wer",
        "best_cer",
        "best_insertions",
        "best_vad_segments",
        "best_vad_speech_seconds",
    )
    with summary_path.open("w", encoding="utf-8-sig", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for configuration, result in results:
            writer.writerow(
                {
                    "threshold": configuration.threshold,
                    "speech_pad_ms": configuration.speech_pad_ms,
                    "best_file": result["file"],
                    "best_wer": result["wer"],
                    "best_cer": result["cer"],
                    "best_insertions": result["insertions"],
                    "best_vad_segments": result["vad_segments"],
                    "best_vad_speech_seconds": result["vad_speech_seconds"],
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the complete VAD grid and produce a selection-oriented summary."""
    # Parsing first lets callers replace the benchmark with several annotated recordings.
    arguments = parse_arguments(argv)
    root = project_root()
    input_files = tuple(arguments.inputs) or default_input_files(root)
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every configuration runs against the same files and decoder settings for fairness.
    results: list[tuple[VadConfiguration, dict[str, str]]] = []
    for configuration in vad_configurations():
        name = configuration_name(configuration)
        print(f"Запуск {name}")
        command = build_command(
            root=root,
            input_files=input_files,
            output_dir=output_dir,
            arguments=arguments,
            configuration=configuration,
        )
        subprocess.run(command, check=True)
        results.append(
            (configuration, read_best_result(output_dir / f"{name}.csv"))
        )

    # The final table makes the grid result useful for a human decision.
    write_summary(output_dir, results)
    print(f"Сводка: {(output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    # A long CPU grid should stop cleanly when the operator interrupts it.
    try:
        main()
    except KeyboardInterrupt:
        print("\nЭксперимент остановлен пользователем.", file=sys.stderr)
        sys.exit(130)
