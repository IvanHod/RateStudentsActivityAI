#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import whisper


REFERENCE = (
    "Да. Не факт, в зависимости от того как ты сделаешь, но если складывать - то да. "
    "Если складывать - то да. Хотя от компилятора тоже зависит. "
    "Смотрите, неявное преобразование типов. Например, вы пишете int какая то переменная "
    "var равно 3.14 что будет лежать в var? Все согласны? Почему единица? "
    "Где вы там единицу вообще нашли?"
)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


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


def transcribe_file(
    model,
    audio_path: Path,
    *,
    start: float,
    end: float | None,
    beam_size: int,
) -> tuple[str, str, float]:

    if end is None:
        clip_timestamps = f"{start}"
    else:
        clip_timestamps = f"{start},{end}"

    result = model.transcribe(
        str(audio_path),
        language="ru",
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        clip_timestamps=clip_timestamps,
        verbose=False,
    )

    text = result["text"].strip()
    language = result.get("language", "ru")

    # Обычный whisper не возвращает language_probability
    language_probability = 1.0

    return text, language, language_probability


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
    parser = argparse.ArgumentParser(
        description="Сравнение качества нескольких аудиоверсий через faster-whisper + WER/CER."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Аудиофайл или папка с несколькими вариантами аудио.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("whisper_comparison.csv"),
        help="Итоговая CSV-таблица.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("whisper_comparison_details.txt"),
        help="Подробный TXT с выравниванием ошибок.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Модель faster-whisper, по умолчанию large-v3.",
    )
    parser.add_argument(
        "--device",
        default="auto",
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
        "--compute-type",
        default="default",
        help='Например: "default", "float16", "int8", "int8_float16".',
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
        default=13.0,
        help="Начало оцениваемого фрагмента в секундах. По умолчанию 13.0.",
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
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start должен быть >= 0")
    if args.end is not None and args.end <= args.start:
        parser.error("--end должен быть больше --start")

    reference = REFERENCE
    if args.reference_file is not None:
        reference = args.reference_file.read_text(encoding="utf-8").strip()

    files = collect_audio_files(args.input)

    print(f"Модель: {args.model}")
    print(f"Файлов: {len(files)}")
    print(f"Фрагмент: {args.start:.2f} с -> {'конец' if args.end is None else f'{args.end:.2f} с'}")
    print("Загружаю Whisper...")

    device = args.device

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        device = f"cuda:{args.device_index}"

    model = whisper.load_model(args.model, device=device)

    rows = []
    details = []

    for i, audio_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {audio_path.name}")

        try:
            text, detected_language, language_probability = transcribe_file(
                model,
                audio_path,
                start=args.start,
                end=args.end,
                beam_size=args.beam_size,
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
