"""Точка входа: синхронизация лекций и начисление баллов."""

import json
import os
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

# Settings must be available before graph modules create their language-model clients.
load_dotenv(".env")

from student_score_ai.archives import ARCHIVE_EXTENSIONS, extract_videos
from student_score_ai.graph import build_graph
from student_score_ai.preprocess import compress_video
from student_score_ai.sheets import get_students
from student_score_ai.yandex_disk import RemoteVideo, download_video, iter_public_videos


YANDEX_PUBLIC_FOLDER_URL: Final = "https://disk.yandex.ru/d/TKX3hy06-WFSpA"
OUTPUT_DIRECTORY: Final = Path("output")
COMPLETION_FILE: Final = "completed.json"
UNCONFIRMED_LOG: Final = "unconfirmed_students.jsonl"
DATE_PATTERN: Final = re.compile(r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)")
RETENTION_PERIOD: Final = timedelta(days=14)
PIPELINE_VERSION: Final = 2


def main() -> None:
    """Скачать и обработать все ещё не завершённые лекции публичной папки."""
    settings = _get_settings()

    # Очистка запускается заранее, чтобы старые результаты не занимали место во время загрузки.
    _cleanup_old_results(OUTPUT_DIRECTORY)
    students = get_students(
        service_account_path=settings["service_account_path"],
        sheet_id=settings["sheet_id"],
        worksheet_name=settings["worksheet_name"],
        start_row=settings["score_start_row"],
    )
    if not students:
        raise RuntimeError("В диапазоне A4:A не найдено ни одного студента")

    # Один граф обслуживает все видео, а папка результата отделяет кэш каждой лекции.
    app = build_graph()
    lectures = list(iter_public_videos(YANDEX_PUBLIC_FOLDER_URL))
    print(f"Найдено лекций: {len(lectures)}")
    for lecture in lectures:
        _process_lecture(app, lecture, students, settings)


def _process_lecture(
    app: Any,
    lecture: RemoteVideo,
    students: list[str],
    settings: dict[str, str | int | float],
) -> None:
    """Скачать и обработать видео или архив одной лекции.

    Args:
        app: скомпилированный граф обработки.
        lecture: метаданные файла лекции с Яндекс Диска.
        students: официальный список студентов.
        settings: настройки интеграций и начисления.
    """
    # Дата в имени единожды определяет колонку для всех видео из одного архива.
    lecture_date = _previous_monday(_date_from_filename(lecture.name))
    work_dir = OUTPUT_DIRECTORY / lecture.name.rsplit(".", maxsplit=1)[0]
    completion_path = work_dir / COMPLETION_FILE
    if _is_completed(completion_path, lecture):
        print(f"Пропущено (уже обработано): {lecture.name}")
        return

    work_items = _cached_work_items(work_dir)
    if work_items:
        print(f"Кэш готов: {lecture.name}; сжатие и Whisper пропущены")
    else:
        video_paths = _existing_compressed_videos(work_dir)
        if video_paths:
            print(f"Шаг 1/5: используются сжатые копии — {lecture.name}")
        else:
            local_lecture = work_dir / "source" / lecture.name
            print(f"Шаг 1/5: скачивание/поиск исходника — {lecture.name}")
            downloaded = download_video(YANDEX_PUBLIC_FOLDER_URL, lecture, local_lecture)
            print(f"{'Скачано' if downloaded else 'Уже скачано'}: {lecture.name}")
            video_paths = _compress_lecture_videos(local_lecture, work_dir)
        work_items = [
            (work_dir if len(video_paths) == 1 else work_dir / video_path.stem, video_path)
            for video_path in video_paths
        ]

    # Все итоги видео объединяются до создания маркера, исключающего повторное начисление.
    updates: list[dict[str, str | int | float]] = []
    for video_dir, video_path in work_items:
        print(f"Шаг 2/5: обработка результатов — {video_path.name}")
        result = app.invoke({
            "work_dir": video_dir,
            "video_path": video_path,
            "students": students,
            "service_account_path": settings["service_account_path"],
            "sheet_id": settings["sheet_id"],
            "worksheet_name": settings["worksheet_name"],
            "lecture_date": lecture_date,
            "points_per_mention": settings["points_per_mention"],
            "score_start_row": settings["score_start_row"],
            "logs": [],
        })
        updates.extend(result.get("sheet_updates", []))
        _write_unconfirmed_students(work_dir, lecture, result)

    _write_completion(completion_path, lecture, updates)
    print(f"Шаг 5/5: маркер завершения сохранён — {lecture.name}")
    print(f"Завершено: {lecture.name}; начислений: {len(updates)}")


def _compress_lecture_videos(local_lecture: Path, work_dir: Path) -> list[Path]:
    """Подготовить сжатые видеокопии лекции для дальнейшей обработки.

    Args:
        local_lecture: скачанный файл лекции.
        work_dir: каталог результата лекции.

    Returns:
        Пути к сжатым видео, которые следует передать конвейеру.
    """
    # Архив раскрывается до сжатия, после чего его исходная копия больше не нужна.
    if local_lecture.suffix.lower() in ARCHIVE_EXTENSIONS:
        source_videos = extract_videos(local_lecture, work_dir / "extracted")
        local_lecture.unlink(missing_ok=True)
    else:
        source_videos = [local_lecture]

    # Каждое видео сжимается отдельно, а оригинал удаляется только после удачного результата.
    compressed_videos: list[Path] = []
    for source_video in source_videos:
        compressed_path = work_dir / "compressed" / f"{source_video.stem}.mkv"
        print(f"Шаг 1/5: сжатие видео — {source_video.name}")
        compressed_video = compress_video(source_video, compressed_path)
        source_video.unlink(missing_ok=True)
        print(f"Шаг 1/5: сжатие завершено — {compressed_video.name}")
        compressed_videos.append(compressed_video)
    return compressed_videos


def _existing_compressed_videos(work_dir: Path) -> list[Path]:
    """Вернуть готовые сжатые видео из предыдущего незавершённого запуска.

    Args:
        work_dir: каталог результата лекции.

    Returns:
        Отсортированные пути к готовым MKV-файлам.
    """
    # Готовые MKV позволяют повторить валидацию, не скачивая и не сжимая лекцию снова.
    compressed_dir = work_dir / "compressed"
    if not compressed_dir.exists():
        return []
    return sorted(compressed_dir.glob("*.mkv"))


def _cached_work_items(work_dir: Path) -> list[tuple[Path, Path]]:
    """Найти полностью готовые кэши, не требующие доступа к видео.

    Args:
        work_dir: каталог результата лекции.

    Returns:
        Пары рабочей папки и виртуального пути видео для запуска графа.
    """
    # Полные кэши позволяют повторить валидацию ФИО без сжатия, Whisper и исходного видео.
    candidate_directories = [work_dir]
    if work_dir.exists():
        candidate_directories.extend(
            directory
            for directory in work_dir.iterdir()
            if directory.is_dir()
            and directory.name not in {
                "audio",
                "audio_chunks",
                "compressed",
                "extracted",
                "source",
                "transcripts",
            }
        )
    cached_items: list[tuple[Path, Path]] = []
    for directory in candidate_directories:
        audio_paths = list((directory / "audio").glob("*.wav"))
        chunk_count = len(list((directory / "audio_chunks").glob("*.wav")))
        transcript_count = len(list((directory / "transcripts").glob("*.txt")))
        if len(audio_paths) != 1 or chunk_count == 0 or chunk_count != transcript_count:
            continue
        virtual_video_path = directory / "cached" / f"{audio_paths[0].stem}.mkv"
        cached_items.append((directory, virtual_video_path))
    return cached_items


def _date_from_filename(filename: str) -> date:
    """Извлечь дату формата ДД.ММ.ГГГГ из имени лекции.

    Args:
        filename: имя исходного видео или архива.

    Returns:
        Дата, указанная в имени файла.

    Raises:
        RuntimeError: если дата отсутствует или некорректна.
    """
    # Явная проверка не позволит начислить баллы в неверную колонку при нестандартном имени.
    match = DATE_PATTERN.search(filename)
    if not match:
        raise RuntimeError(f"В имени лекции нет даты ДД.ММ.ГГГГ: {filename}")
    try:
        return datetime.strptime(match.group(1), "%d.%m.%Y").date()
    except ValueError as error:
        raise RuntimeError(f"Некорректная дата в имени лекции: {filename}") from error


def _previous_monday(lecture_date: date) -> date:
    """Вернуть строго предыдущий понедельник для даты лекции.

    Args:
        lecture_date: дата из имени лекции.

    Returns:
        Дата предшествующего понедельника.
    """
    # Формула возвращает понедельник прошлой недели и для лекции, проведённой в понедельник.
    days_since_monday = lecture_date.weekday() or 7
    return lecture_date - timedelta(days=days_since_monday)


def _write_unconfirmed_students(work_dir: Path, lecture: RemoteVideo, result: dict[str, Any]) -> None:
    """Записать неподтверждённые ФИО в локальный и общий журналы.

    Args:
        work_dir: папка результата лекции.
        lecture: метаданные исходной лекции.
        result: итог работы графа для видео.
    """
    # В журнал попадают uncertain и invalid, чтобы преподаватель мог проверить все спорные находки.
    validation = result.get("validated_students")
    if validation is None:
        return
    entries = [
        {
            "lecture": lecture.name,
            "raw_name": item.raw_name,
            "status": item.status,
            "suggested_name": item.matched_name,
            "confidence": item.confidence,
            "minute": item.minute,
            "comment": item.comment,
        }
        for item in validation.results
        if item.status != "valid"
    ]
    if not entries:
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    _append_json_lines(work_dir / UNCONFIRMED_LOG, entries)
    _append_json_lines(OUTPUT_DIRECTORY / UNCONFIRMED_LOG, entries)


def _append_json_lines(path: Path, entries: list[dict[str, object]]) -> None:
    """Дополнить JSONL-файл записями ручной проверки.

    Args:
        path: путь к файлу журнала.
        entries: записи, которые нужно добавить.
    """
    # JSON Lines легко пополнять между запусками и читать построчно любым инструментом.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        for entry in entries:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _get_settings() -> dict[str, str | int | float]:
    """Прочитать и проверить необходимые переменные окружения.

    Returns:
        Настройки запуска с приведёнными к нужному типу значениями.
    """
    # Единая проверка конфигурации выдаёт понятную ошибку до долгого скачивания и транскрибации.
    required = {
        "SPREADSHEET_ID_2026": os.getenv("SPREADSHEET_ID_2026"),
        "GOOGLE_SERVICE_ACCOUNT_FILE": os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Не заданы переменные окружения: {', '.join(missing)}")
    return {
        "sheet_id": required["SPREADSHEET_ID_2026"] or "",
        "service_account_path": required["GOOGLE_SERVICE_ACCOUNT_FILE"] or "",
        "worksheet_name": os.getenv("WORKSHEET_NAME_2026", "Семестр II"),
        "points_per_mention": float(os.getenv("POINTS_PER_MENTION", "0.5")),
        "score_start_row": int(os.getenv("STUDENT_START_ROW", "4")),
    }


def _is_completed(completion_path: Path, lecture: RemoteVideo) -> bool:
    """Проверить, завершено ли начисление для этой версии лекции.

    Args:
        completion_path: путь к маркеру завершения.
        lecture: метаданные текущей лекции.

    Returns:
        ``True``, если обработана та же версия файла.
    """
    # Размер и путь отличают обновлённый файл от старого маркера с тем же именем.
    if not completion_path.exists():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        completion.get("remote_path") == lecture.path
        and completion.get("size") == lecture.size
        and completion.get("pipeline_version") == PIPELINE_VERSION
    )


def _write_completion(
    completion_path: Path,
    lecture: RemoteVideo,
    updates: list[dict[str, str | int | float]],
) -> None:
    """Сохранить маркер успешного начисления баллов для лекции.

    Args:
        completion_path: путь итогового маркера.
        lecture: обработанная лекция.
        updates: сведения о начислениях из Google Sheets.
    """
    # Метка времени управляет очисткой только полностью завершённых результатов.
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = completion_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "remote_path": lecture.path,
                "size": lecture.size,
                "completed_at": datetime.now().isoformat(),
                "pipeline_version": PIPELINE_VERSION,
                "sheet_updates": updates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(completion_path)


def _cleanup_old_results(output_dir: Path) -> None:
    """Удалить завершённые результаты лекций старше срока хранения.

    Args:
        output_dir: корневая папка всех результатов.
    """
    # Удаляются только дочерние папки с валидным маркером, незавершённые работы сохраняются.
    if not output_dir.exists():
        return
    cutoff = datetime.now() - RETENTION_PERIOD
    for directory in output_dir.iterdir():
        completion_path = directory / COMPLETION_FILE
        if not directory.is_dir() or not completion_path.exists():
            continue
        try:
            completed_at = datetime.fromisoformat(
                json.loads(completion_path.read_text(encoding="utf-8"))["completed_at"]
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if completed_at < cutoff:
            shutil.rmtree(directory)
            print(f"Удалены результаты старше 14 дней: {directory.name}")


if __name__ == "__main__":
    main()
