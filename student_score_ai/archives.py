"""Извлечение видео из архивов лекций."""

from pathlib import Path

import py7zr

from student_score_ai.yandex_disk import VIDEO_EXTENSIONS


ARCHIVE_EXTENSIONS = frozenset({".7z", ".zip"})


def extract_videos(archive_path: Path, destination: Path) -> list[Path]:
    """Распаковать архив и вернуть все видео из него.

    Args:
        archive_path: путь к локальному архиву лекции.
        destination: папка распаковки внутри результата лекции.

    Returns:
        Список извлечённых видео.

    Raises:
        RuntimeError: если архив не поддерживается или не содержит видео.
    """
    # Повторный запуск использует уже распакованные файлы, не тратя время и место на дубликаты.
    videos = _find_videos(destination)
    if videos:
        return videos

    # Поддерживаются форматы из условия; остальные не распаковываются молча.
    destination.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".7z":
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=destination)
    elif suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(path=destination)
    else:
        raise RuntimeError(f"Неподдерживаемый архив: {archive_path.name}")

    videos = _find_videos(destination)
    if not videos:
        raise RuntimeError(f"В архиве {archive_path.name} не найдено видео")
    return videos


def _find_videos(directory: Path) -> list[Path]:
    """Найти видео рекурсивно в каталоге распаковки.

    Args:
        directory: каталог для поиска.

    Returns:
        Отсортированные пути к видео.
    """
    # Рекурсивный поиск покрывает архивы с внутренней структурой папок.
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
