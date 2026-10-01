"""Работа с публичной папкой Яндекс Диска."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx


YANDEX_PUBLIC_RESOURCES_URL = "https://cloud-api.yandex.net/v1/disk/public/resources"
YANDEX_PUBLIC_DOWNLOAD_URL = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download"
)
VIDEO_EXTENSIONS = frozenset(
    {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
)
LECTURE_EXTENSIONS = VIDEO_EXTENSIONS | frozenset({".7z", ".zip"})


@dataclass(frozen=True)
class RemoteVideo:
    """Описание видео из публичной папки."""

    path: str
    name: str
    size: int


def iter_public_videos(public_url: str) -> Iterator[RemoteVideo]:
    """Вернуть видео и архивы лекций из публичной папки и её подпапок.

    Args:
        public_url: публичная ссылка на папку Яндекс Диска.

    Yields:
        Метаданные каждого найденного видео.

    Raises:
        httpx.HTTPStatusError: если публичная ссылка недоступна.
    """
    # Рекурсивный обход нужен, чтобы подпапки преподавателя не оставались без обработки.
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        yield from _iter_directory_videos(client, public_url, path="")


def download_video(public_url: str, video: RemoteVideo, destination: Path) -> bool:
    """Скачать видео, если корректной локальной копии ещё нет.

    Args:
        public_url: публичная ссылка на папку Яндекс Диска.
        video: метаданные скачиваемого файла.
        destination: путь локальной копии видео.

    Returns:
        ``True``, если файл был скачан в этом запуске, иначе ``False``.

    Raises:
        httpx.HTTPStatusError: если Яндекс Диск не отдал ссылку или файл.
    """
    # Проверка размера делает повторные запуски быстрыми и не считает оборванную загрузку готовой.
    if destination.exists() and destination.stat().st_size == video.size:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")

    # Временный файл сохраняет предыдущую рабочую копию, пока новая загрузка не завершится.
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(
            YANDEX_PUBLIC_DOWNLOAD_URL,
            params={"public_key": public_url, "path": video.path},
        )
        response.raise_for_status()
        download_url = response.json()["href"]

        with client.stream("GET", download_url) as download_response:
            download_response.raise_for_status()
            with temporary_path.open("wb") as output_file:
                for chunk in download_response.iter_bytes(chunk_size=1024 * 1024):
                    output_file.write(chunk)

    if temporary_path.stat().st_size != video.size:
        raise RuntimeError(f"Неполная загрузка {video.name}: размер не совпадает")

    temporary_path.replace(destination)
    return True


def _iter_directory_videos(
    client: httpx.Client,
    public_url: str,
    path: str,
) -> Iterator[RemoteVideo]:
    """Обойти одну папку публичного ресурса.

    Args:
        client: HTTP-клиент для запросов к API.
        public_url: публичная ссылка на корневой ресурс.
        path: путь папки относительно корня ресурса.

    Yields:
        Метаданные видео в текущей папке и её потомках.
    """
    # Пагинация нужна, поскольку публичная папка может содержать больше 100 файлов.
    offset = 0
    while True:
        response = client.get(
            YANDEX_PUBLIC_RESOURCES_URL,
            params={"public_key": public_url, "path": path, "limit": 100, "offset": offset},
        )
        response.raise_for_status()
        embedded = response.json().get("_embedded", {})
        items: list[dict[str, Any]] = embedded.get("items", [])

        for item in items:
            item_path = str(item["path"])
            if item.get("type") == "dir":
                yield from _iter_directory_videos(client, public_url, item_path)
            elif Path(str(item["name"])).suffix.lower() in LECTURE_EXTENSIONS:
                yield RemoteVideo(
                    path=item_path,
                    name=str(item["name"]),
                    size=int(item.get("size", 0)),
                )

        offset += len(items)
        if offset >= int(embedded.get("total", 0)) or not items:
            return
