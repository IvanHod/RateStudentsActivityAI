import subprocess
from pathlib import Path

import numpy as np

import noisereduce as nr
from scipy.io import wavfile


def video_to_audio(video_path: Path, path_output: Path) -> Path:
    """Извлечь оптимизированную дорожку для распознавания речи.

    Args:
        video_path: путь к исходному видео.
        path_output: путь итогового WAV-файла.

    Returns:
        Путь к созданному WAV-файлу.

    Raises:
        RuntimeError: если FFmpeg не смог обработать видео.
    """
    # FFmpeg обрабатывает длинные лекции потоково, не занимая память полной WAV-записью.
    path_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path.absolute()),
        "-vn",
        "-ac", "1",
        "-ar", "16000",  # 16000 Hz mono - this is ok for the human voice
        "-af",
        "highpass=f=80",                    # Убираем низкочастотный гул (менее 80 Гц)
        # "loudnorm"
        # "afftdn=nf=-25,"                    # Шумоподавление: мягкий режим (-25...-30 дБ)
        # "acompressor=threshold=-20dB:ratio=3:attack=10:release=100,"  # Компрессия для выравнивания громкости
        # "loudnorm=I=-16:TP=-1.5:LRA=7,"     # Нормализация под стандарт Whisper (-16 LUFS)
        # "lowpass=f=7500",                   # Мягкий срез выше 7.5 кГц (убирает шипение, но сохраняет речь)
        "-c:a", "pcm_s16le",
        str(path_output.absolute())
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    # Ошибка должна прервать конвейер до Whisper, иначе будет обработан повреждённый файл.
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg не обработал {video_path.name}: {result.stderr}")

    return path_output


def compress_video(video_path: Path, path_output: Path) -> Path:
    """Сжать видео в совместимый MP4-файл.

    Args:
        video_path: путь к исходному видео.
        path_output: путь к сжатому видео в контейнере MP4.

    Returns:
        Путь к сжатой видеокопии.

    Raises:
        RuntimeError: если FFmpeg не смог сжать видео.
    """
    # Готовая MP4-копия позволяет безопасно продолжить прерванный запуск без повторного сжатия.
    if path_output.exists():
        return path_output

    path_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path_output.with_name(f"{path_output.stem}.partial{path_output.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path.absolute()),
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(temporary_path.absolute()),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Исходник удаляется только после успешного и атомарного создания сжатой копии.
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg не сжал {video_path.name}: {result.stderr}")
    temporary_path.replace(path_output)
    return path_output


def reduce_noize(path_in: Path, path_out: Path) -> None:
    """Применить шумоподавление к WAV-файлу.

    Args:
        path_in: путь к исходному WAV-файлу.
        path_out: путь для сохранения результата.
    """
    # Функция сохранена для возможного отдельного этапа улучшения аудио.
    rate, data = wavfile.read(path_in)

    data_f = data.astype(np.float32)

    reduced = nr.reduce_noise(y=data_f, sr=rate, stationary=False, prop_decrease=0.6)

    reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
    wavfile.write(path_out, rate, reduced)
