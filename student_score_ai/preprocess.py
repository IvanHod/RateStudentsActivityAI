import subprocess
from pathlib import Path

import numpy as np

import noisereduce as nr
from scipy.io import wavfile


def video_to_audio(video_path: Path, path_output: Path) -> Path:
    path_output.parent.mkdir(parents=True, exist_ok=True)

    path_output_tmp = path_output.parent / f'{path_output.stem}_draft{path_output.suffix}'
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
        str(path_output_tmp.absolute())
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        print("❌ Ошибка FFmpeg:")
        print(result.stderr)  # Здесь будет точная причина

    reduce_noize(path_output_tmp, path_output)
    return path_output


def reduce_noize(path_in: Path, path_out: Path):
    rate, data = wavfile.read(path_in)

    data_f = data.astype(np.float32)

    reduced = nr.reduce_noise(y=data_f, sr=rate, stationary=False, prop_decrease=0.6)

    reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
    wavfile.write(path_out, rate, reduced)