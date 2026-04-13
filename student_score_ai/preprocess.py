import os
import subprocess
from pathlib import Path


def video_to_audio(video_path: Path, output_dir: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16000 Hz mono - this is ok for the human voice
        "-ac", "1",
        output_dir
    ]
    subprocess.run(cmd, check=True)
    return output_dir