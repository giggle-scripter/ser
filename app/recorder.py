from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


def record_audio(duration: float = 3.0, sr: int = 22050) -> np.ndarray:
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.squeeze()

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def save_to_wav(audio: np.ndarray, path: str | Path, sr: int = 22050) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())

    return path
