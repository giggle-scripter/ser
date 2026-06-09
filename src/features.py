# features.py — trích xuất đặc trưng âm thanh
# Hàm chính: load_audio(), extract_mfcc(), augment()

import random

import librosa
import numpy as np

try:
    from src.config import (
        DURATION,
        HOP_LENGTH,
        N_FFT,
        N_MELS,
        N_MFCC,
        SAMPLE_RATE,
    )
except ModuleNotFoundError:
    from config import (
        DURATION,
        HOP_LENGTH,
        N_FFT,
        N_MELS,
        N_MFCC,
        SAMPLE_RATE,
    )


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y


def fix_length(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    target = int(sr * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    return y


def extract_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    feature_type: str = "mfcc",
) -> np.ndarray:
    """Return shape: (channels, time_frames) for CNN-1D."""
    y = fix_length(y, sr)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    if feature_type == "mfcc":
        feature = mfcc
    elif feature_type == "mfcc_delta":
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feature = np.concatenate([mfcc, delta, delta2], axis=0)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type!r}")

    feature = (feature - feature.mean()) / (feature.std() + 1e-8)
    return feature.astype(np.float32)


def extract_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return shape: (1, n_mels, time_frames) — dùng cho CNN-2D."""
    y = fix_length(y, sr)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db[np.newaxis, :, :].astype(np.float32)


def augment(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Chỉ gọi trên train set."""
    choice = random.randint(0, 2)
    if choice == 0:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    elif choice == 1:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    else:
        y = y + 0.005 * np.random.randn(len(y))
    return y.astype(np.float32)


def process_audio(
    path: str,
    use_augment: bool = False,
    feature_type: str = "mfcc",
) -> np.ndarray:
    """Hàm Dataset gọi trong __getitem__()."""
    y = load_audio(path)
    if use_augment:
        y = augment(y)
    if feature_type in {"mfcc", "mfcc_delta"}:
        return extract_mfcc(y, feature_type=feature_type)
    if feature_type == "mel":
        return extract_mel(y)
    raise ValueError(f"Unknown feature_type: {feature_type!r}")
