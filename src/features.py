# features.py — trích xuất đặc trưng âm thanh
# Phụ trách: Phương
# Hàm chính: load_audio(), extract_mfcc(), augment()
import random
import librosa
import numpy as np
from src.config import SAMPLE_RATE, DURATION, N_MFCC, N_FFT, HOP_LENGTH

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

def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return shape: (N_MFCC, time_frames)"""
    y = fix_length(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc.astype(np.float32)

def extract_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return shape: (1, 128, time_frames) — dùng cho CNN-2D"""
    y = fix_length(y, sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db[np.newaxis, :, :].astype(np.float32)

def augment(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """CHỈ gọi trên train set."""
    choice = random.randint(0, 2)
    if choice == 0:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    elif choice == 1:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    else:
        y = y + 0.005 * np.random.randn(len(y))
    return y.astype(np.float32)

def process_audio(path: str, use_augment: bool = False,
                  feature_type: str = "mfcc") -> np.ndarray:
    """Hàm Linh gọi trong Dataset.__getitem__()."""
    y = load_audio(path)
    if use_augment:
        y = augment(y)
    return extract_mfcc(y) if feature_type == "mfcc" else extract_mel(y)