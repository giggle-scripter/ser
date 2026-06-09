# dataset.py — PyTorch Dataset cho RAVDESS
# Hàm chính: parse_filename(), RAVDESSDataset, build_dataset()

from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from src.features import process_audio
    from src.config import (
        DATA_DIR,
        EMOTIONS,
        EMOTION_LIST,
        FEATURE_TYPE,
        RANDOM_SEED,
        TEST_RATIO,
        TRAIN_RATIO,
        VAL_RATIO,
    )
except ModuleNotFoundError:
    from features import process_audio
    from config import (
        DATA_DIR,
        EMOTIONS,
        EMOTION_LIST,
        FEATURE_TYPE,
        RANDOM_SEED,
        TEST_RATIO,
        TRAIN_RATIO,
        VAL_RATIO,
    )

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(EMOTION_LIST)}
VALID_FEATURE_TYPES = {"mfcc", "mfcc_delta"}

def parse_filename(path: str | Path) -> str:
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return "unknown"
    emotion_code = parts[2]
    return EMOTIONS.get(emotion_code, "unknown")

class RAVDESSDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        use_augment: bool = False,
        feature_type: str = FEATURE_TYPE,
    ):
        if feature_type not in VALID_FEATURE_TYPES:
            raise ValueError(
                f"Invalid feature_type: {feature_type!r}. "
                f"Expected one of {sorted(VALID_FEATURE_TYPES)}."
            )
        self.file_paths   = file_paths
        self.labels       = labels
        self.use_augment  = use_augment
        self.feature_type = feature_type

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path    = self.file_paths[idx]
        label   = self.labels[idx]
        feature = process_audio(
            path,
            use_augment=self.use_augment,
            feature_type=self.feature_type,
        )
        return feature, label

def build_dataset(
    data_dir: str | Path = DATA_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
    feature_type: str = FEATURE_TYPE,
) -> Tuple[RAVDESSDataset, RAVDESSDataset, RAVDESSDataset, List[str]]:
    if feature_type not in VALID_FEATURE_TYPES:
        raise ValueError(
            f"Invalid feature_type: {feature_type!r}. "
            f"Expected one of {sorted(VALID_FEATURE_TYPES)}."
        )

    data_dir = Path(data_dir)
    ratio_sum = train_ratio + val_ratio + test_ratio

    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "Tổng train_ratio + val_ratio + test_ratio phải bằng 1.0. "
            f"Hiện tại là {ratio_sum:.4f}."
        )

    all_paths, all_labels = [], []
    for wav_file in sorted(data_dir.glob("*/*.wav")):
        label_str = parse_filename(wav_file)
        label_idx = LABEL_TO_INDEX.get(label_str)
        if label_idx is None:
            continue
        all_paths.append(str(wav_file))
        all_labels.append(label_idx)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy file .wav nào trong {data_dir}. "
            "Hãy kiểm tra đường dẫn DATA_DIR trong config.py."
        )

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=all_labels,
    )

    test_share_in_temp = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_share_in_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    train_ds = RAVDESSDataset(list(X_train), list(y_train),
                               use_augment=True,  feature_type=feature_type)
    val_ds   = RAVDESSDataset(list(X_val),   list(y_val),
                               use_augment=False, feature_type=feature_type)
    test_ds  = RAVDESSDataset(list(X_test),  list(y_test),
                               use_augment=False, feature_type=feature_type)

    return train_ds, val_ds, test_ds, EMOTION_LIST.copy()
