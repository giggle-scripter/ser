from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

try:
    from src.features import process_audio
    from src.config import (
        DATA_DIR,
        EMOTIONS,
        EMOTION_LIST,
        FEATURE_TYPE,
        LABEL_ENCODING,
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
        LABEL_ENCODING,
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
    label_encoding: str = LABEL_ENCODING,
) -> Tuple[RAVDESSDataset, RAVDESSDataset, RAVDESSDataset, List[str]]:
    if feature_type not in VALID_FEATURE_TYPES:
        raise ValueError(
            f"Invalid feature_type: {feature_type!r}. "
            f"Expected one of {sorted(VALID_FEATURE_TYPES)}."
        )
    if label_encoding not in {"ravdess", "legacy"}:
        raise ValueError(
            f"Invalid label_encoding: {label_encoding!r}. "
            "Expected 'ravdess' or 'legacy'."
        )

    data_dir = Path(data_dir)
    ratio_sum = train_ratio + val_ratio + test_ratio

    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "Tổng train_ratio + val_ratio + test_ratio phải bằng 1.0. "
            f"Hiện tại là {ratio_sum:.4f}."
        )

    all_paths, all_label_names = [], []
    for wav_file in sorted(data_dir.glob("*/*.wav")):
        label_str = parse_filename(wav_file)
        if label_str not in LABEL_TO_INDEX:
            continue
        all_paths.append(str(wav_file))
        all_label_names.append(label_str)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy file .wav nào trong {data_dir}. "
            "Hãy kiểm tra đường dẫn DATA_DIR trong config.py."
        )

    if label_encoding == "legacy":
        encoder = LabelEncoder()
        all_labels = encoder.fit_transform(all_label_names).tolist()
        class_names = [str(label) for label in encoder.classes_]
    else:
        all_labels = [LABEL_TO_INDEX[label] for label in all_label_names]
        class_names = EMOTION_LIST.copy()

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

    return train_ds, val_ds, test_ds, class_names
