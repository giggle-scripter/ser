# dataset.py — PyTorch Dataset cho RAVDESS
# Phụ trách: Linh
# Hàm chính: parse_filename(), RAVDESSDataset, build_dataset()

from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

try:
    from src.features import process_audio
    from src.config import DATA_DIR
except ModuleNotFoundError:
    from features import process_audio
    from config import DATA_DIR

# ── Mapping mã cảm xúc theo chuẩn RAVDESS ──────────────────────────────────
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def parse_filename(path: str | Path) -> str:
    stem = Path(path).stem          # bỏ đuôi .wav
    parts = stem.split("-")         # tách theo dấu '-'
    emotion_code = parts[2]         # vị trí thứ 3 = mã cảm xúc
    return EMOTION_MAP.get(emotion_code, "unknown")

class RAVDESSDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        use_augment: bool = False,
        feature_type: str = "mfcc",
    ):
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
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    # test_ratio tự động = 1 - train - val = 0.15
    random_state: int  = 42,
    feature_type: str  = "mfcc",
) -> Tuple[RAVDESSDataset, RAVDESSDataset, RAVDESSDataset, LabelEncoder]:
    
    data_dir = Path(data_dir)

    # ── Thu thập tất cả file và nhãn ──
    all_paths, all_labels_str = [], []
    for wav_file in sorted(data_dir.glob("*/*.wav")):
        label_str = parse_filename(wav_file)
        if label_str == "unknown":
            continue
        all_paths.append(str(wav_file))
        all_labels_str.append(label_str)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy file .wav nào trong {data_dir}. "
            "Hãy kiểm tra đường dẫn DATA_DIR trong config.py."
        )

    # ── Encode nhãn string → số nguyên ──
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels_str)   # 'angry'→0, 'calm'→1, ...

    # ── Stratified split: train | temp(val+test) ──
    temp_ratio = 1.0 - train_ratio                  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=all_labels,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    train_ds = RAVDESSDataset(X_train, y_train.tolist(),
                               use_augment=True,  feature_type=feature_type)
    val_ds   = RAVDESSDataset(X_val,   y_val.tolist(),
                               use_augment=False, feature_type=feature_type)
    test_ds  = RAVDESSDataset(X_test,  y_test.tolist(),
                               use_augment=False, feature_type=feature_type)

    print(f"Dataset đã sẵn sàng:")
    print(f"  Train : {len(train_ds)} mẫu")
    print(f"  Val   : {len(val_ds)} mẫu")
    print(f"  Test  : {len(test_ds)} mẫu")
    print(f"  Classes: {list(le.classes_)}")

    return train_ds, val_ds, test_ds, le