# utils.py — các hàm tiện ích dùng chung
# Hàm chính: set_seed(), logging helpers
import random, csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

try:
    from src.config import RANDOM_SEED, EXPERIMENT_DIR
except ModuleNotFoundError:
    from config import RANDOM_SEED, EXPERIMENT_DIR

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_experiment(run_id: str, config: dict, metrics: dict, notes: str = "") -> None:
    csv_path = EXPERIMENT_DIR / "results.csv"
    fieldnames = ["run_id","date","model","feature_type","lr","dropout","batch",
                  "augment","epochs_trained","train_acc","val_acc",
                  "test_acc","f1_macro","f1_weighted","notes"]
    row = {"run_id": run_id,
           "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           **config, **metrics, "notes": notes}
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def append_result_csv(
    result_dict: dict,
    csv_path: str = "experiments/results.csv",
) -> None:
    fieldnames = [
        "run_name",
        "model",
        "feature_type",
        "test_loss",
        "test_acc",
        "test_f1_macro",
        "test_f1_weight",
        "notes",
    ]
    path = Path(csv_path)
    if not path.is_absolute():
        path = EXPERIMENT_DIR.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {key: result_dict.get(key, "") for key in fieldnames}
    existing_header = None
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None)

    write_header = existing_header != fieldnames
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
