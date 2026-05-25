# utils.py — các hàm tiện ích dùng chung
# Hàm chính: set_seed(), logging helpers
import random, csv
from datetime import datetime
import numpy as np
import torch
from src.config import RANDOM_SEED, EXPERIMENT_DIR

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
    fieldnames = ["run_id","date","model","lr","dropout","batch",
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