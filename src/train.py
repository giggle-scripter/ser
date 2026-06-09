# train.py — training loop
# Hàm chính: train()
# Bao gồm: Adam optimizer, Weighted CrossEntropyLoss, Early Stopping, checkpoint save
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

try:
    from src.config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        DROPOUT,
        EARLY_STOP_PATIENCE,
        EPOCHS,
        FEATURE_TYPE,
        LEARNING_RATE,
        LR_SCHEDULER_FACTOR,
        LR_SCHEDULER_PATIENCE,
        N_FEATURES,
        N_MFCC,
        NUM_CLASSES,
        RUN_NAME,
        WEIGHT_DECAY,
    )
    from src.dataset import build_dataset
    from src.model import CNN1D
    from src.utils import append_result_csv, get_device, set_seed
except ModuleNotFoundError:
    from config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        DROPOUT,
        EARLY_STOP_PATIENCE,
        EPOCHS,
        FEATURE_TYPE,
        LEARNING_RATE,
        LR_SCHEDULER_FACTOR,
        LR_SCHEDULER_PATIENCE,
        N_FEATURES,
        N_MFCC,
        NUM_CLASSES,
        RUN_NAME,
        WEIGHT_DECAY,
    )
    from dataset import build_dataset
    from model import CNN1D
    from utils import append_result_csv, get_device, set_seed


RUN_NOTES = {
    "cnn1d_mfcc_baseline": "MFCC only baseline",
    "cnn1d_mfcc_delta_fix1": "MFCC + delta + delta-delta final model",
}


def build_config_dict() -> dict:
    return {
        "run_name": RUN_NAME,
        "model": "CNN1D",
        "feature_type": FEATURE_TYPE,
        "n_mfcc": N_MFCC,
        "n_features": N_FEATURES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config_dict: dict,
    run_name: str,
    is_best: bool = False,
) -> Path:
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config_dict,
    }

    last_path = run_dir / "last_model.pt"
    torch.save(checkpoint, last_path)

    if is_best:
        torch.save(checkpoint, run_dir / "best_model.pt")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return run_dir


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_preds = []

    for x, y in loader:
        x = x.float().to(device)
        y = y.long().to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

        all_targets.extend(y.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def train() -> None:
    set_seed()
    device = get_device()
    config_dict = build_config_dict()
    checkpoint_dir = CHECKPOINT_DIR / RUN_NAME

    print(f"Using device: {device}")
    print(f"RUN_NAME: {RUN_NAME}")
    print(f"FEATURE_TYPE: {FEATURE_TYPE}")
    print(f"N_FEATURES: {N_FEATURES}")
    print("model class: CNN1D")
    print(f"checkpoint directory: {checkpoint_dir}")

    train_ds, val_ds, test_ds, class_names = build_dataset()
    config_dict["class_names"] = class_names

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CNN1D(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

    class_weights = compute_class_weights(train_ds.labels, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR,
    )

    best_score = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_ckpt_path = checkpoint_dir / "best_model.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        epoch_metrics = {
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        current_score = epoch_metrics.get("val_f1_macro", epoch_metrics["val_acc"])
        is_best = current_score > best_score

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_metrics,
            config_dict=config_dict,
            run_name=RUN_NAME,
            is_best=is_best,
        )

        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"Saved best checkpoint to: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, device)

    print("\nFinal test metrics:")
    print(f"test_loss      = {test_metrics['loss']:.4f}")
    print(f"test_acc       = {test_metrics['acc']:.4f}")
    print(f"test_f1_macro  = {test_metrics['f1_macro']:.4f}")
    print(f"test_f1_weight = {test_metrics['f1_weighted']:.4f}")

    final_metrics = {
        "best_epoch": best_epoch,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weight": test_metrics["f1_weighted"],
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    append_result_csv(
        {
            "run_name": RUN_NAME,
            "model": "CNN1D",
            "feature_type": FEATURE_TYPE,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_weight": test_metrics["f1_weighted"],
            "notes": RUN_NOTES.get(RUN_NAME, ""),
        }
    )


if __name__ == "__main__":
    train()
