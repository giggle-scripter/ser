from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

try:
    from src.config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        DROPOUT,
        FEATURE_TYPE,
        FIGURE_DIR,
        LABEL_ENCODING,
        MODEL_NAME,
        N_FEATURES,
        NUM_CLASSES,
        RUN_NAME,
    )
    from src.dataset import build_dataset
    from src.model import create_model
    from src.utils import append_result_csv, get_device, set_seed
except ModuleNotFoundError:
    from config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        DROPOUT,
        FEATURE_TYPE,
        FIGURE_DIR,
        LABEL_ENCODING,
        MODEL_NAME,
        N_FEATURES,
        NUM_CLASSES,
        RUN_NAME,
    )
    from dataset import build_dataset
    from model import create_model
    from utils import append_result_csv, get_device, set_seed


RUN_NOTES = {
    "cnn1d_mfcc_baseline_organic": "MFCC only baseline",
    "cnn1d_mfcc_delta_fix1_organic": "MFCC + delta + delta-delta ablation",
    "cnn1d_mfcc_delta_ls005_do025_adam": (
        "MFCC + delta + label smoothing + dropout + Adam"
    ),
    "cnn1d_mfcc_delta_ls005_adamw": (
        "MFCC + delta + label smoothing + dropout + AdamW"
    ),
}


def safe_filename(text: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in text
    ).strip("_")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.long().to(device)

            logits = model(x)

            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)

            total_samples += x.size(0)
            all_targets.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = {
        "acc": accuracy_score(all_targets, all_preds),
        "f1_macro": f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            all_targets, all_preds, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_targets, all_preds),
        "classification_report": classification_report(
            all_targets,
            all_preds,
            digits=4,
            zero_division=0,
        ),
        "y_true": all_targets,
        "y_pred": all_preds,
    }

    if criterion is not None:
        metrics["loss"] = total_loss / total_samples

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: bool = True,
    title_note: str | None = None,
) -> None:
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    title = "Confusion Matrix" + (" (Normalized by row)" if normalize else "")
    if title_note:
        title = f"{title}\n{title_note}"

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_metrics = checkpoint.get("metrics")
    saved_feature_type = checkpoint_config.get("feature_type", FEATURE_TYPE)
    saved_n_features = checkpoint_config.get("n_features", N_FEATURES)
    saved_model_name = checkpoint_config.get("model", MODEL_NAME)
    saved_label_encoding = checkpoint_config.get("label_encoding", LABEL_ENCODING)

    print(f"checkpoint path: {checkpoint_path}")
    print(f"saved model: {saved_model_name}")
    print(f"saved feature_type: {saved_feature_type}")
    print(f"saved label_encoding: {saved_label_encoding}")
    print(f"saved n_features: {saved_n_features}")
    if checkpoint_metrics is not None:
        print(f"saved metrics: {checkpoint_metrics}")

    class_names = checkpoint.get("class_names") or checkpoint_config.get("class_names")
    if class_names is None:
        _, _, _, class_names = build_dataset(
            feature_type=saved_feature_type,
            label_encoding=saved_label_encoding,
        )

    model = create_model(
        model_name=saved_model_name,
        in_channels=saved_n_features,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, checkpoint_config


def main() -> None:
    set_seed()
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / RUN_NAME / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại: {checkpoint_path}\n"
            "Hãy train model trước bằng `uv run python -m src.train`."
        )

    model, class_names, checkpoint_config = load_checkpoint_model(checkpoint_path, device)
    run_name = checkpoint_config.get("run_name", RUN_NAME)
    model_name = checkpoint_config.get("model", MODEL_NAME)
    feature_type = checkpoint_config.get("feature_type", FEATURE_TYPE)
    label_encoding = checkpoint_config.get("label_encoding", LABEL_ENCODING)

    _, _, test_ds, _ = build_dataset(
        feature_type=feature_type,
        label_encoding=label_encoding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, test_loader, device, criterion=criterion)

    print(f"Using device: {device}")
    print("\nTest metrics:")
    print(f"loss         = {metrics['loss']:.4f}")
    print(f"acc          = {metrics['acc']:.4f}")
    print(f"f1_macro     = {metrics['f1_macro']:.4f}")
    print(f"f1_weighted  = {metrics['f1_weighted']:.4f}")

    print("\nClassification report:")
    print(metrics["classification_report"])

    figure_name = safe_filename(
        f"confusion_matrix_{run_name}_{model_name}_{feature_type}"
    ) + ".png"
    save_path = FIGURE_DIR / figure_name
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        save_path=save_path,
        normalize=True,
        title_note=f"run={run_name} | model={model_name} | feature={feature_type}",
    )
    print(f"\nSaved confusion matrix to: {save_path}")
    append_result_csv(
        {
            "run_name": run_name,
            "model": model_name,
            "feature_type": feature_type,
            "test_loss": metrics["loss"],
            "test_acc": metrics["acc"],
            "test_f1_macro": metrics["f1_macro"],
            "test_f1_weight": metrics["f1_weighted"],
            "notes": RUN_NOTES.get(run_name, ""),
        }
    )


if __name__ == "__main__":
    main()
