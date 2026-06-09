# inference.py — dự đoán từ audio buffer
# Hàm chính: predict()
# Dùng bởi streamlit_app.py — áp softmax ở đây (KHÔNG trong model.forward())
from __future__ import annotations

from pathlib import Path

import torch

try:
    from src.config import EMOTION_LIST, N_FEATURES
    from src.features import process_audio
    from src.model import CNN1D
except ModuleNotFoundError:
    from config import EMOTION_LIST, N_FEATURES
    from features import process_audio
    from model import CNN1D


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    saved_n_features = config.get("n_features", N_FEATURES)

    print(f"checkpoint path: {checkpoint_path}")
    print(f"saved feature_type: {config.get('feature_type')}")
    print(f"saved n_features: {saved_n_features}")
    if checkpoint.get("metrics") is not None:
        print(f"saved metrics: {checkpoint['metrics']}")

    model = CNN1D(in_channels=saved_n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict(
    audio_path: str | Path,
    model: CNN1D,
    checkpoint: dict,
    device: torch.device,
) -> dict:
    config = checkpoint.get("config", {})
    feature_type = config.get("feature_type")
    if feature_type not in {"mfcc", "mfcc_delta"}:
        raise ValueError(
            "Checkpoint config must contain feature_type='mfcc' or 'mfcc_delta'."
        )

    feature = process_audio(
        str(audio_path),
        use_augment=False,
        feature_type=feature_type,
    )
    x = torch.from_numpy(feature).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(probs.argmax().item())
    class_names = config.get("class_names", EMOTION_LIST)
    return {
        "label": class_names[pred_idx],
        "class_index": pred_idx,
        "probability": float(probs[pred_idx].item()),
        "probabilities": probs.cpu().tolist(),
    }
