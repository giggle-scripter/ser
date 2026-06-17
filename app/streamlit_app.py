from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.recorder import record_audio, save_to_wav
from src.config import CHECKPOINT_DIR, SAMPLE_RATE
from src.inference import load_model, predict
from src.utils import get_device


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="SER",
    layout="centered",
)


FINAL_RUN_NAME = "cnn1d_mfcc_delta_ls005_adamw"


APP_STYLES = """
<style>
    :root {
        --ser-bg: #fff8fb;
        --ser-card: #ffffff;
        --ser-border: #f0ccd8;
        --ser-primary: #b83268;
        --ser-primary-dark: #8e2853;
        --ser-ink: #33242b;
        --ser-muted: #6f5a64;
    }

    .stApp {
        background: linear-gradient(180deg, #fff8fb 0%, #ffffff 48%, #fff4f8 100%);
        color: var(--ser-ink);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--ser-border);
    }

    [data-testid="stSidebar"] h1 {
        color: var(--ser-primary-dark);
        font-size: 1.35rem;
    }

    .ser-title {
        padding: 1rem 0 0.5rem;
    }

    .ser-title h1 {
        color: var(--ser-primary-dark);
        font-size: 2.25rem;
        line-height: 1.12;
        margin: 0;
    }

    .ser-title p {
        color: var(--ser-muted);
        font-size: 1rem;
        margin: 0.55rem 0 0;
    }

    .ser-card {
        background: var(--ser-card);
        border: 1px solid var(--ser-border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0;
        box-shadow: 0 10px 24px rgba(184, 50, 104, 0.06);
    }

    .ser-card-title {
        color: var(--ser-primary-dark);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .ser-result {
        background: #fff;
        border: 1px solid var(--ser-border);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin: 1rem 0 0.75rem;
        box-shadow: 0 10px 24px rgba(184, 50, 104, 0.08);
    }

    .ser-result-label {
        color: var(--ser-primary-dark);
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
    }

    .ser-result-caption {
        color: var(--ser-muted);
        margin: 0.15rem 0 0;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: var(--ser-primary);
        border: 1px solid var(--ser-primary);
        color: white;
        border-radius: 8px;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: var(--ser-primary-dark);
        border-color: var(--ser-primary-dark);
        color: white;
    }

    [data-testid="stMetricValue"] {
        color: var(--ser-primary-dark);
    }

    [data-testid="stMetricLabel"] {
        color: var(--ser-muted);
    }
</style>
"""


def final_checkpoint_path() -> Path:
    return CHECKPOINT_DIR / FINAL_RUN_NAME / "best_model.pt"


def format_checkpoint(path: Path) -> str:
    if path.parent == CHECKPOINT_DIR:
        return path.name
    return f"{path.parent.name}/{path.name}"


@st.cache_resource
def cached_model(checkpoint_path: str):
    device = get_device()
    model, checkpoint = load_model(Path(checkpoint_path), device)
    return model, checkpoint, device


def load_saved_metrics(checkpoint_path: Path, checkpoint: dict) -> dict:
    metrics = checkpoint.get("metrics", {}) or {}
    if any(key.startswith("test_") for key in metrics):
        return metrics

    metrics_path = checkpoint_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    return metrics


def run_prediction(audio_path: Path, checkpoint_path: Path) -> None:
    model, checkpoint, device = cached_model(str(checkpoint_path))
    config = checkpoint.get("config", {})
    result = predict(audio_path, model, checkpoint, device)

    st.markdown(
        f"""
        <div class="ser-result">
            <p class="ser-result-label">{result["label"].title()}</p>
            <p class="ser-result-caption">
                Confidence: {result["probability"]:.2%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_names = config.get("class_names", [])
    probs = result["probabilities"]
    if class_names and len(class_names) == len(probs):
        df = pd.DataFrame(
            {
                "emotion": class_names,
                "probability": probs,
            }
        ).sort_values("probability", ascending=False)
        st.bar_chart(df.set_index("emotion"))


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def render_metric_summary(metrics: dict) -> None:
    metric_cols = [
        ("test_acc", "Accuracy"),
        ("test_f1_macro", "Macro F1"),
        ("test_f1_weight", "Weighted F1"),
        ("test_loss", "Loss"),
    ]

    cols = st.columns(len(metric_cols))
    for col, (key, label) in zip(cols, metric_cols):
        value = metrics.get(key)
        if isinstance(value, float):
            col.metric(label, f"{value:.4f}")
        else:
            col.metric(label, "N/A")


def main() -> None:
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ser-title">
            <h1>Speech Emotion Recognition</h1>
            <p>Upload or record a short voice clip to predict the speaker emotion.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_path = final_checkpoint_path()
    if not checkpoint_path.exists():
        st.error(
            "Final checkpoint not found. Train or restore "
            f"`{FINAL_RUN_NAME}` first."
        )
        st.stop()

    model, checkpoint, _ = cached_model(str(checkpoint_path))
    config = checkpoint.get("config", {})
    metrics = load_saved_metrics(checkpoint_path, checkpoint)

    with st.sidebar:
        st.title("Model Summary")
        st.caption("Using the selected final checkpoint")
        st.write(format_checkpoint(checkpoint_path))

        st.divider()
        st.write(f"**Architecture:** {model.__class__.__name__}")
        st.write(f"**Feature:** {config.get('feature_type', 'unknown')}")
        st.write(f"**Channels:** {config.get('n_features', 'unknown')}")

        if metrics:
            st.divider()
            st.caption("Test metrics")
            st.metric("Accuracy", f"{metrics.get('test_acc', 0.0):.4f}")
            st.metric("Macro F1", f"{metrics.get('test_f1_macro', 0.0):.4f}")

        with st.expander("Config", expanded=False):
            st.json(config)

    if metrics:
        st.markdown(
            """
            <div class="ser-card">
                <div class="ser-card-title">Final model</div>
                CNN1D with MFCC + delta + delta-delta features, label smoothing,
                dropout tuning, and AdamW.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_metric_summary(metrics)

    upload_tab, record_tab = st.tabs(["Upload Audio", "Record Audio"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "ogg"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Choose a short voice clip to run emotion prediction.")
        else:
            audio_path = save_uploaded_file(uploaded_file)
            st.audio(uploaded_file)
            run_prediction(audio_path, checkpoint_path)

    with record_tab:
        duration = st.slider(
            "Duration",
            min_value=1.0,
            max_value=8.0,
            value=3.0,
            step=0.5,
        )
        st.caption(
            "Recording uses the local machine microphone. "
            "For cloud deployment, use Upload Audio instead."
        )

        if st.button("Record", type="primary"):
            with st.spinner("Recording..."):
                audio = record_audio(duration=duration, sr=SAMPLE_RATE)
                audio_path = Path(tempfile.gettempdir()) / "ser_recording.wav"
                save_to_wav(audio, audio_path, sr=SAMPLE_RATE)

            st.audio(str(audio_path))
            run_prediction(audio_path, checkpoint_path)


if __name__ == "__main__":
    main()
