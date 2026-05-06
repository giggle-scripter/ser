# Speech Emotion Recognition (SER)

Bài tập lớn môn **Nhập môn Trí tuệ Nhân tạo** — Phân loại cảm xúc từ giọng nói bằng CNN.

**Nhóm:** Phạm Thị Bích Phương · Nguyễn Hải Yến Nhi · Nguyễn Thị Hải Linh

---

## Tổng quan

| | |
|---|---|
| **Dataset** | RAVDESS — 1.440 file `.wav`, 8 nhãn cảm xúc |
| **Model** | CNN-1D trên đặc trưng MFCC (40 hệ số) |
| **Mục tiêu** | Accuracy ≥ 65%, F1 macro ≥ 0.60 |
| **Không dùng** | Pretrained weights |

**8 nhãn cảm xúc:** `neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

---

## Yêu cầu

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — package manager

---

## Cài đặt

### 1. Clone repo

```bash
git clone <repo-url>
cd ser
```

### 2. Cài dependencies

```bash
uv sync
```

> `uv sync` tự tạo `.venv` và cài đúng phiên bản Python từ `.python-version`.

### 3. Thêm thư viện (nếu cần)

```bash
uv add torch torchaudio
uv add librosa sounddevice streamlit matplotlib
uv add scikit-learn
```

---

## Download Dataset

1. Tải **RAVDESS** tại: https://zenodo.org/record/1188976
2. Giải nén vào thư mục `data/ravdess/` sao cho cấu trúc là:

```
data/
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

> `data/` không được commit lên GitHub.

---

## Chạy ứng dụng

### Demo Streamlit (upload file hoặc ghi âm)

```bash
uv run streamlit run app/streamlit_app.py
```

### Training

```bash
uv run python main.py
```

### Training trực tiếp từ module

```bash
uv run python -m src.train
```

### Evaluate model

```bash
uv run python -m src.evaluate
```

---

## Cấu trúc thư mục

```
ser/
├── data/ravdess/               # Dataset RAVDESS (không commit)
├── src/
│   ├── config.py               # Tất cả hyperparameters — chỉnh ở đây
│   ├── features.py             # load_audio(), extract_mfcc(), augment()
│   ├── dataset.py              # RAVDESSDataset, build_dataset()
│   ├── model.py                # CNN1D (baseline), CNN2D (fallback)
│   ├── train.py                # Training loop
│   ├── evaluate.py             # F1, Confusion Matrix
│   ├── inference.py            # predict() dùng cho Streamlit
│   └── utils.py                # set_seed(), logging
├── notebooks/
│   ├── 01_explore_data.ipynb   # EDA: phân phối nhãn, waveform
│   ├── 02_feature_demo.ipynb   # Visualize MFCC, Mel Spectrogram
│   └── 03_results_analysis.ipynb
├── checkpoints/                # Model weights (không commit)
├── app/
│   ├── streamlit_app.py        # Giao diện demo
│   └── recorder.py             # Ghi âm từ microphone
├── experiments/results.csv     # Experiment tracking
├── report/figures/
├── pyproject.toml
└── main.py
```

---

## Hyperparameters

Tất cả tham số tập trung tại [src/config.py](src/config.py). Chỉnh ở đây, không hardcode ở file khác.

| Tham số | Giá trị mặc định |
|---|---|
| Sample rate | 22.050 Hz |
| Duration | 3.0 giây |
| MFCC coefficients | 40 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Epochs | 50 |
| Dropout | 0.3 |
| Early stop patience | 7 |
| Train / Val / Test | 70% / 15% / 15% |
| Random seed | 42 |

---

## Kiến trúc Model (CNN-1D)

```
Input: (batch, 40, time_frames)
  → Conv1D(40→64, k=5) → BN → ReLU → MaxPool(2)
  → Conv1D(64→128, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
  → Conv1D(128→256, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
  → AdaptiveAvgPool1D(1) → Flatten
  → Linear(256→128) → ReLU → Dropout(0.3)
  → Linear(128→8)
Output: (batch, 8)  — logits
```

---

## Phân công

| Người | Phụ trách |
|---|---|
| **Phương** | `features.py`, `train.py`, `inference.py`, hyperparameter tuning |
| **Nhi** | `model.py`, `evaluate.py`, lý thuyết báo cáo |
| **Linh** | `dataset.py`, `streamlit_app.py`, `recorder.py`, README |

---

## Experiment Tracking

Sau mỗi lần train, kết quả được ghi vào [experiments/results.csv](experiments/results.csv):

```
run_id | date | model | lr | dropout | augment | val_acc | test_acc | f1_macro
```
