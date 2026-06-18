# Speech Emotion Recognition (SER)

Nhận diện cảm xúc giọng nói bằng CNN1D trên bộ dữ liệu RAVDESS, sử dụng PyTorch.

## Tổng quan

| Hạng mục | Chi tiết |
|---|---|
| Dataset | RAVDESS — 1.440 file `.wav`, 24 diễn viên, 8 cảm xúc |
| Kiến trúc | CNN1D (3 conv block + AdaptiveAvgPool + FC) |
| Feature | `mfcc` (40 chiều) hoặc `mfcc_delta` (40×3 = 120 chiều) |
| Nhãn | `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised` |
| Kết quả tốt nhất | **Test Acc 91.67%**, Macro F1 91.45% (`cnn1d_mfcc_delta_ls005_adamw`) |
| Demo | Streamlit — upload file hoặc ghi âm trực tiếp |

---

## Cài đặt

Yêu cầu: Python 3.12+, [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

---

## Chuẩn bị dữ liệu

Tải RAVDESS từ [Zenodo](https://zenodo.org/record/1188976) và giải nén vào:

```
data/
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

> Thư mục `data/` không được commit (đã có trong `.gitignore`).

---

## Cấu trúc thư mục

```
ser/
├── app/
│   ├── recorder.py          # Ghi âm từ microphone
│   └── streamlit_app.py     # Giao diện demo
├── checkpoints/             # Checkpoint theo từng run
│   └── <run_name>/
│       ├── best_model.pt
│       ├── last_model.pt
│       ├── config.json
│       └── metrics.json
├── data/ravdess/            # Dataset (không commit)
├── docs/PLAN.md             # Kế hoạch dự án
├── experiments/
│   └── results.csv          # Tổng hợp kết quả tất cả experiment
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_feature_demo.ipynb
│   └── 03_results_analysis.ipynb
├── report/figures/          # Hình ảnh cho báo cáo
├── src/
│   ├── config.py            # Hyperparameter và đường dẫn
│   ├── dataset.py           # Dataset class + stratified split
│   ├── evaluate.py          # Đánh giá + confusion matrix
│   ├── features.py          # Load audio + extract MFCC/delta
│   ├── inference.py         # Load checkpoint + predict
│   ├── model.py             # CNN1D, CNN1D_ResSE
│   ├── train.py             # Training loop + checkpoint
│   └── utils.py             # Seed, device, ghi kết quả
├── main.py
└── pyproject.toml
```

---

## Pipeline

```
File .wav
  → load + trim silence (librosa)
  → pad / truncate về 3 giây
  → extract MFCC (40 hệ số)
      [mfcc_delta] → thêm delta + delta-delta → 120 chiều
  → normalize per-sample (mean=0, std=1)
  → CNN1D → logits
      [inference] → softmax → nhãn cảm xúc
```

---

## Cấu hình experiment

Hyperparameter đặt trong [`src/config.py`](src/config.py) hoặc ghi đè qua biến môi trường:

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `SER_RUN_NAME` | `cnn1d_mfcc_delta_fix1` | Tên checkpoint / experiment |
| `SER_FEATURE_TYPE` | `mfcc_delta` | `mfcc` hoặc `mfcc_delta` |
| `SER_OPTIMIZER` | `adamw` | `adam` hoặc `adamw` |
| `SER_LABEL_SMOOTHING` | `0.0` | Label smoothing (0.0–0.1) |
| `SER_DROPOUT` | `0.3` | Dropout rate |
| `SER_LEARNING_RATE` | `1e-3` | Learning rate |
| `SER_RANDOM_SEED` | `42` | Random seed |

---

## Chạy dự án

**Training:**

```bash
# Bash/Linux/macOS
SER_RUN_NAME=cnn1d_mfcc_delta_ls005_adamw \
SER_FEATURE_TYPE=mfcc_delta \
SER_OPTIMIZER=adamw \
SER_LABEL_SMOOTHING=0.05 \
SER_DROPOUT=0.25 \
uv run python -m src.train
```

```powershell
# PowerShell (Windows)
$env:SER_RUN_NAME="cnn1d_mfcc_delta_ls005_adamw"
$env:SER_FEATURE_TYPE="mfcc_delta"
$env:SER_OPTIMIZER="adamw"
$env:SER_LABEL_SMOOTHING="0.05"
$env:SER_DROPOUT="0.25"
uv run python -m src.train
```

**Evaluate:**

```bash
uv run python -m src.evaluate
```

**Smoke test model:**

```bash
uv run python src/model.py
```

**Demo Streamlit:**

```bash
uv run streamlit run app/streamlit_app.py
```

---

## Kết quả experiment

| Run | Feature | Optimizer | LS | DO | Test Acc | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|---:|---:|
| `cnn1d_mfcc_baseline_organic` | MFCC | Adam | 0.00 | 0.30 | 0.7315 | 0.7251 | 0.7312 |
| `cnn1d_mfcc_delta_fix1_organic` | MFCC+Δ | Adam | 0.00 | 0.30 | 0.6435 | 0.6292 | 0.6416 |
| `cnn1d_mfcc_delta_ls005_do025_adam` | MFCC+Δ | Adam | 0.05 | 0.25 | 0.6991 | 0.6895 | 0.6940 |
| **`cnn1d_mfcc_delta_ls005_adamw`** | **MFCC+Δ** | **AdamW** | **0.05** | **0.25** | **0.9167** | **0.9145** | **0.9173** |

> LS = label smoothing, DO = dropout. Δ = delta + delta-delta.

**Nhận xét:**
- Thêm delta/delta-delta riêng lẻ (`fix1_organic`) không cải thiện mà còn giảm accuracy — feature động học khó hội tụ hơn nếu chưa có regularization phù hợp.
- Kết hợp label smoothing + dropout + AdamW mới khai thác được lợi thế của MFCC+Δ, đẩy accuracy từ ~73% lên **91.67%**.
- Model cuối được chọn: `cnn1d_mfcc_delta_ls005_adamw` (best epoch 46/50, test loss 0.5792).

---

## Experiment tracking

Mỗi lần train/evaluate ghi kết quả vào `experiments/results.csv`:

```
run_name, model, feature_type, test_loss, test_acc, test_f1_macro, test_f1_weight, notes
```

Checkpoint lưu tại `checkpoints/<run_name>/`:

```
best_model.pt   — checkpoint tốt nhất theo val_loss
last_model.pt   — checkpoint cuối epoch
config.json     — hyperparameter đầy đủ
metrics.json    — kết quả test set
```

---

## Hạn chế & hướng phát triển

- Model chưa generalize tốt sang domain khác (SAVEE, `emotion_audio`) do phụ thuộc vào giọng diễn viên RAVDESS.
- Hướng tiếp theo: augmentation đa domain, fine-tune trên dữ liệu thực tế, hoặc chuyển sang pretrained audio backbone (wav2vec2, HuBERT).
