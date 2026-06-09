# Speech Emotion Recognition

Hệ thống nhận diện cảm xúc giọng nói sử dụng PyTorch trên bộ dữ liệu RAVDESS. Dự án hỗ trợ trích xuất đặc trưng MFCC hoặc MFCC + delta + delta-delta, huấn luyện mô hình CNN1D, đánh giá kết quả và lưu từng experiment theo cấu trúc reproducible.

## Tổng Quan

| Hạng mục | Mô tả |
|---|---|
| Dataset | RAVDESS, 1.440 file `.wav`, 8 cảm xúc |
| Model chính | CNN1D |
| Feature | `mfcc` hoặc `mfcc_delta` |
| Output | 8 nhãn: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised` |
| Demo | Giao diện upload/ghi âm để dự đoán cảm xúc |

## Cài Đặt

Yêu cầu:

- Python 3.12+
- `uv`

```bash
uv sync
```

## Chuẩn Bị Dataset

Tải RAVDESS từ Zenodo và giải nén vào:

```text
data/
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

Thư mục `data/` không được commit.

## Cấu Hình Experiment

Chỉnh experiment trong [src/config.py](src/config.py):

```python
FEATURE_TYPE = "mfcc_delta"  # "mfcc" hoặc "mfcc_delta"
RUN_NAME = "cnn1d_mfcc_delta_fix1"
```

Chạy baseline MFCC:

```python
FEATURE_TYPE = "mfcc"
RUN_NAME = "cnn1d_mfcc_baseline"
```

Chạy model MFCC + delta + delta-delta:

```python
FEATURE_TYPE = "mfcc_delta"
RUN_NAME = "cnn1d_mfcc_delta_fix1"
```

## Chạy Dự Án

Training:

```bash
uv run python -m src.train
```

Evaluate:

```bash
uv run python -m src.evaluate
```

Smoke test model:

```bash
uv run python src/model.py
```

Demo giao diện:

```bash
uv run streamlit run app/streamlit_app.py
```

## Cấu Trúc Thư Mục

```text
ser/
├── app/                         # Giao diện demo và ghi âm
├── checkpoints/                 # Checkpoint theo từng run
│   └── <run_name>/
│       ├── best_model.pt
│       ├── last_model.pt
│       ├── config.json
│       └── metrics.json
├── data/ravdess/                # Dataset RAVDESS
├── docs/                        # Tài liệu kế hoạch
├── experiments/results.csv      # Tổng hợp kết quả experiment
├── notebooks/                   # Notebook EDA và phân tích
├── report/figures/              # Hình ảnh báo cáo
├── src/
│   ├── config.py                # Hyperparameters và paths
│   ├── dataset.py               # Dataset và stratified split
│   ├── evaluate.py              # Đánh giá và confusion matrix
│   ├── features.py              # Tiền xử lý audio và feature extraction
│   ├── inference.py             # Load model và predict
│   ├── model.py                 # CNN1D
│   ├── train.py                 # Training loop và checkpoint
│   └── utils.py                 # Seed, device, result logging
└── main.py
```

## Pipeline

```text
Audio .wav
  -> load + trim silence
  -> pad/truncate về 3 giây
  -> extract MFCC hoặc MFCC + delta + delta-delta
  -> normalize per sample
  -> CNN1D
  -> logits
  -> softmax khi inference
```

## Experiment Tracking

Mỗi lần train/evaluate có thể ghi kết quả vào:

```text
experiments/results.csv
```

Các cột chính:

```text
run_name, model, feature_type, test_loss, test_acc, test_f1_macro, test_f1_weight, notes
```

Checkpoint được lưu theo từng experiment:

```text
checkpoints/<RUN_NAME>/
├── best_model.pt
├── last_model.pt
├── config.json
└── metrics.json
```

## Kết Quả Hiện Tại

| Run | Feature | Test Acc | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| `cnn1d_mfcc_baseline` | MFCC | 0.6991 | 0.6963 | 0.7025 |
| `cnn1d_mfcc_delta_fix1` | MFCC + delta + delta-delta | 0.7315 | 0.7274 | 0.7350 |
