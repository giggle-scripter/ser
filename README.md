# Speech Emotion Recognition

Hệ thống nhận diện cảm xúc giọng nói sử dụng PyTorch trên bộ dữ liệu RAVDESS. Dự án hỗ trợ trích xuất đặc trưng MFCC hoặc MFCC + delta + delta-delta, huấn luyện mô hình CNN1D, đánh giá kết quả và demo dự đoán cảm xúc bằng Streamlit.

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

Có thể chỉnh experiment bằng biến môi trường hoặc trong [src/config.py](src/config.py):

| Biến | Ý nghĩa | Ví dụ |
|---|---|---|
| `SER_RUN_NAME` | Tên checkpoint/experiment | `cnn1d_mfcc_delta_ls005_adamw` |
| `SER_FEATURE_TYPE` | Loại đặc trưng | `mfcc`, `mfcc_delta` |
| `SER_OPTIMIZER` | Optimizer | `adam`, `adamw` |
| `SER_LABEL_SMOOTHING` | Label smoothing | `0.05` |
| `SER_DROPOUT` | Dropout | `0.25` |

Chạy baseline MFCC:

```powershell
$env:SER_RUN_NAME="cnn1d_mfcc_baseline_organic"
$env:SER_FEATURE_TYPE="mfcc"
$env:SER_OPTIMIZER="adam"
uv run python -m src.train
```

Chạy model cuối:

```powershell
$env:SER_RUN_NAME="cnn1d_mfcc_delta_ls005_adamw"
$env:SER_FEATURE_TYPE="mfcc_delta"
$env:SER_OPTIMIZER="adamw"
$env:SER_LABEL_SMOOTHING="0.05"
$env:SER_DROPOUT="0.25"
uv run python -m src.train
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

| Run | Mục đích | Feature | Optimizer | Test Acc | Macro F1 | Weighted F1 |
|---|---|---|---|---:|---:|---:|
| `cnn1d_mfcc_baseline_organic` | Baseline chính | MFCC | Adam | 0.7315 | 0.7251 | 0.7312 |
| `cnn1d_mfcc_delta_fix1_organic` | Thử thêm delta/delta-delta | MFCC + delta + delta-delta | Adam | 0.6435 | 0.6292 | 0.6416 |
| `cnn1d_mfcc_delta_ls005_do025_adam` | Thử regularization | MFCC + delta + delta-delta | Adam | 0.6991 | 0.6895 | 0.6940 |
| `cnn1d_mfcc_delta_ls005_adamw` | Model cuối | MFCC + delta + delta-delta | AdamW | 0.9167 | 0.9145 | 0.9173 |

Model cuối được chọn là `cnn1d_mfcc_delta_ls005_adamw` vì đạt kết quả tốt nhất trên test set sau khi re-evaluate checkpoint hiện tại. Thử nghiệm delta/delta-delta riêng lẻ không ổn định trên split hiện tại, nên phần cải thiện chính đến từ cấu hình regularization và AdamW.
