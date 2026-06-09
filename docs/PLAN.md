# Project Plan: Speech Emotion Recognition

## 1. Mục Tiêu

Xây dựng hệ thống nhận diện cảm xúc từ giọng nói bằng mô hình học sâu. Đầu vào là file âm thanh `.wav` hoặc đoạn ghi âm từ giao diện demo; đầu ra là nhãn cảm xúc dự đoán và xác suất tương ứng.

Mục tiêu kỹ thuật:

| Chỉ số | Mục tiêu |
|---|---|
| Dataset | RAVDESS |
| Số lớp | 8 cảm xúc |
| Model chính | CNN1D |
| Feature | MFCC hoặc MFCC + delta + delta-delta |
| Test accuracy | Tối thiểu khoảng 65% |
| Macro F1 | Tối thiểu khoảng 0.60 |
| Pretrained weights | Không sử dụng |

## 2. Phạm Vi

Hệ thống tập trung vào speech emotion recognition trên audio ngắn, đã được chuẩn hóa độ dài. Demo hỗ trợ upload file hoặc ghi âm ngắn theo luồng record-then-predict, không xử lý real-time streaming liên tục.

Không nằm trong phạm vi chính:

- Audio nhiễu nặng hoặc môi trường quá khác biệt so với dataset
- Nhận diện người nói
- Nhận diện ngôn ngữ
- Streaming inference liên tục theo thời gian thực

## 3. Dataset

RAVDESS gồm 1.440 file audio với 8 nhãn cảm xúc:

```text
neutral, calm, happy, sad, angry, fearful, disgust, surprised
```

File được đặt theo convention của RAVDESS, trong đó emotion code nằm ở phần thứ ba của tên file:

```text
03-01-06-01-02-01-12.wav
      ^^
      emotion code
```

Emotion mapping:

```text
01 -> neutral
02 -> calm
03 -> happy
04 -> sad
05 -> angry
06 -> fearful
07 -> disgust
08 -> surprised
```

Split strategy:

- Train / validation / test: 70% / 15% / 15%
- Stratified split theo nhãn
- Random seed cố định để tái lập kết quả

## 4. Feature Pipeline

Pipeline xử lý audio:

```text
Raw .wav
  -> librosa.load(sr=22050)
  -> trim silence
  -> pad/truncate về 3.0 giây
  -> extract MFCC
  -> tùy chọn thêm delta và delta-delta
  -> concatenate theo trục feature
  -> normalize per sample
  -> float32 tensor
```

Hai chế độ feature:

| Feature type | Output shape |
|---|---|
| `mfcc` | `(40, T)` |
| `mfcc_delta` | `(120, T)` |

Augmentation chỉ áp dụng trên train set:

- Time stretch
- Pitch shift
- White noise

Validation và test không dùng augmentation.

## 5. Model

Model chính là CNN1D nhận input dạng:

```text
(batch, n_features, time_frames)
```

Kiến trúc tổng quát:

```text
Input
  -> Conv1D + BatchNorm + ReLU + MaxPool
  -> Conv1D + BatchNorm + ReLU + MaxPool + Dropout
  -> Conv1D + BatchNorm + ReLU + MaxPool + Dropout
  -> AdaptiveAvgPool1D
  -> Linear + ReLU + Dropout
  -> Linear
  -> logits
```

Model trả về logits, không thêm Softmax trong `forward()`. Softmax chỉ dùng ở inference hoặc khi cần hiển thị xác suất.

## 6. Training

Training pipeline gồm:

- Load dataset bằng stratified split
- DataLoader cho train, validation và test
- CNN1D với input channels lấy từ `N_FEATURES`
- Weighted CrossEntropyLoss để giảm ảnh hưởng class imbalance
- Adam optimizer
- ReduceLROnPlateau scheduler
- Early stopping
- Lưu checkpoint theo từng experiment

Checkpoint structure:

```text
checkpoints/<run_name>/
├── best_model.pt
├── last_model.pt
├── config.json
└── metrics.json
```

Best checkpoint được chọn theo validation metric, ưu tiên `val_f1_macro`; nếu không có thì dùng `val_acc`.

## 7. Evaluation

Evaluation cần tính:

- Test loss
- Test accuracy
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix

Kết quả cuối được append vào:

```text
experiments/results.csv
```

Schema:

```text
run_name,model,feature_type,test_loss,test_acc,test_f1_macro,test_f1_weight,notes
```

## 8. Inference Và Demo

Inference phải load checkpoint kèm config đã lưu. Feature extraction khi predict phải khớp với `feature_type` trong checkpoint:

- `mfcc` -> input channels = 40
- `mfcc_delta` -> input channels = 120

Demo giao diện dự kiến hỗ trợ:

- Upload file `.wav`
- Ghi âm ngắn từ microphone
- Hiển thị nhãn cảm xúc dự đoán
- Hiển thị xác suất hoặc confidence score

## 9. Cấu Trúc Project

```text
ser/
├── app/
│   ├── recorder.py
│   └── streamlit_app.py
├── checkpoints/
│   └── <run_name>/
├── data/
│   └── ravdess/
├── docs/
│   └── PLAN.md
├── experiments/
│   └── results.csv
├── notebooks/
├── report/
│   └── figures/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── README.md
└── main.py
```

## 10. Milestones

### Milestone 1: Setup Và Data

- Tạo cấu trúc project
- Cài dependencies
- Chuẩn bị RAVDESS
- Viết parser filename và label mapping
- Kiểm tra DataLoader trả về batch đúng shape

### Milestone 2: Feature Và Dataset

- Implement MFCC extraction
- Implement MFCC + delta + delta-delta
- Normalize per sample
- Thêm augmentation cho train set
- Đảm bảo validation/test không augmentation

### Milestone 3: Model Và Training

- Implement CNN1D
- Thêm shape validation trong model
- Viết training loop
- Lưu checkpoint theo run
- Lưu config và metrics cùng checkpoint

### Milestone 4: Evaluation

- Load checkpoint theo run
- Đánh giá trên test set
- Tính accuracy, macro F1, weighted F1
- Lưu confusion matrix
- Append kết quả vào `experiments/results.csv`

### Milestone 5: Demo

- Load model từ checkpoint
- Extract feature đúng theo checkpoint config
- Kết nối upload/record audio vào inference
- Hiển thị kết quả dự đoán trên giao diện

### Milestone 6: Báo Cáo

- Tổng hợp experiment table
- Phân tích kết quả baseline và improved model
- Phân tích confusion matrix
- Nêu hạn chế về domain mismatch giữa RAVDESS và giọng thật
- Đề xuất hướng cải thiện

## 11. Kết Quả Experiment Chính

| Run | Feature | Test Loss | Test Acc | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| `cnn1d_mfcc_baseline` | MFCC | 0.8450 | 0.6991 | 0.6963 | 0.7025 |
| `cnn1d_mfcc_delta_fix1` | MFCC + delta + delta-delta | 0.7817 | 0.7315 | 0.7274 | 0.7350 |

## 12. Rủi Ro Và Hướng Xử Lý

| Rủi ro | Hướng xử lý |
|---|---|
| Dataset nhỏ, dễ overfit | Dropout, augmentation, early stopping |
| Class imbalance | Weighted CrossEntropyLoss |
| Giọng thật khác giọng diễn viên trong RAVDESS | Trình bày như domain mismatch |
| Giao diện ghi âm lỗi trên máy demo | Chuẩn bị phương án upload file `.wav` |
| Input feature sai shape | Lưu config trong checkpoint và validate input channels |
