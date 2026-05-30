# Kế Hoạch Bài Tập Lớn: Speech Emotion Recognition (SER)

> **Môn:** Nhập môn Trí tuệ Nhân tạo  
> **Nhóm:** Phạm Thị Bích Phương (Lead) · Nguyễn Hải Yến Nhi · Nguyễn Thị Hải Linh

---

## 1. Bài Toán

### Mô tả
Xây dựng hệ thống **phân loại cảm xúc từ giọng nói** (Speech Emotion Recognition). Đầu vào là một đoạn âm thanh (.wav), đầu ra là nhãn cảm xúc kèm xác suất dự đoán.

### Phạm vi
- Phân loại **6–8 nhãn cảm xúc**: Neutral, Calm, Happy, Sad, Angry, Fearful *(mở rộng Disgust, Surprised nếu kịp tiến độ)*
- Đầu vào: file `.wav` upload **hoặc** ghi âm trực tiếp từ microphone (record-then-predict, không phải real-time streaming)
- Không xử lý: âm thanh nhiễu nặng, giọng đa ngữ, luồng âm thanh thời gian thực liên tục

### Phân biệt "live recording" vs "real-time streaming"
> **Live recording** = ghi âm 3–5 giây → lưu buffer → predict → hiện kết quả.  
> Về kỹ thuật đây giống upload file, chỉ khác nguồn tạo ra file. Khác hoàn toàn với real-time streaming (model chạy liên tục trên audio chưa kết thúc). Vì vậy **không vi phạm phạm vi đề ra**.

### Mục tiêu kỹ thuật
| Chỉ số | Mục tiêu |
|---|---|
| Accuracy (test set) | ≥ 65% |
| F1-Score (macro) | ≥ 0.60 |
| Latency demo (record → result) | < 2 giây |
| Pretrained weights | Không sử dụng |

---

## 2. Dữ Liệu

### Dataset: RAVDESS
| Thuộc tính | Chi tiết |
|---|---|
| Tên đầy đủ | Ryerson Audio-Visual Database of Emotional Speech and Song |
| Số file | 1.440 file `.wav` |
| Số diễn viên | 24 (12 nam, 12 nữ) |
| Số nhãn emotion | 8 (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised) |
| Sample rate | 48.000 Hz (chuẩn hóa về 22.050 Hz khi xử lý) |
| Kích thước | ~24 MB |

### File naming convention
```
03-01-06-01-02-01-12.wav
│  │  │  │  │  │  └─ Actor ID
│  │  │  │  │  └──── Repetition
│  │  │  │  └─────── Statement
│  │  │  └────────── Intensity (1=normal, 2=strong)
│  │  └───────────── Emotion (01=neutral, 02=calm, ..., 08=surprised)
│  └──────────────── Vocal channel
└─────────────────── Modality
```

**Emotion mapping:**
```
"01" → neutral   "02" → calm     "03" → happy    "04" → sad
"05" → angry     "06" → fearful  "07" → disgust  "08" → surprised
```

### Đặc điểm cần lưu ý
- **Class imbalance**: Neutral ít hơn các emotion khác (chỉ có 1 intensity level) → dùng Weighted Cross-Entropy Loss
- **Domain mismatch**: Giọng diễn viên studio ≠ giọng thật của người dùng → cần trình bày trong báo cáo
- **Split strategy**: Stratified split theo nhãn, tỉ lệ 70/15/15 (train/val/test), seed cố định = 42
- **Label order**: Cố định theo `EMOTION_LIST` trong `config.py` để tránh lệch thứ tự class giữa train, evaluate và inference

### Feature Extraction Pipeline
```
Raw .wav
   ↓ librosa.load(sr=22050)
   ↓ librosa.effects.trim()        ← cắt silence đầu/cuối
   ↓ Fix length (3.0 giây)         ← padding zeros / truncate
   ↓ librosa.feature.mfcc()        ← 40 hệ số MFCC
   ↓ Normalize (trừ mean, chia std) ← per-sample
   ↓ shape: (40, time_frames)      ← input cho CNN-1D
```

### Data Augmentation (chỉ áp dụng train set)
```python
# Random chọn 1 trong 3 kỹ thuật:
time_stretch  → rate = random(0.9, 1.1)
pitch_shift   → n_steps = random(-2, +2)
white_noise   → y + 0.005 * np.random.randn(len(y))
```

---

## 3. Cấu Trúc Project

```
speech-emotion-recognition/
├── data/                              # RAVDESS (gitignore)
│   └── ravdess/
│       ├── Actor_01/
│       └── ...
│
├── src/                               # Source code chính
│   ├── __init__.py
│   ├── config.py                      # Tất cả hyperparameters & paths
│   ├── features.py                    # extract_mfcc(), augment()
│   ├── dataset.py                     # RAVDESSDataset, build_dataset()
│   ├── model.py                       # CNN1D, CNN2D (nếu cần)
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # F1, Confusion Matrix, plots
│   └── utils.py                       # Helpers (seed, logging...)
│
├── notebooks/
│   ├── 01_explore_data.ipynb          # EDA: phân phối nhãn, plot audio
│   ├── 02_feature_demo.ipynb          # Visualize MFCC, Mel Spectrogram
│   └── 03_results_analysis.ipynb      # So sánh các run, phân tích lỗi
│
├── checkpoints/                       # Model weights (gitignore)
│   └── best_model.pt
│
├── app/
│   ├── streamlit_app.py               # Demo chính
│   └── recorder.py                    # Live recording module
│
├── report/
│   ├── report.tex / report.docx
│   └── figures/                       # Biểu đồ, confusion matrix
│
├── experiments/
│   └── results.csv                    # Experiment tracking (Phương giữ)
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py                            # Entry point
```

### config.py — tất cả tham số tập trung tại đây
```python
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "ravdess"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Audio
SAMPLE_RATE = 22050
DURATION = 3.0          # seconds
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# Labels
EMOTIONS = {
    "01": "neutral", "02": "calm",    "03": "happy",    "04": "sad",
    "05": "angry",   "06": "fearful", "07": "disgust",  "08": "surprised"
}
EMOTION_LIST = list(EMOTIONS.values())

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.3
EARLY_STOP_PATIENCE = 7

# Split
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_SEED = 42
```

### dataset.py — trách nhiệm rõ ràng
- `parse_filename()` chỉ đọc emotion code từ tên file và map qua `EMOTIONS` trong `config.py`
- `RAVDESSDataset.__getitem__()` chỉ gọi `process_audio()` và trả về `(feature, label)`
- `build_dataset()` chỉ làm 3 việc: quét file `.wav`, stratified split theo ratio trong `config.py`, và trả về `train_ds, val_ds, test_ds, class_names`
- `dataset.py` không giữ bản copy riêng của emotion mapping và không tự tính class weights cho loss

---

## 4. Kiến Trúc Model

### 4.1 Model chính — CNN-1D (baseline)

**Lý do chọn CNN-1D:**  
MFCC là đặc trưng dạng chuỗi thời gian 1D (time_frames). CNN-1D trượt kernel dọc theo chiều thời gian, phù hợp hơn CNN-2D với dữ liệu dạng sequence. Nhẹ hơn, train nhanh hơn trên dataset nhỏ.

```
Input: (batch, 40, time_frames)
   │
   ├─ Conv1D(40→64, k=5) → BN → ReLU → MaxPool(2)
   ├─ Conv1D(64→128, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
   ├─ Conv1D(128→256, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
   │
   ├─ AdaptiveAvgPool1D(1)     ← Global Average Pooling
   ├─ Flatten → (batch, 256)
   ├─ Linear(256→128) → ReLU → Dropout(0.3)
   └─ Linear(128→8)            ← logits (KHÔNG softmax ở đây)

Output: (batch, 8)   logits → softmax khi inference
```

**Lý do từng thành phần:**

| Layer | Lý do |
|---|---|
| Conv1D | Trích đặc trưng cục bộ theo thời gian |
| BatchNorm | Ổn định training, giảm phụ thuộc vào learning rate |
| ReLU | Non-linearity đơn giản, không vanishing gradient |
| MaxPool | Giảm chiều, tăng receptive field |
| Dropout | Giảm overfitting trên dataset nhỏ (1440 file) |
| GlobalAvgPool | Thay thế Flatten lớn, giảm số params |
| CrossEntropyLoss | Loss chuẩn cho multi-class, đã include log-softmax |

**Hyperparameters mặc định:**
```python
optimizer = Adam(lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=3, factor=0.5)
criterion = CrossEntropyLoss(weight=class_weights)
```

**Ước tính số parameters:** ~400k–600k

---

### 4.2 Kiến trúc nâng cao 1 — CNN-2D + Mel Spectrogram

**Khi nào dùng:** Nếu CNN-1D không đạt 65% accuracy sau tuần 5.

**Lý do:** Mel Spectrogram là ma trận 2D (frequency × time) — có thể xử lý như ảnh, cho phép CNN-2D học được pattern 2 chiều đồng thời.

```
Input: Raw audio
   ↓ librosa.feature.melspectrogram()    → shape: (128, time)
   ↓ librosa.power_to_db()               → dB scale
   ↓ Normalize
   ↓ shape: (1, 128, time)               → như grayscale image

CNN-2D:
   ├─ Conv2D(1→32, k=3×3) → BN → ReLU → MaxPool(2×2)
   ├─ Conv2D(32→64, k=3×3) → BN → ReLU → MaxPool(2×2) → Dropout
   ├─ Conv2D(64→128, k=3×3) → BN → ReLU → MaxPool(2×2) → Dropout
   ├─ AdaptiveAvgPool2D(1,1) → Flatten
   ├─ Linear(128→64) → ReLU → Dropout
   └─ Linear(64→8)
```

---

### 4.3 Kiến trúc nâng cao 2 — CNN-1D + LSTM (nếu có thời gian)

**Khi nào dùng:** Nếu cả hai model trên đã ổn định và nhóm muốn thêm experiment cho báo cáo.

**Lý do:** LSTM nắm bắt dependency dài hạn theo thời gian tốt hơn CNN thuần. Kết hợp CNN trích đặc trưng cục bộ + LSTM học ngữ cảnh thời gian.

```
Input: (batch, 40, T)
   ↓ CNN-1D blocks (như model chính, bỏ GAP)
   ↓ shape: (batch, 256, T')
   ↓ Permute → (batch, T', 256)
   ↓ LSTM(256, hidden=128, layers=2, bidirectional=True)
   ↓ Lấy hidden state cuối → (batch, 256)
   ↓ Linear(256→8)
```

---

## 5. Kế Hoạch Thực Hiện

> Không tính theo ngày/tháng — tính theo **lượng công việc** (story points).  
> Mỗi story point ≈ 2–3 giờ làm việc tập trung.

### Phân công cố định

| Người | Role | Trách nhiệm chính |
|---|---|---|
| **Phương** | Lead & Core ML | Feature extraction, training pipeline, hyperparameter tuning, điều phối |
| **Nhi** | Model Design & Theory | Thiết kế CNN architecture, evaluate.py, viết lý thuyết báo cáo |
| **Linh** | Data & Demo | DataLoader, Streamlit app, live recording, README |

---

### Giai đoạn 0 — Tự học (song song với Giai đoạn 1)

**Mục tiêu:** Cả 3 người đủ nền tảng để hỗ trợ nhau và trả lời câu hỏi thầy.

| Tài liệu | Phương | Nhi | Linh |
|---|---|---|---|
| 3Blue1Brown Neural Networks (4 video) | Xem lướt (~30') | Xem kỹ (~1.5h) | Xem kỹ (~1.5h) |
| Valerio Velardo — Audio Signal Processing (video 1,2,4,5,10,15,16,17,18,19,20) | Video 1,2,4,5 (~1.5h) | Xem đủ 11 video (~4.5h) | Xem đủ 11 video (~4.5h) |
| StatQuest: Softmax, F1, BatchNorm, Dropout, Adam | Lướt nhanh | Xem kỹ (~1h) | Xem kỹ (~1h) |
| PyTorch Basics (7 bài) + CIFAR10 tutorial | Bỏ qua | Tự thực hành (~2h) | Tự thực hành (~2h) |
| CS231n CNN article | Bỏ qua | Đọc kỹ (~1h) | Đọc lướt (~30') |
| **Tổng** | **~2 tiếng** | **~10 tiếng** | **~10 tiếng** |

---

### Giai đoạn 1 — Setup & Khảo sát (~8 story points)

**Mục tiêu:** Repo chạy được, data load được, cả nhóm hiểu dataset.

#### Phương (Lead) — 3 SP
- [ ] **SP1.1** Tạo GitHub repo, setup cấu trúc thư mục, viết `.gitignore`, tạo branch `dev`
- [ ] **SP1.2** Viết `requirements.txt`, `config.py`, `utils.py` (seed, logging)
- [ ] **SP1.3** Code `features.py`: `load_audio()`, `extract_mfcc()`, `augment()` + notebook demo MFCC

#### Nhi — 2 SP
- [ ] **SP1.4** Nghiên cứu CNN architecture: tính output shape từng lớp trên giấy, phác thảo `model.py`
- [ ] **SP1.5** Đọc CS231n + PyTorch Conv1d docs → chuẩn bị trình bày kiến trúc cho team

#### Linh — 3 SP
- [ ] **SP1.6** Download RAVDESS, đọc file naming convention, nghe thử từng emotion
- [ ] **SP1.7** Viết `notebook 01_explore_data.ipynb`: phân phối nhãn, plot waveform, thống kê độ dài
- [ ] **SP1.8** Viết `dataset.py`: `parse_filename()`, `RAVDESSDataset`, `build_dataset()` với stratified split; dùng ratio/label mapping từ `config.py`

**Checkpoint G1:** `DataLoader(train_ds, batch_size=32)` chạy không lỗi, in ra batch shape `(32, 40, T)`.

---

### Giai đoạn 2 — Xây dựng Model & Pipeline (~7 story points)

**Mục tiêu:** Pipeline end-to-end chạy được, dù accuracy thấp.

#### Phương — 3 SP
- [ ] **SP2.1** Viết `train.py`: training loop, Adam optimizer, Weighted CrossEntropy, Early Stopping
- [ ] **SP2.2** Viết checkpoint save/load, setup `experiments/results.csv` để tracking
- [ ] **SP2.3** Chạy training lần đầu, vẽ loss/accuracy curve, báo cáo kết quả cho team

#### Nhi — 2 SP
- [ ] **SP2.4** Code `model.py` theo thiết kế đã phác thảo, unit test output shape
- [ ] **SP2.5** Viết `evaluate.py`: `evaluate()`, `plot_confusion_matrix()`, tính F1 macro/weighted

#### Linh — 2 SP
- [ ] **SP2.6** Hoàn thiện luồng augmentation cho `train_ds`; cung cấp labels/class names ổn định để `train.py` tính Weighted Loss weights
- [ ] **SP2.7** Tạo `app/streamlit_app.py` skeleton: 2 tab (upload file / ghi âm), chưa cần kết nối model

**Checkpoint G2:** `python src/train.py` chạy end-to-end 10 epoch, lưu checkpoint, không lỗi.

---

### Giai đoạn 3 — Tối ưu Model & Hoàn thiện Demo (~8 story points)

**Mục tiêu:** Đạt accuracy ≥ 65%, demo chạy được cả 2 mode.

#### Phương — 4 SP
- [ ] **SP3.1** Grid search hyperparameters: learning rate ∈ {1e-3, 5e-4, 1e-4}, dropout ∈ {0.2, 0.3, 0.5}
- [ ] **SP3.2** Thử Data Augmentation on/off, so sánh kết quả, ghi vào `results.csv`
- [ ] **SP3.3** Nếu CNN-1D < 65% → implement Mel Spectrogram + CNN-2D, so sánh 2 hướng trong notebook riêng
- [ ] **SP3.4** Viết `src/inference.py`: hàm predict từ buffer audio → logits → softmax → label + xác suất (Linh dùng để tích hợp vào Streamlit)

#### Nhi — 2 SP
- [ ] **SP3.5** Phân tích Confusion Matrix: nhãn nào nhầm nhiều nhất? Tại sao?
- [ ] **SP3.6** Bắt đầu viết báo cáo phần lý thuyết (mục 2): MFCC, Softmax, CrossEntropy, F1

#### Linh — 2 SP
- [ ] **SP3.7** Viết `app/recorder.py`: `record_audio()`, `save_to_wav()`, test với microphone thật
- [ ] **SP3.8** Kết nối model vào Streamlit dùng `inference.py`, polish UI: waveform visualizer, countdown timer, color per emotion

**Checkpoint G3:**
- `streamlit run app/streamlit_app.py` → upload file + ghi âm đều predict được
- Val accuracy ≥ 65%

---

### Giai đoạn 4 — Đánh giá, Báo cáo & Bàn giao (~6 story points)

**Mục tiêu:** Tài liệu hoàn chỉnh, sẵn sàng thuyết trình.

#### Phương — 3 SP
- [ ] **SP4.1** Tổng hợp bảng experiment từ `results.csv`, viết mục 3 (Phương pháp) và mục 4 (Kết quả)
- [ ] **SP4.2** Làm slide thuyết trình 12–15 trang, quay video demo backup
- [ ] **SP4.3** Test demo với giọng thật của cả 3 người, ghi lại tỉ lệ đúng/sai, viết phân tích domain mismatch vào báo cáo

#### Nhi — 2 SP
- [ ] **SP4.4** Hoàn thiện phần lý thuyết báo cáo, giải thích tại sao chọn từng lớp CNN có công thức toán
- [ ] **SP4.5** Viết mục Thảo luận: domain mismatch, nhãn khó, hướng cải tiến

#### Linh — 1 SP
- [ ] **SP4.6** Viết `README.md` đầy đủ, clean code, push GitHub final

**Checkpoint G4 (Final):**
- [ ] Demo chạy được live trên máy tính nộp
- [ ] Báo cáo hoàn chỉnh ≥ 8 trang
- [ ] Slide ready
- [ ] Cả 3 người trả lời được 7 câu hỏi chuẩn bị sẵn

---

### Tổng hợp story points

| Giai đoạn | Phương | Nhi | Linh | Tổng |
|---|---|---|---|---|
| G0 (Tự học) | 1 SP | 5 SP | 5 SP | 11 SP |
| G1 (Setup) | 3 SP | 2 SP | 3 SP | 8 SP |
| G2 (Build) | 3 SP | 2 SP | 2 SP | 7 SP |
| G3 (Optimize) | 4 SP | 2 SP | 2 SP | 8 SP |
| G4 (Report) | 3 SP | 2 SP | 1 SP | 6 SP |
| **Tổng** | **14 SP** | **13 SP** | **13 SP** | **40 SP** |

> 1 SP ≈ 2–3 giờ. Tổng ~80–120 giờ cho cả nhóm. Phương ~28–42h, Nhi & Linh ~26–39h mỗi người.

---

### Câu hỏi chuẩn bị cho buổi báo cáo

> Thầy có thể hỏi bất kỳ ai — cả 3 người đều phải trả lời được.

1. Tại sao dùng MFCC mà không dùng raw waveform?
2. Tại sao Conv1D mà không phải Conv2D?
3. Batch Normalization đặt ở đâu, làm được gì?
4. Tại sao dùng Adam, khác SGD thế nào?
5. F1-Score khác Accuracy ở điểm gì? Khi nào F1 quan trọng hơn?
6. Model overfit — làm sao biết và xử lý thế nào?
7. Test với giọng thật tại sao không chính xác bằng dataset?

---

## Rủi ro & Giải pháp

| Rủi ro | Khả năng | Giải pháp |
|---|---|---|
| CNN-1D không đạt 65% acc | Trung bình | Switch sang Mel Spectrogram + CNN-2D (Phương quyết định ở G3) |
| Overfitting do dataset nhỏ | Cao | Dropout + Augmentation + Early Stopping + Weighted Loss |
| Class imbalance | Chắc chắn | Weighted CrossEntropyLoss (weights tỉ lệ nghịch tần suất nhãn) |
| Live recording không hoạt động khi demo | Thấp | Quay video demo backup, upload file .wav thay thế |
| Domain mismatch giọng thật vs RAVDESS | Chắc chắn | Trình bày như phân tích trong báo cáo, không coi là lỗi |

---

*Tài liệu này tổng hợp toàn bộ thảo luận của nhóm. Cập nhật lần cuối trước khi bắt đầu code.*
