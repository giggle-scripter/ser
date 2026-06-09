# config.py — nguồn sự thật cho tất cả hyperparameters và paths
# KHÔNG hardcode giá trị này ở file khác — luôn import từ đây
from pathlib import Path

ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data" / "ravdess"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
FIGURE_DIR     = ROOT_DIR / "report" / "figures"

for d in [CHECKPOINT_DIR, EXPERIMENT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
DURATION    = 3.0
FEATURE_TYPE = "mfcc_delta"  # options: "mfcc", "mfcc_delta"
RUN_NAME = "cnn1d_mfcc_delta_fix1"
N_MFCC      = 40
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128

if FEATURE_TYPE == "mfcc":
    N_FEATURES = N_MFCC
elif FEATURE_TYPE == "mfcc_delta":
    N_FEATURES = N_MFCC * 3
else:
    raise ValueError(f"Unknown FEATURE_TYPE: {FEATURE_TYPE!r}")

EMOTIONS = {
    "01": "neutral", "02": "calm",    "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fearful", "07": "disgust", "08": "surprised",
}
EMOTION_LIST = list(EMOTIONS.values())
NUM_CLASSES  = len(EMOTIONS)

BATCH_SIZE            = 32
LEARNING_RATE         = 1e-3
WEIGHT_DECAY          = 1e-4
EPOCHS                = 50
DROPOUT               = 0.3
EARLY_STOP_PATIENCE   = 7
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR   = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
