import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data" / "ravdess"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
FIGURE_DIR     = ROOT_DIR / "report" / "figures"

for d in [CHECKPOINT_DIR, EXPERIMENT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
DURATION    = 3.0
FEATURE_TYPE = env_str("SER_FEATURE_TYPE", "mfcc_delta")  # options: "mfcc", "mfcc_delta"
RUN_NAME = env_str("SER_RUN_NAME", "cnn1d_mfcc_delta_fix1")
MODEL_NAME = env_str("SER_MODEL_NAME", "CNN1D")  # options: "CNN1D", "CNN1D_ResSE"
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
LABEL_ENCODING = env_str("SER_LABEL_ENCODING", "ravdess")  # "ravdess" or "legacy"

BATCH_SIZE            = 32
LEARNING_RATE         = env_float("SER_LEARNING_RATE", 1e-3)
WEIGHT_DECAY          = env_float("SER_WEIGHT_DECAY", 1e-4)
OPTIMIZER_NAME        = env_str("SER_OPTIMIZER", "adamw")  # options: "adam", "adamw"
EPOCHS                = 50
DROPOUT               = env_float("SER_DROPOUT", 0.3)
EARLY_STOP_PATIENCE   = 7
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR   = 0.5
BEST_METRIC           = env_str("SER_BEST_METRIC", "val_loss")  # options: "val_loss", "val_f1_macro", "val_acc"
BEST_METRIC_MODE      = env_str("SER_BEST_METRIC_MODE", "min")  # "min" for loss, "max" for acc/F1

AUG_TIME_STRETCH_MIN  = env_float("SER_AUG_TIME_STRETCH_MIN", 0.9)
AUG_TIME_STRETCH_MAX  = env_float("SER_AUG_TIME_STRETCH_MAX", 1.1)
AUG_PITCH_SHIFT_MIN   = env_float("SER_AUG_PITCH_SHIFT_MIN", -2.0)
AUG_PITCH_SHIFT_MAX   = env_float("SER_AUG_PITCH_SHIFT_MAX", 2.0)
AUG_NOISE_STD         = env_float("SER_AUG_NOISE_STD", 0.005)

CLASS_WEIGHT_SMOOTHING = env_float("SER_CLASS_WEIGHT_SMOOTHING", 0.0)
LABEL_SMOOTHING        = env_float("SER_LABEL_SMOOTHING", 0.0)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = env_int("SER_RANDOM_SEED", 42)
