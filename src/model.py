# model.py — kiến trúc CNN
# Class chính: CNN1D (baseline), CNN2D (fallback nếu acc < 65%)
#
# NOTE:
# File này hiện đang chứa một bản baseline "dummy" để nối pipeline train/test.
# "Dummy" ở đây nghĩa là:
# - Kiến trúc đã hợp lệ và có thể train được
# - Đúng với hướng CNN-1D trong plan
# - Nhưng vẫn là bản đầu để kiểm tra end-to-end, chưa phải phiên bản final tối ưu

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from src.config import DROPOUT, N_FEATURES, NUM_CLASSES
except ModuleNotFoundError:
    from config import DROPOUT, N_FEATURES, NUM_CLASSES


class CNN1D(nn.Module):
    """Baseline CNN-1D cho SER trên đặc trưng MFCC.

    Input mong đợi:
        x.shape = (batch_size, channels, time_frames)
        Trong đó channels = N_FEATURES trong config.py.

    Output:
        logits.shape = (batch_size, num_classes)

    Đây là bản dummy baseline:
    - Dùng để chốt contract giữa model.py và train.py
    - Dùng để test shape/output trước khi tinh chỉnh sâu hơn
    - Chưa được coi là kiến trúc tối ưu cuối cùng
    """

    def __init__(
        self,
        in_channels: int = N_FEATURES,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # AdaptiveAvgPool1d(1) giúp model không phụ thuộc cứng vào số time frames.
        # Dù T thay đổi đôi chút, đầu ra sau pooling vẫn về (batch, 256, 1).
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "CNN1D expected a 3D input shaped "
                "(batch, channels, time_frames)."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                "CNN1D expected input channels "
                f"{self.in_channels}, but got {x.size(1)}. "
                "If your input is shaped (batch, time_frames, n_features), "
                "use x.permute(0, 2, 1)."
            )
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class CNN2D(nn.Module):
    """Placeholder cho hướng mở rộng Mel Spectrogram + CNN-2D."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "CNN2D chưa được implement. Hiện tại chỉ dùng CNN1D baseline."
        )


def _smoke_test() -> None:
    """Kiểm tra nhanh output shape bằng input giả."""
    model = CNN1D()
    dummy_x = torch.randn(4, N_FEATURES, 130)
    dummy_logits = model(dummy_x)
    print("Input shape :", tuple(dummy_x.shape))
    print("Output shape:", tuple(dummy_logits.shape))


if __name__ == "__main__":
    _smoke_test()
