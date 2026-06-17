from __future__ import annotations

import torch
import torch.nn as nn

try:
    from src.config import DROPOUT, MODEL_NAME, N_FEATURES, NUM_CLASSES
except ModuleNotFoundError:
    from config import DROPOUT, MODEL_NAME, N_FEATURES, NUM_CLASSES


class CNN1D(nn.Module):
    """CNN-1D classifier for SER features.

    Expected input:
        x.shape = (batch_size, channels, time_frames)

    Output:
        logits.shape = (batch_size, num_classes)
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

        # Keeps the classifier independent from small changes in frame count.
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


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for Conv1D feature maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1)
        return x * scale


class ResBlock1D(nn.Module):
    """Residual Conv1D block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.3,
        use_pool: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1D(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.relu(out + residual)
        out = self.se(out)
        out = self.pool(out)
        return out


class CNN1D_ResSE(nn.Module):
    """CNN1D variant with residual blocks and squeeze-excitation."""

    def __init__(
        self,
        in_channels: int = N_FEATURES,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResBlock1D(96, 128, kernel_size=5, dropout=dropout, use_pool=True),
            ResBlock1D(128, 192, kernel_size=5, dropout=dropout, use_pool=True),
            ResBlock1D(192, 256, kernel_size=5, dropout=dropout, use_pool=True),
            ResBlock1D(256, 256, kernel_size=3, dropout=dropout, use_pool=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "CNN1D_ResSE expected a 3D input shaped "
                "(batch, channels, time_frames)."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                "CNN1D_ResSE expected input channels "
                f"{self.in_channels}, but got {x.size(1)}. "
                "If your input is shaped (batch, time_frames, n_features), "
                "use x.permute(0, 2, 1)."
            )

        x = self.stem(x)
        x = self.features(x)
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        x = torch.cat([avg, mx], dim=1)
        logits = self.classifier(x)
        return logits


class CNN2D(nn.Module):
    """Placeholder for a Mel spectrogram CNN-2D variant."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "CNN2D chưa được implement. Hiện tại chỉ dùng CNN1D baseline."
        )


MODEL_REGISTRY = {
    "CNN1D": CNN1D,
    "CNN1D_ResSE": CNN1D_ResSE,
}


def create_model(
    model_name: str = MODEL_NAME,
    in_channels: int = N_FEATURES,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_name: {model_name!r}. "
            f"Expected one of {sorted(MODEL_REGISTRY)}."
        ) from exc

    return model_cls(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


def _smoke_test() -> None:
    model = create_model()
    sample_x = torch.randn(4, N_FEATURES, 130)
    sample_logits = model(sample_x)
    print("Model       :", model.__class__.__name__)
    print("Input shape :", tuple(sample_x.shape))
    print("Output shape:", tuple(sample_logits.shape))


if __name__ == "__main__":
    _smoke_test()
