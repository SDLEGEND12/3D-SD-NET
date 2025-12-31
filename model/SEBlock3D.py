import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.
    Designed for mid-scale (e.g., 16x16x16) attention.

    Benefits:
    - Very stable (batch size = 1 safe)
    - Extremely low compute
    - Improves NCR / semantic consistency
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        b, c, _, _, _ = x.shape

        # Squeeze: global context
        y = self.avg_pool(x).view(b, c)

        # Excitation: channel importance
        y = self.fc(y).view(b, c, 1, 1, 1)

        # Reweight channels
        return x * y
