import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead3D(nn.Module):
    """
    Segmentation head for 3D volumes.

    Expected input:
        x: (B, in_channels, D', H', W')  # e.g., (B,64,32,32,32)
    Output:
        logits: (B, n_classes, D_out, H_out, W_out)
        probs:  optional activation applied (sigmoid/softmax) if return_probs=True
    """

    def __init__(self,
                 in_channels: int = 64,
                 n_classes: int = 1,
                 mid_channels: int = 64,
                 up_factor: int = 4,
                 dropout: float = 0.0,
                 final_activation: str = "sigmoid"):
        """
        Args:
            in_channels: channels from decoder (default 64)
            n_classes: number of segmentation classes (1 for binary)
            mid_channels: channels inside the head conv block
            up_factor: integer scale factor to upsample to original spatial size
                       (e.g., decoder outputs 32 -> original 128 => up_factor=4)
            dropout: dropout rate before final conv (0.0 disables)
            final_activation: "sigmoid", "softmax", or None
        """
        super().__init__()
        self.up_factor = up_factor
        self.n_classes = n_classes
        self.final_activation = final_activation

        # refinement conv block
        self.refine = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # final 1x1 conv to produce logits
        self.classifier = nn.Conv3d(mid_channels, n_classes, kernel_size=1)

    def forward(self, x, return_probs: bool = False):
        """
        Args:
            x: (B, in_channels, D', H', W')
            return_probs: if True, applies final activation to return probabilities
        Returns:
            logits OR (logits, probs) if return_probs True
        """
        x = self.refine(x)                    # (B, mid_channels, D', H', W')
        x = self.dropout(x)

        # Upsample to target resolution
        if self.up_factor != 1:
            x = F.interpolate(x,
                              scale_factor=self.up_factor,
                              mode='trilinear',
                              align_corners=False)

        logits = self.classifier(x)           # (B, n_classes, D_out, H_out, W_out)

        if return_probs:
            if self.final_activation is None:
                probs = logits
            elif self.final_activation == "sigmoid":
                probs = torch.sigmoid(logits)
            elif self.final_activation == "softmax":
                # softmax across channel dim
                probs = torch.softmax(logits, dim=1)
            else:
                raise ValueError("final_activation must be one of {None, 'sigmoid', 'softmax'}")
            return logits, probs

        return logits
