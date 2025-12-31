import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# UPSAMPLE BLOCK (ConvTranspose or Interp+Conv)
# ---------------------------------------------------------
class UpSample3D(nn.Module):
    """
    3D Upsampling Block:
        1. Upsample (trilinear or transposed conv)
        2. 3D Conv + InstanceNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, use_transpose=False):
        super().__init__()
        self.use_transpose = use_transpose

        if use_transpose:
            self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                         kernel_size=2, stride=2)
        else:
            # interpolation does not change channels
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # x: deeper tensor
        # skip: encoder skip tensor

        if self.use_transpose:
            x = self.up(x)
        else:
            x = self.up(x)
            x = self.conv1x1(x)

        # Ensure matching shapes (sometimes off by 1 voxel)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        if x.shape[-2] != skip.shape[-2]:
            x = F.pad(x, (0, 0, 0, skip.shape[-2] - x.shape[-2]))
        if x.shape[-3] != skip.shape[-3]:
            x = F.pad(x, (0, 0, 0, 0, 0, skip.shape[-3] - x.shape[-3]))

        # Concatenate along channels
        x = torch.cat([skip, x], dim=1)

        # Convolution refinement
        x = self.conv(x)
        return x


# ---------------------------------------------------------
# DECODER (mirrors encoder)
# ---------------------------------------------------------
class Decoder3D(nn.Module):
    """
    Decoder:
        Up from 1024 → 512 → 256 → 128 → 64
        Skip connections:
           encoder down4 → up1
           encoder down3 → up2
           encoder down2 → up3
           encoder down1 → up4

    """

    def __init__(self, embed_dim=64):
        super().__init__()

        # Up path
        self.up1 = UpSample3D(embed_dim * 16, embed_dim * 8)    # 1024 → 512
        self.up2 = UpSample3D(embed_dim * 8, embed_dim * 4)     # 512 → 256
        self.up3 = UpSample3D(embed_dim * 4, embed_dim * 2)     # 256 → 128
        self.up4 = UpSample3D(embed_dim * 2, embed_dim)         # 128 → 64

    def forward(self, x, skips):
        """
        x: bottleneck output (B,1024,2,2,2)
        skips: list of skip tensors from encoder:
            [skip1, skip2, skip3, skip4]
            shapes:
            skip4: (B,512,4,4,4)
            skip3: (B,256,8,8,8)
            skip2: (B,128,16,16,16)
            skip1: (B,64,32,32,32)
        """

        skip1, skip2, skip3, skip4 = skips

        # Match each skip with correct upsample level:
        x = self.up1(x, skip4)    # 1024→512, skip:512
        x = self.up2(x, skip3)    # 512→256, skip:256
        x = self.up3(x, skip2)    # 256→128, skip:128
        x = self.up4(x, skip1)    # 128→64,  skip:64

        return x
