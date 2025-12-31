import torch
import torch.nn as nn

from .encoder import Encoder3D
from .BottleNeck import Bottleneck3D
from .decoder import Decoder3D
from .seg_head import SegmentationHead3D


class Custom3DSegModel(nn.Module):
    """
    Full 3D Segmentation Model:
        PatchEmbedding (dynamic channels)
        Encoder
        Bottleneck (ASPP + Global Attention)
        Decoder (with skip connections)
        Segmentation Head
    """

    def __init__(self,in_channels=4,
                 embed_dim=64,
                 n_classes=1,
                 final_activation="sigmoid"):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder = Encoder3D(in_channels=in_channels,embed_dim=embed_dim)

        # Bottleneck takes highest channel count: embed_dim * 16 = 1024
        self.bottleneck = Bottleneck3D(channels=embed_dim * 16)

        self.decoder = Decoder3D(embed_dim=embed_dim)

        # Final segmentation head
        self.seg_head = SegmentationHead3D(
            in_channels=embed_dim,  # decoder final output is 64 channels
            n_classes=n_classes,
            up_factor=4,            # decoder outputs 32³ → upsample to 128³
            final_activation=final_activation
        )

    def forward(self, x):
        # ✅ Use encoder's forward method
        x, skips = self.encoder(x, return_skips=True)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder(x, skips)
        
        # Segmentation head
        logits = self.seg_head(x, return_probs=False)
        return logits
