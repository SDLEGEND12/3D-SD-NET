import torch
import torch.nn as nn

from .PatchEmbedding import PatchEmbedding3D
from .Downsample3D import Downsample3D
from .attention import LocalSelfAttention3D
from .SEBlock3D import SEBlock3D

class Encoder3D(nn.Module):
    """
    Encoder:
        PatchEmbedding (4x downsample)
        ↓
        Downsample
        ↓
        Downsample
        ↓
        LocalSelfAttention
        ↓
        Downsample
        ↓
        Downsample
        ↓
        LocalSelfAttention
    """

    def __init__(self,in_channels=4, embed_dim=64, window_size=4):
        super().__init__()

        self.embed_dim = embed_dim

        # -------------------------------
        # 1. PATCH EMBEDDING
        # -------------------------------
        self.patch_embed = PatchEmbedding3D(in_channels=in_channels,embed_dim=embed_dim,
                                            patch_size=(4,4,4))

        # -------------------------------
        # 2. FIRST TWO DOWNSAMPLES
        # -------------------------------
        # Output channels: 64 → 128 → 256
        self.down1 = Downsample3D(embed_dim, embed_dim * 2)       # 64 → 128
        self.attn16 = SEBlock3D(channels=embed_dim * 2)
        self.down2 = Downsample3D(embed_dim * 2, embed_dim * 4)   # 128 → 256

        # -------------------------------
        # 3. FIRST LOCAL SELF ATTENTION
        # Feature map at this stage: 256 × 8 × 8 × 8
        # -------------------------------
        self.attn1 = LocalSelfAttention3D(dim=embed_dim * 4,
                                          window_size=window_size,
                                          heads=4)

        # -------------------------------
        # 4. NEXT TWO DOWNSAMPLES
        # -------------------------------
        # Channels: 256 → 512 → 1024
        self.down3 = Downsample3D(embed_dim * 4, embed_dim * 8)    # 256 → 512
        self.down4 = Downsample3D(embed_dim * 8, embed_dim * 16)   # 512 → 1024

        # -------------------------------
        # 5. SECOND LOCAL SELF ATTENTION
        # Feature map here: 1024 × 2 × 2 × 2
        # -------------------------------
        self.attn2 = LocalSelfAttention3D(dim=embed_dim * 16,
                                          window_size=2,   # small because spatial = 2
                                          heads=8)

    def forward(self, x, return_skips=False):
        # Patch embedding
        x = self.patch_embed(x)
        B, N, C = x.shape
        S = round(N ** (1/3))
        if S ** 3 != N:
            raise ValueError(f"Cannot infer cubic shape from N={N}")
        x = x.transpose(1, 2).reshape(B, C, S, S, S)
        skip1 = x.clone()
        
        # Downsampling with SE block
        x = self.down1(x)
        x = self.attn16(x)  # ✅ SE Block is used!
        skip2 = x.clone()
        
        x = self.down2(x)
        x = self.attn1(x)
        skip3 = x.clone()
        
        x = self.down3(x)
        skip4 = x.clone()
        
        x = self.down4(x)
        x = self.attn2(x)
        
        if return_skips:
            return x, [skip1, skip2, skip3, skip4]
        return x
