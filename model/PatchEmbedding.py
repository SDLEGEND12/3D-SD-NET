import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, patch_size=(4,4,4)):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # ✅ Initialize layers properly in __init__
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # B, C, D, H, W
        x = self.proj(x)      # B, embed_dim, D', H', W'
        x = x.flatten(2)      # B, embed_dim, N
        x = x.transpose(1, 2) # B, N, embed_dim
        x = self.norm(x)
        return x