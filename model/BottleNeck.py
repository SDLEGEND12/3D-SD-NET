import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# GLOBAL SELF ATTENTION (operates on whole 2x2x2 or 4x4x4)
# ---------------------------------------------------------
class GlobalSelfAttention3D(nn.Module):
    """Stable global self-attention"""
    
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
        # ✅ Small initialization
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.02)

    def forward(self, x):
        """x: (B, C, D, H, W)"""
        B, C, D, H, W = x.shape

        # Flatten: (B, D*H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # QKV
        qkv = self.qkv(x).reshape(B, D*H*W, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # ✅ Stable attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-10, max=10)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, D*H*W, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Back to (B, C, D, H, W)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        return out


class ASPP3D(nn.Module):
    """Stable ASPP"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # ✅ Reduced dilation rates for stability
        self.branch2 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=1, dilation=1  # ✅ Changed from 2
        )

        self.branch3 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=2, dilation=2  # ✅ Changed from 4
        )

        self.branch4 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=3, dilation=3  # ✅ Changed from 6
        )

        self.fuse = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Bottleneck3D(nn.Module):
    """Stable bottleneck"""

    def __init__(self, channels):
        super().__init__()

        self.aspp = ASPP3D(in_channels=channels, out_channels=channels)
        self.attn = GlobalSelfAttention3D(dim=channels, heads=8)
        self.conv_out = nn.Conv3d(channels, channels, kernel_size=1)
        
        # ✅ Add residual connection
        self.use_residual = True

    def forward(self, x):
        identity = x
        
        x = self.aspp(x)
        x = self.attn(x)
        x = self.conv_out(x)
        
        # ✅ Residual connection for stability
        if self.use_residual:
            x = x + identity
            
        return x