import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSelfAttention3D(nn.Module):
    """
    Local (windowed) self-attention for 3D feature maps.
    Splits volume into small 3D windows (e.g. 4x4x4), applies attention inside each window.
    Much cheaper than global attention.
    """

    def __init__(self, dim, window_size=4, heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # ✅ No bias for stability
        self.proj = nn.Linear(dim, dim, bias=False)
        
        # ✅ Add dropout for regularization
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
        # ✅ Initialize with small weights
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.02)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.02)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        N = x.shape[1]

        # Check perfect cube
        s = round((N) ** (1/3))
        if s ** 3 != N:
            raise ValueError(f"LocalAttention: N={N} is not a perfect cube")

        # Reshape to 3D grid
        x = x.reshape(B, s, s, s, C)

        # Window partitioning
        w = self.window_size

        # Pad if needed
        pad_d = (w - s % w) % w
        pad_h = (w - s % w) % w
        pad_w = (w - s % w) % w

        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[1:4]

        # Split into windows
        x = x.unfold(1, w, w).unfold(2, w, w).unfold(3, w, w)
        x = x.contiguous().view(-1, w*w*w, C)

        # QKV
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # ✅ Attention with clamping for stability
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-10, max=10)  # ✅ Prevent extreme values
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Restore structure
        out = out.view(B, Dp // w, Hp // w, Wp // w, w, w, w, C)
        out = out.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, Dp, Hp, Wp, C)

        # Remove padding
        out = out[:, :s, :s, :s, :]

        # Convert back to (B, C, D, H, W)
        out = out.permute(0, 4, 1, 2, 3)

        return out
