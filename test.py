'''import torch
from model.model import Custom3DSegModel
# Test with different channel counts
test_cases = [
    (1, "CT Scan (Liver/Kidney)"),
    (2, "Dual-contrast MRI"),
    (4, "BraTS (current)"),
    (3, "RGB pathology"),
]

for in_ch, desc in test_cases:
    model = Custom3DSegModel(
        in_channels=in_ch,
        embed_dim=64,
        n_classes=4
    )
    
    x = torch.randn(1, in_ch, 128, 128, 128)
    out = model(x)
    
    print(f"✅ {desc}: Input {x.shape} → Output {out.shape}")

# Expected output:
# ✅ CT Scan (Liver/Kidney): Input torch.Size([1, 1, 128, 128, 128]) → Output torch.Size([1, 4, 128, 128, 128])
# ✅ Dual-contrast MRI: Input torch.Size([1, 2, 128, 128, 128]) → Output torch.Size([1, 4, 128, 128, 128])
# ✅ BraTS (current): Input torch.Size([1, 4, 128, 128, 128]) → Output torch.Size([1, 4, 128, 128, 128])
# ✅ RGB pathology: Input torch.Size([1, 3, 128, 128, 128]) → Output torch.Size([1, 4, 128, 128, 128])'''


'''
import torch
from model.model import Custom3DSegModel
model = Custom3DSegModel()
model.eval()

# Hook to detect if attn16 is called
called = {'attn16': False}

def hook(module, input, output):
    called['attn16'] = True

model.encoder.attn16.register_forward_hook(hook)

# Test forward pass
x = torch.randn(1, 4, 128, 128, 128)
with torch.no_grad():
    out = model(x)

print(f"SEBlock (attn16) called: {called['attn16']}")  # Should print True'''

import torch
from model.model import Custom3DSegModel
model = Custom3DSegModel(in_channels=4, embed_dim=64, n_classes=1)
model.eval()

# Track if SEBlock is called
called = {'attn16': False}

def hook(module, input, output):
    called['attn16'] = True
    print(f"✅ SEBlock called! Input shape: {input[0].shape}")

model.encoder.attn16.register_forward_hook(hook)

# Test forward pass
x = torch.randn(1, 4, 128, 128, 128)
with torch.no_grad():
    logits = model(x)

print(f"\n{'='*50}")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {logits.shape}")
print(f"SEBlock used: {called['attn16']}")
print(f"{'='*50}")

# Expected output:
# ✅ SEBlock called! Input shape: torch.Size([1, 128, 16, 16, 16])
# Input shape:  torch.Size([1, 4, 128, 128, 128])
# Output shape: torch.Size([1, 1, 128, 128, 128])
# SEBlock used: True