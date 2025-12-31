**Encoder**
*PatchEmbedding3D* (128³ → 32³, 4→64 channels)
↓
*Downsample* (32³ → 16³, 64→128 channels)
↓
*SE Block (Channel Attention)* (16³, 128 channels)
↓
*Downsample* (16³ → 8³, 128→256 channels)
↓
*Local Self Attention 1* (8³, 256 channels, window=4)
↓
*Downsample* (8³ → 4³, 256→512 channels)
↓
*Downsample* (4³ → 2³, 512→1024 channels)
↓
*Local Self Attention 2* (2³, 1024 channels, window=2)



**Bottleneck** (2³, 1024 channels)
*ASPP:*
  *- 1×1×1 conv*
  *- 3×3×3 dilated conv (dilation=1)*
  *- 3×3×3 dilated conv (dilation=2)*
  *- 3×3×3 dilated conv (dilation=3)*
  *→ concat → conv (1×1×1) → InstanceNorm → ReLU*
↓
*Global Self Attention* (8 heads, dropout=0.1)
↓
*conv (1×1×1)*
↓
*Residual Connection* (+ input)


**Decoder** (with skip connections)
*Upsample* (2³ → 4³, 1024→512) *+ skip4 (512 channels)*
↓
*Upsample* (4³ → 8³, 512→256) *+ skip3 (256 channels)*
↓
*Upsample* (8³ → 16³, 256→128) *+ skip2 (128 channels)*
↓
*Upsample* (16³ → 32³, 128→64) *+ skip1 (64 channels)*
↓
*Segmentation Head:*
  *- 3×3×3 conv refinement (64 channels)*
  *- Trilinear upsample (32³ → 128³)*
  *- 1×1×1 conv (64→1 channel)*
  *→ Logits (128³)*