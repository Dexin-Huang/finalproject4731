# SAM3D Body Inference Optimization Plan

Target: Real-time inference (~60-80ms for 4 frames) on RTX 4090

## Current Bottleneck

SAM3D Body pipeline (~800ms total):
1. SAM segmentation (~100ms/frame)
2. Depth estimation (~50ms/frame)
3. SMPL mesh fitting (~50ms/frame)

## Optimization Strategies

### 1. TensorRT the SAM Backbone (Biggest Impact)

```python
# SAM's image encoder is 60-70% of time
import torch_tensorrt

sam.image_encoder = torch_tensorrt.compile(
    sam.image_encoder,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 1024, 1024])],
    enabled_precisions={torch.float16}
)
```

### 2. Swap SAM for FastSAM

```python
# FastSAM is 50x faster, minimal accuracy loss
from fastsam import FastSAM
model = FastSAM('FastSAM-x.pt')  # ~10ms vs ~100ms
```

### 3. Batch Process 4 Frames

```python
# Single forward pass instead of 4 sequential
frames_batch = torch.stack([f0, f1, f2, f3])  # [4, 3, H, W]
embeddings = sam.image_encoder(frames_batch)
```

### 4. Cache Frame Embeddings

```python
# Consecutive frames are similar - compute once, reuse
embedding = sam.image_encoder(release_frame)
# Reuse for t-2, t-1, t, t+1 with decoder only
```

### 5. 4090-Specific Settings

```python
# Enable TF32 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use Flash Attention for ViT
# Compile with torch.compile() for PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")
```

## Expected Results

| Optimization | Time (4 frames) |
|--------------|-----------------|
| Baseline SAM3D | ~800ms |
| + TensorRT FP16 | ~300ms |
| + FastSAM swap | ~150ms |
| + Batch processing | ~100ms |
| + Frame caching | ~60-80ms |

**Target: ~60-80ms = 12-15 FPS** (sufficient for real-time betting window)

## Hardware Requirements

- RTX 4090 (24GB VRAM)
- CUDA 12.x
- TensorRT 8.6+
- PyTorch 2.0+

## Next Steps

1. Benchmark current pipeline on 4090
2. Export SAM encoder to TensorRT
3. Test FastSAM as drop-in replacement
4. Implement batched inference
5. Profile and identify remaining bottlenecks
