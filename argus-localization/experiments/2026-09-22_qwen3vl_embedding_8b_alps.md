# Qwen3-VL-Embedding-8B zero-shot on EarthLoc Alps

**Date:** 2026-09-22. **Branch:** `replace/vlm-earthloc`. **Code:** commit `b69882f`. **Machine:** shared GPU workstation, one RTX 4090 (24 GB).

**Question:** does the larger Qwen embedding model beat the 2B, and does either approach EarthLoc? Follows [the 2B run](2026-09-22_qwen3vl_embedding_zero_shot_alps.md). (There is no 7B in this family: Qwen3-VL-Embedding ships as 2B and 8B. Qwen2.5-VL-7B is a text-generating model and cannot be used as a retriever.)

## Setup

Identical to the 2B run except the model and batch size: the same 2,393 Alps astronaut photos against 52,951 Sentinel-2 tiles (2021, zooms 9-11) at 4 rotations, correct at footprint IoU >= 0.2, same `scripts/evaluate.py`, images squash-resized to 320x320 (100 visual tokens each), default instruction `Represent the user's input.`, last-token pooling, bf16, sdpa.

| | |
|---|---|
| Model | [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B), 4 shards, 16.29 GB, all SHA-256s verified after download |
| Size | **8.14B parameters** (measured on load), 4096-d embedding, 36 layers |
| Batch | 16 (down from the 2B's 32, to stay inside 24 GB) |

```
RETRIEVER_OPTS="model=Qwen3-VL-Embedding-8B batch_size=16" \
  PYTHON=~/envs/argus-vlm/bin/python scripts/run_zero_shot_alps.sh qwen3vl_embedding
```

## Results

| Alps, same harness, zero-shot | R@1 | R@5 | R@10 | R@15 | R@100 | No correct tile in top 100 |
|---|---|---|---|---|---|---|
| RemoteCLIP ViT-B-32, GELU | 2.9 | 7.7 | 10.2 | 13.2 | 34.4 | 1,571 |
| **Qwen3-VL-Embedding-8B** | **8.7** | **19.1** | **25.8** | **30.0** | **59.5** | **970** |
| Qwen3-VL-Embedding-2B | 10.5 | 24.2 | 32.4 | 38.3 | 66.9 | 791 |
| RemoteCLIP ViT-L-14, QuickGELU | 12.3 | 26.5 | 35.9 | 40.3 | 69.0 | 742 |
| EarthLoc (README baseline) | 56.8 | 69.7 | -- | 76.4 | -- | -- |

**The 8B is worse than the 2B, and the gap is real, not noise.** Paired at R@5: 320 queries only the 2B gets, 199 only the 8B gets, exact two-sided McNemar p = 1.2e-07.

**Cost**, measured, same GPU and settings as the other runs:

| | Qwen-8B | Qwen-2B | RemoteCLIP ViT-L-14 QG |
|---|---|---|---|
| Database build (211,804 images) | **5,727 s (95 min)** | 3,626 s (60 min) | 994 s (17 min) |
| Throughput | **37 img/s** | 58 img/s | 213 img/s |
| Retrieval over 2,393 queries | **175 s** | 111 s | 55 s |
| Wall clock, end to end | **98 min** (15:50-17:28) | 62 min | 17 min |
| GPU memory, peak | **16.8 GB** | 5.6 GB | 4.7 GB |
| GPU utilization (60 s samples) | **mean 62%, peak 100%** | mean 37%, peak 96% | ~70% |
| Parameters | 8.14B | 2.13B | 304M |
| Weights on disk | 16.3 GB | 4.3 GB | 1.7 GB |
| Descriptor / index | 4096-d, 3.5 GB | 2048-d, 1.7 GB | 768-d, 0.65 GB |

## Summary

The 8B is the **worst value of anything tested**: 3.8x the parameters of the 2B, 1.6x the build time, 3x the GPU memory, and lower recall (R@5 19.1 vs 24.2). It is nowhere near EarthLoc's 69.7.

- **Scaling the model up does not help on this task.** Within RemoteCLIP, bigger did help (ViT-L-14 beat ViT-B-32 everywhere). Within Qwen it hurt. Both Qwen models were trained for general multimodal retrieval, so extra capacity buys better scene semantics, which is not what distinguishing neighbouring 100 km tiles needs.
- **This matches the picture from the other runs:** no zero-shot vision-language model comes close, and the best of them (RemoteCLIP ViT-L-14 QuickGELU, R@5 26.5) is still 43 points behind EarthLoc, which was trained on these exact tiles.

## Checks

- **Clean load:** the run log has no missing, unexpected or reinitialized-weight warnings, and the model reports 8.14B parameters, consistent with the 16.29 GB checkpoint.
- **Self-retrieval:** 6 randomly chosen rotated reference tiles each retrieve themselves at rank 1 (6/6), so the index, rotations and search work.
- **Weights:** all four shard checksums plus the tokenizer verified against Hugging Face after download.

## Raw outputs (not in git)

- **On this machine:** `argus-localization/output/server_runs/Alps_Qwen3-VL-Embedding-8B-320_*.json` plus logs and `nvidia-smi` logs.
- **On the workstation:** `~/argus-work/argus-localization/output/`, cache `cache/db_Qwen3-VL-Embedding-8B-320_Alps/` (3.5 GB), weights `weights/Qwen3-VL-Embedding-8B/` (16 GB). The root disk sits at 97% with these in place; both are safe to delete now that the result is recorded.
