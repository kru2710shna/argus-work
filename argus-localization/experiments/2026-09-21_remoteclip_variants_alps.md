# RemoteCLIP variants (ViT-L-14, QuickGELU) zero-shot on EarthLoc Alps

**Date:** 2026-09-21. **Branch:** `replace/vlm-earthloc`. **Code:** commit `b69882f`. **Machine:** shared GPU workstation, one RTX 4090.

**Question:** does a larger RemoteCLIP (ViT-L-14), or OpenAI CLIP's QuickGELU activation, close the gap to EarthLoc? This follows up [the ViT-B-32 run](2026-09-21_remoteclip_zero_shot_alps.md).

## Setup

Everything except the model matches the ViT-B-32 run:
- **Queries:** the same 2,393 Alps astronaut-photo queries.
- **Database:** the same 52,951 Sentinel-2 tiles (2021, zooms 9-11), each embedded at 4 rotations.
- **Ground truth:** footprint IoU >= 0.2.
- **Harness:** the same `scripts/evaluate.py`.

| Variant | Checkpoint | Parameters used | Embedding | Activation |
|---|---|---|---|---|
| ViT-B-32 | `RemoteCLIP-ViT-B-32.pt` (SHA-256 `60014e39…85c4`) | 87.85M | 512-d | GELU / QuickGELU |
| ViT-L-14 | `RemoteCLIP-ViT-L-14.pt` (SHA-256 `fcc2a7e2…cecb`) | 303.97M | 768-d | GELU / QuickGELU |

All inputs are 224x224, squash-resized, with OpenAI CLIP mean/std, bf16, timm backend. The activation is only a flag: the same weights load with either. RemoteCLIP's README builds plain GELU, but nothing documents which activation training used.

Commands (from `argus-localization/`, run in sequence through the GPU queue):

```
export PYTHON=~/envs/argus-vlm/bin/python
RETRIEVER_OPTS="quick_gelu=true"                     scripts/run_zero_shot_alps.sh remoteclip
RETRIEVER_OPTS="model_name=ViT-L-14"                 scripts/run_zero_shot_alps.sh remoteclip
RETRIEVER_OPTS="model_name=ViT-L-14 quick_gelu=true" scripts/run_zero_shot_alps.sh remoteclip
```

## Results

| Alps, same harness, zero-shot | R@1 | R@5 | R@10 | R@15 | R@100 | Queries with no correct tile in top 100 |
|---|---|---|---|---|---|---|
| RemoteCLIP ViT-B-32, GELU (previous run) | 2.9 | 7.7 | 10.2 | 13.2 | 34.4 | 1,571 |
| RemoteCLIP ViT-B-32, QuickGELU | 1.9 | 7.0 | 11.2 | 14.4 | 35.8 | 1,536 |
| RemoteCLIP ViT-L-14, GELU | 7.7 | 18.0 | 23.9 | 28.8 | 56.8 | 1,033 |
| **RemoteCLIP ViT-L-14, QuickGELU** | **12.3** | **26.5** | **35.9** | **40.3** | **69.0** | **742** |
| EarthLoc (README baseline) | 56.8 | 69.7 | -- | 76.4 | -- | -- |

**Paired comparison at R@5.** This counts queries where one variant hits and the other misses, with an exact two-sided McNemar test:
- **L-14 QuickGELU vs L-14 GELU:** 318 queries only QuickGELU finds, 113 only GELU finds, p = 1e-23.
- **B-32 QuickGELU vs B-32 GELU:** 53 vs 71, p = 0.13, not significant.
- **L-14 QuickGELU vs B-32 GELU:** 539 vs 89, p = 2e-79.

**Timing:**

| Variant | Database build | Throughput | Retrieval |
|---|---|---|---|
| B-32 QuickGELU | 669 s | 317 img/s | 44 s |
| L-14 GELU | 943 s | 225 img/s | 52 s |
| L-14 QuickGELU | 994 s | 213 img/s | 55 s |

ViT-L-14 used about 4.7 GB of GPU memory at about 70% utilization. Nothing else ran on that GPU.

## Summary

- **Best variant:** ViT-L-14 with QuickGELU, at R@5 26.5 vs EarthLoc's 69.7. That's 3.4x the ViT-B-32 result but still far from usable zero-shot. Even its R@100 (69.0) is below EarthLoc's R@5.
- **Model size matters:** at the same activation, ViT-L-14 beats ViT-B-32 at every k.
- **The activation matters for ViT-L-14 only.** QuickGELU gives L-14 a large, highly significant gain (R@5 +8.5) but makes no significant difference to B-32. That suggests, though nothing confirms it, that at least the L-14 checkpoint was trained with QuickGELU, OpenAI CLIP's own activation. If so, RemoteCLIP's README (plain GELU) under-sells L-14.
- **Training data, as in the ViT-B-32 record:** RemoteCLIP trained on high-resolution aerial, satellite and UAV images (about 0.05 to a few m/px). It never saw astronaut photos or 75-300 m/px tiles like these. EarthLoc trained on these exact tiles, though never on the astronaut photos either.

## Checks

- **Local, against open_clip:** all three new variants match open_clip's `encode_image` within 2.4e-7.
- **Workstation, against local:** the workstation's embeddings (timm 1.0.14) match the local ones within 3.6e-7.
- **Self-retrieval, ViT-L-14 QuickGELU:** 20 randomly chosen rotated reference tiles each retrieve themselves at rank 1 (20/20).

## Raw outputs (not in git)

- **On this machine:** `argus-localization/output/server_runs/`
  - `Alps_remoteclip-ViT-B-32-224-squash-quickgelu_*.json`, `Alps_remoteclip-ViT-L-14-224-squash_*.json` and `Alps_remoteclip-ViT-L-14-224-squash-quickgelu_*.json`: recalls, settings, database scope, and each query's first-hit rank.
  - `variants_alps.nohup.log`, per-run logs and `nvidia-smi` logs.
- **On the workstation:** `~/argus-work/argus-localization/output/`, plus the database caches `cache/db_remoteclip-*_Alps/`.
