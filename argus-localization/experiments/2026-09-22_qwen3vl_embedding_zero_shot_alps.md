# Qwen3-VL-Embedding-2B zero-shot on EarthLoc Alps

**Date:** 2026-09-22. **Branch:** `replace/vlm-earthloc`. **Code:** commit `b69882f`. **Machine:** shared GPU workstation, one RTX 4090.

**Question:** can a general multimodal embedding model (Qwen3-VL-Embedding) replace EarthLoc as the retriever without training? Companion to the [ViT-B-32](2026-09-21_remoteclip_zero_shot_alps.md) and [ViT-L-14 / QuickGELU](2026-09-21_remoteclip_variants_alps.md) RemoteCLIP runs.

## Setup

Same queries, database, ground truth and harness as the RemoteCLIP runs: 2,393 Alps astronaut photos against 52,951 Sentinel-2 tiles (2021, zooms 9-11) embedded at 4 rotations, correct at footprint IoU >= 0.2, `scripts/evaluate.py`.

| | |
|---|---|
| Model | [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), `model.safetensors` SHA-256 `c73fa9ca…09c1` |
| Size | 2.13B parameters, **2048-d** embedding (EarthLoc: 27.6M, 4096-d; RemoteCLIP ViT-L-14: 304M, 768-d) |
| Input | Images squash-resized to 320x320 -> 100 visual tokens each, bf16, sdpa attention, batch 32 |
| Prompt | The model's default instruction, as a system message: `Represent the user's input.` Images carry no text of their own |
| Pooling | Last token's final hidden state, L2-normalized |
| Implementation | `retrievers/qwen3vl_embedding_retriever.py`, written on transformers' own Qwen3-VL classes. No `trust_remote_code`, and the HF repo's `scripts/` folder was deliberately not downloaded |
| Environment | `~/envs/argus-vlm` with `transformers` 5.17.0, installed there only with the admin's approval (2026-09-22) |

```
PYTHON=~/envs/argus-vlm/bin/python scripts/run_zero_shot_alps.sh qwen3vl_embedding
```

## Results

| Alps, same harness, zero-shot | R@1 | R@5 | R@10 | R@15 | R@100 | No correct tile in top 100 |
|---|---|---|---|---|---|---|
| RemoteCLIP ViT-B-32, GELU | 2.9 | 7.7 | 10.2 | 13.2 | 34.4 | 1,571 |
| **Qwen3-VL-Embedding-2B** | **10.5** | **24.2** | **32.4** | **38.3** | **66.9** | **791** |
| RemoteCLIP ViT-L-14, QuickGELU | 12.3 | 26.5 | 35.9 | 40.3 | 69.0 | 742 |
| EarthLoc (README baseline) | 56.8 | 69.7 | -- | 76.4 | -- | -- |

**Paired comparison at R@5** (queries one model gets and the other misses, exact two-sided McNemar):
- **vs RemoteCLIP ViT-L-14 QuickGELU:** 306 only Qwen, 362 only RemoteCLIP, p = 0.03. RemoteCLIP ViT-L-14 is slightly ahead, and the two disagree on a lot of queries.
- **vs RemoteCLIP ViT-B-32:** 505 only Qwen, 111 only RemoteCLIP, p = 6e-61.

**Cost**, measured on one RTX 4090 (24 GB), CUDA 12.4, torch 2.6, bf16, batch 32, 4 JPEG-decode threads, `nice 10`:

| | Qwen3-VL-Embedding-2B | RemoteCLIP ViT-L-14 QuickGELU | RemoteCLIP ViT-B-32 |
|---|---|---|---|
| Database build (211,804 images: 52,951 tiles x 4 rotations) | **3,626 s (60 min)** | 994 s (17 min) | 669 s (11 min) |
| Throughput | **58 img/s** | 213 img/s | 317 img/s |
| Retrieval over 2,393 queries (embed + search + IoU ground truth) | **111 s** | 55 s | 44 s |
| Wall clock, end to end | **62 min** (13:55-14:57) | 17 min | 12 min |
| GPU memory, peak | **5.6 GB** | 4.7 GB | ~2 GB |
| GPU utilization while running (63 samples, 60 s apart) | **mean 37%, peak 96%** | ~70% | -- |
| Parameters | 2.13B | 304M | 88M |
| Weights on disk | 4.3 GB (safetensors) | 1.7 GB | 0.6 GB |
| Descriptor / index size | 2048-d, 1.7 GB | 768-d, 0.65 GB | 512-d, 0.43 GB |

GPU utilization averaging 37% says the run is partly bound by JPEG decode and the processor on the CPU, not purely by the GPU, so a faster input pipeline would cut the 60 minutes somewhat. For scale, EarthLoc itself (27.6M parameters, 4096-d) builds the same database in about 11 min at 4096-d / 3.5 GB.

## Summary

Zero-shot, Qwen3-VL-Embedding-2B is **not** a usable replacement: R@5 24.2 vs EarthLoc's 69.7.

- It beats RemoteCLIP ViT-B-32 by a wide margin but lands just behind RemoteCLIP ViT-L-14 with QuickGELU, while costing about 4x the embedding time and 24x the parameters. Nothing about it justifies the extra cost here.
- Its reported training data and benchmarks contain no remote sensing, and the model has never seen astronaut photos or 75-300 m/px tiles. EarthLoc was trained on these exact reference tiles.
- A general-purpose multimodal embedder recognises *what kind of scene* an image shows. This task needs *which* 100 km patch it is, and telling neighbouring tiles of the same terrain apart is exactly what these models don't do.

**Untested variations that could move the number:** a task-specific instruction in place of the default, an asymmetric setup (instruction on queries, plain images for tiles), a larger input (512 px = 256 tokens), and the 8B model. None would plausibly close a 45-point gap without fine-tuning.

## Checks

- **Implementation:** matches the model's official helper to within 4e-4 (bf16 rounding), verified earlier on this machine's local copy.
- **Self-retrieval:** 10 randomly chosen rotated reference tiles each retrieve themselves at rank 1 (10/10).
- **Weights:** both published checksums verified after download.

## Raw outputs (not in git)

- **On this machine:** `argus-localization/output/server_runs/Alps_Qwen3-VL-Embedding-2B-320_*.json` plus logs and `nvidia-smi` logs.
- **On the workstation:** `~/argus-work/argus-localization/output/`, cache `cache/db_Qwen3-VL-Embedding-2B-320_Alps/`.
