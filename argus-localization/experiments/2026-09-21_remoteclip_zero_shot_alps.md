# RemoteCLIP zero-shot on EarthLoc Alps

**Date:** 2026-09-21. **Branch:** `replace/vlm-earthloc`. **Code:** commit `fa74bd4`. **Machine:** shared GPU workstation, one RTX 4090.

**Question:** can RemoteCLIP replace EarthLoc as the retriever without any training?

## Setup

| | |
|---|---|
| Model | RemoteCLIP ViT-B-32 image tower ([ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), HF `chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt`, SHA-256 `60014e39…85c4`) |
| Model size | **87.85M parameters** used for retrieval (151.28M in the full image+text checkpoint); 512-d embedding; 224x224 input. EarthLoc: about 27.6M, 4096-d, 320x320 |
| Settings | `remoteclip-ViT-B-32-224-squash`: squash-resize to 224, OpenAI CLIP mean/std, plain GELU, timm backend, bf16 |
| Queries | 2,393 astronaut photos from EarthLoc's query set (NASA *Gateway to Astronaut Photography of Earth*), taken with the ISS within 2,500 km of (45N, 10E). 2,394 are in range; 1 has no correct tile and is skipped |
| Database | 52,951 EarthLoc Sentinel-2 tiles: 2021, zooms 9-11, within 5,000 km. Each is embedded at 4 rotations, giving 211,804 vectors |
| Ground truth | A retrieved tile is correct if its footprint IoU with the photo's footprint is >= 0.2 |
| Harness | `scripts/evaluate.py`, the same one that produced the README's EarthLoc baseline, with exact cosine search (numpy fallback, since the workstation has no faiss) |

Command (from `argus-localization/`, through the GPU queue):

```
PYTHON=~/envs/argus-vlm/bin/python scripts/run_zero_shot_alps.sh remoteclip
```

## Results

| Alps, same harness | R@1 | R@5 | R@10 | R@15 | R@100 |
|---|---|---|---|---|---|
| RemoteCLIP ViT-B-32, zero-shot | **2.9** | **7.7** | **10.2** | **13.2** | **34.4** |
| EarthLoc (README baseline) | 56.8 | 69.7 | -- | 76.4 | -- |

For 1,571 of the 2,393 queries (66%), no correct tile appears anywhere in the top 100. The database build took 668 s (317 img/s), and retrieval over all queries took 64 s.

## Summary

Zero-shot, RemoteCLIP is not a usable replacement for EarthLoc: 7.7 vs 69.7 at R@5.

- **What RemoteCLIP was trained on:** high-resolution aerial, satellite and UAV images (about 0.05 to a few m/px), captioned by object, such as buildings, ships and planes. It has never seen astronaut photos, and never seen tiles like these: 75-300 m/px, about 100 km across.
- **What EarthLoc was trained on:** these exact Sentinel-2 reference tiles. It was never trained on astronaut photos either. So the comparison favours EarthLoc, but not by enough to explain a gap of about 60 points.

## Checks that the number is real

- **Embeddings:** they match open_clip's `encode_image` to within about 1e-6, both locally and on the workstation.
- **Self-retrieval:** 20 randomly chosen reference tiles, rotated 90°, each retrieve themselves at rank 1 (20/20). The index, rotations and search are working.
- **Same database as the baseline:** it has exactly the 52,951 tiles behind the README's EarthLoc numbers.

## Notes

- **Zoom scoping:** the first attempt took every 2021 folder, 215,611 tiles. The folders are named `<year>_<zoom>`, and the data now holds zooms 8-12. Commit `fa74bd4` added `eval.db_zooms: [9, 10, 11]` to match the baseline; nothing was reported from the unscoped attempt.
- **Environment:** the server env `~/envs/argus-vlm` is a venv on top of the shared `partstad` conda env (torch 2.6, timm 1.0.14). Nothing was installed.

## Raw outputs (not in git)

- **On this machine:** `argus-localization/output/server_runs/`
  - `Alps_remoteclip-ViT-B-32-224-squash_20260921-144630.json`: recalls, settings, database scope, and each query's first-hit rank.
  - run logs, preflight output, and `nvidia-smi` logs.
- **On the workstation:** `~/argus-work/argus-localization/output/`, plus the database cache `cache/db_remoteclip-ViT-B-32-224-squash_Alps/`.
