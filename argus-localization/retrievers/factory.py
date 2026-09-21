"""Builds the configured Retriever, so eval scripts can swap models without code changes.

config.yaml `retriever.kind` picks the implementation, and per-kind settings
live under `retriever.<kind>`. Checkpoint and model paths are machine-specific,
so they live in user_config.yaml. Nothing here downloads weights: every path
must already exist locally.

retriever_id() names a (kind, settings) combination. It keys the reference-DB
cache and the results files, so two retrievers, or two settings of one, never
share a cache.
"""

import hashlib
import os
from typing import Any

from core.interfaces import Retriever

RETRIEVER_KINDS = ("earthloc", "remoteclip", "qwen3vl_embedding")

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "earthloc": {},
    "remoteclip": {
        "model_name": "ViT-B-32",
        "image_size": 224,
        "quick_gelu": False,
        "resize_mode": "squash",
        "backend": "auto",
        "max_batch": 128,
    },
    "qwen3vl_embedding": {
        "model": "Qwen3-VL-Embedding-2B",
        "image_size": 320,
        "dim": None,
        "instruction": None,
        "batch_size": 32,
        "attn_implementation": "sdpa",
    },
}


def retriever_settings(
    kind: str, config: dict, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Defaults, then config.yaml's retriever.<kind> section, then CLI overrides."""
    if kind not in RETRIEVER_KINDS:
        raise ValueError(f"unknown retriever kind {kind!r}; choose from {RETRIEVER_KINDS}")
    settings = dict(DEFAULT_SETTINGS[kind])
    settings.update((config.get("retriever") or {}).get(kind) or {})
    settings.update(overrides or {})
    unknown = set(settings) - set(DEFAULT_SETTINGS[kind])
    if unknown:
        raise ValueError(f"unknown {kind} setting(s): {sorted(unknown)}")
    return settings


def retriever_id(kind: str, settings: dict[str, Any]) -> str:
    """Filesystem-safe name for a (kind, settings) combination.

    Only settings that change the embeddings are included. Batch sizes and the
    loading backend are not.
    """
    if kind == "earthloc":
        return "earthloc"
    if kind == "remoteclip":
        parts = ["remoteclip", settings["model_name"], str(settings["image_size"]), settings["resize_mode"]]
        if settings["quick_gelu"]:
            parts.append("quickgelu")
        return "-".join(parts)
    parts = [settings["model"], str(settings["image_size"])]
    if settings["dim"]:
        parts.append(f"d{settings['dim']}")
    if settings["instruction"]:
        # Instructions are free text; keep the id short and stable.
        parts.append("instr" + hashlib.sha1(settings["instruction"].encode()).hexdigest()[:8])
    return "-".join(parts)


def _require_file(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. Put it there by hand (nothing is downloaded "
            "automatically), or fix the path in user_config.yaml."
        )
    return path


def build_retriever(
    kind: str, settings: dict[str, Any], user_config: dict, device: str
) -> Retriever:
    if kind == "earthloc":
        from retrievers.earthloc_retriever import EarthLocRetriever

        checkpoint = _require_file(user_config["earthloc_checkpoint"], "EarthLoc checkpoint")
        return EarthLocRetriever(checkpoint, device=device)

    if kind == "remoteclip":
        from retrievers.remoteclip_retriever import RemoteCLIPRetriever, checkpoint_filename

        checkpoint = _require_file(
            os.path.join(user_config["remoteclip_checkpoint_dir"], checkpoint_filename(settings["model_name"])),
            "RemoteCLIP checkpoint",
        )
        return RemoteCLIPRetriever(checkpoint, device=device, **settings)

    if kind == "qwen3vl_embedding":
        from retrievers.qwen3vl_embedding_retriever import Qwen3VLEmbeddingRetriever

        model_dir = os.path.join(user_config["qwen3vl_embedding_dir"], settings["model"])
        _require_file(os.path.join(model_dir, "config.json"), f"{settings['model']} model directory")
        kwargs = {k: v for k, v in settings.items() if k != "model"}
        return Qwen3VLEmbeddingRetriever(model_dir, device=device, **kwargs)

    raise ValueError(f"unknown retriever kind {kind!r}; choose from {RETRIEVER_KINDS}")
