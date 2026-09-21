"""Read-only preflight: can this machine run a given retriever, and is a GPU idle?

    python scripts/check_env.py                         # report on every retriever
    python scripts/check_env.py --retriever remoteclip  # exit 1 unless remoteclip can run

It never installs or downloads anything (on the shared GPU workstation that is
against the rules; see README "Running on the shared GPU workstation"). When
something is missing it says what, so it can be requested from the admin.
"""

import argparse
import importlib
import os
import subprocess
import sys
import zipfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.factory import RETRIEVER_KINDS, parse_retriever_opts, retriever_settings  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISSING_HINT = "missing: ask the workstation admin (do not pip install on the shared workstation)"


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - any import failure means "can't use it"
        return None
    return getattr(module, "__version__", "installed")


def version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split("+")[0].split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else '!!'}] {label}{': ' + detail if detail else ''}")
    return ok


def check_common(user_config: dict) -> bool:
    print("Python / GPU")
    ok = check("python", True, sys.version.split()[0])
    torch_version = module_version("torch")
    ok &= check("torch", torch_version is not None, torch_version or _MISSING_HINT)
    if torch_version:
        import torch

        cuda = torch.cuda.is_available()
        check(
            "CUDA",
            cuda,
            f"{torch.version.cuda}, {torch.cuda.device_count()} device(s)" if cuda else "not available (CPU only)",
        )
    try:
        from scripts.gpu_queue import describe, query_nvidia_smi

        gpus, apps = query_nvidia_smi()
        print(describe(gpus, apps, max_used_mb=1024, max_util=10))
    except (OSError, subprocess.SubprocessError) as err:
        check("nvidia-smi", False, str(err))

    print("Shared packages")
    for name in ("numpy", "PIL", "yaml", "tqdm", "shapely", "cv2"):
        version = module_version(name)
        ok &= check(name, version is not None, version or _MISSING_HINT)
    faiss_version = module_version("faiss")
    check("faiss", faiss_version is not None, faiss_version or "not installed (numpy index fallback, same results)")

    print("Data (read-only)")
    for key in ("queries_dir", "database_dir"):
        path = user_config.get(key, "")
        ok &= check(key, os.path.isdir(path), path)
    for key in ("cache_dir", "output_dir"):
        path = os.path.abspath(user_config.get(key, ""))
        parent = path if os.path.isdir(path) else os.path.dirname(path)
        check(f"{key} writable", os.access(parent, os.W_OK), path)
    return ok


def check_earthloc(settings: dict, user_config: dict) -> bool:
    ok = check(
        "third_party/EarthLoc (apl_models)",
        os.path.isdir(os.path.join(_REPO_ROOT, "third_party", "EarthLoc", "apl_models")),
    )
    return ok & check("checkpoint", os.path.exists(user_config["earthloc_checkpoint"]), user_config["earthloc_checkpoint"])


def check_remoteclip(settings: dict, user_config: dict) -> bool:
    from retrievers.remoteclip_retriever import _TIMM_ARCHS, checkpoint_filename, timm_arch

    open_clip_version = module_version("open_clip")
    timm_version = module_version("timm")
    backend = settings["backend"]
    if backend == "auto":
        backend = "open_clip" if open_clip_version else "timm"
    check("open_clip", open_clip_version is not None, open_clip_version or "not installed (timm fallback)")
    check("timm", timm_version is not None, timm_version or "not installed")
    if backend == "open_clip":
        ok = check("backend open_clip", open_clip_version is not None, open_clip_version or _MISSING_HINT)
    else:
        ok = check("backend timm", timm_version is not None, timm_version or _MISSING_HINT)
        if timm_version:
            import timm

            supported = settings["model_name"] in _TIMM_ARCHS
            arch = timm_arch(settings["model_name"], settings["quick_gelu"]) if supported else None
            ok &= check(
                f"timm arch for {settings['model_name']}",
                arch is not None and arch in timm.list_models(),
                arch or "RN50 needs open_clip",
            )
    path = os.path.join(user_config.get("remoteclip_checkpoint_dir", ""), checkpoint_filename(settings["model_name"]))
    # torch.save files are zip archives, and a cut-off copy loses the zip's
    # end-of-archive record, so this catches an interrupted transfer cheaply.
    complete = os.path.exists(path) and zipfile.is_zipfile(path)
    detail = path if complete or not os.path.exists(path) else f"{path} (incomplete: not a valid zip, re-copy it)"
    return ok & check("checkpoint", complete, detail)


def check_qwen3vl_embedding(settings: dict, user_config: dict) -> bool:
    version = module_version("transformers")
    ok = check("transformers", version is not None, version or _MISSING_HINT)
    if version:
        new_enough = version_tuple(version) >= (4, 57, 0)
        ok &= check("transformers >= 4.57 (Qwen3-VL)", new_enough, version)
        if new_enough:
            ok &= check(
                "transformers.models.qwen3_vl",
                module_version("transformers.models.qwen3_vl.modeling_qwen3_vl") is not None,
            )
    model_dir = os.path.join(user_config.get("qwen3vl_embedding_dir", ""), settings["model"])
    ok &= check("config.json", os.path.exists(os.path.join(model_dir, "config.json")), model_dir)
    has_weights = os.path.isdir(model_dir) and any(f.endswith(".safetensors") for f in os.listdir(model_dir))
    return ok & check("*.safetensors weights", has_weights, model_dir)


CHECKS = {
    "earthloc": check_earthloc,
    "remoteclip": check_remoteclip,
    "qwen3vl_embedding": check_qwen3vl_embedding,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only preflight for the retriever experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--user-config", default="user_config.yaml")
    parser.add_argument("--retriever", choices=RETRIEVER_KINDS, default=None, help="default: report on all")
    parser.add_argument(
        "--retriever-opt", action="append", default=[], metavar="KEY=VALUE",
        help="same overrides as evaluate.py, so the exact variant's checkpoint and arch are checked",
    )
    args = parser.parse_args()
    overrides = parse_retriever_opts(args.retriever_opt)
    if overrides and not args.retriever:
        parser.error("--retriever-opt needs --retriever")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    ok = check_common(user_config)
    ready = {}
    for kind in [args.retriever] if args.retriever else RETRIEVER_KINDS:
        print(f"Retriever: {kind}")
        ready[kind] = CHECKS[kind](retriever_settings(kind, config, overrides), user_config)

    print("Summary: " + ", ".join(f"{k} {'ready' if v else 'NOT ready'}" for k, v in ready.items()))
    return 0 if ok and all(ready.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
