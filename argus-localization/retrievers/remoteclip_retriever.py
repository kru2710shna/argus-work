"""Zero-shot baseline: the RemoteCLIP image tower as a Retriever.

RemoteCLIP (arXiv 2306.11029, https://github.com/ChenDelong1999/RemoteCLIP) is
OpenAI CLIP continually pretrained on remote-sensing image-text pairs (sub-metre
aerial/UAV imagery, not 75-300 m/px tiles). Only the image tower is used:
image -> L2-normalized embedding, the same contract as EarthLocRetriever. It has
never seen astronaut photos or this tile database, so its recall is zero-shot,
while EarthLoc was trained on these exact reference tiles.

The released checkpoints (HF chendelong/RemoteCLIP, RemoteCLIP-{RN50,ViT-B-32,
ViT-L-14}.pt) are open_clip-format state dicts. Two backends load them:
- open_clip, when installed (the loader RemoteCLIP's own README uses);
- timm, as a fallback for the ViT variants. timm converts OpenAI-CLIP state
  dicts in vision_transformer.checkpoint_filter_fn, and it is already in
  requirements.txt, so this runs on machines where installing open_clip isn't
  allowed. The two backends agree to float precision on the same weights.
Checkpoints load with torch.load(weights_only=True): plain tensors only, no
pickled code runs.

Neither source says whether RemoteCLIP was trained with QuickGELU (OpenAI
CLIP's activation) or plain GELU (what RemoteCLIP's README builds), so it is a
flag. Activations have no weights, so the same checkpoint loads either way.

Preprocessing squash-resizes to image_size x image_size, keeping the full field
of view like EarthLocRetriever, then applies OpenAI CLIP mean/std. open_clip's
own eval transform (shortest-side resize + centre crop) is resize_mode="crop".
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.interfaces import Retriever

# open_clip constants.py OPENAI_DATASET_MEAN / OPENAI_DATASET_STD
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# open_clip model_configs/<name>.json embed_dim
EMBED_DIMS = {"RN50": 1024, "ViT-B-32": 512, "ViT-L-14": 768}
_TIMM_ARCHS = {"ViT-B-32": "vit_base_patch32_clip", "ViT-L-14": "vit_large_patch14_clip"}
RESIZE_MODES = ("squash", "crop")


def checkpoint_filename(model_name: str) -> str:
    return f"RemoteCLIP-{model_name}.pt"


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state_dict.get(key), dict):
            state_dict = state_dict[key]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_open_clip_visual(
    model_name: str, state_dict: dict, image_size: int, quick_gelu: bool
) -> torch.nn.Module:
    import open_clip
    from open_clip.model import resize_pos_embed

    arch = f"{model_name}-quickgelu" if quick_gelu else model_name
    force_image_size = None if image_size == 224 else image_size
    model = open_clip.create_model(arch, pretrained=None, force_image_size=force_image_size)
    if force_image_size is not None:
        resize_pos_embed(state_dict, model)  # interpolates visual.positional_embedding in place
    model.load_state_dict(state_dict)
    return model.visual


def _build_timm_visual(
    model_name: str, state_dict: dict, image_size: int, quick_gelu: bool
) -> torch.nn.Module:
    import timm
    from timm.models.vision_transformer import checkpoint_filter_fn

    arch = f"{_TIMM_ARCHS[model_name]}{'_quickgelu' if quick_gelu else ''}_224"
    model = timm.create_model(
        arch, pretrained=False, num_classes=EMBED_DIMS[model_name], img_size=image_size
    )
    # Renames visual.* keys to timm's, turns proj into head.weight, and resamples
    # pos_embed when image_size != 224.
    model.load_state_dict(checkpoint_filter_fn(state_dict, model))
    return model


def _open_clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
    except ImportError:
        return False
    return True


class RemoteCLIPRetriever(Retriever):
    descriptor_dim: int

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "ViT-B-32",
        device: str = "cuda",
        image_size: int = 224,
        quick_gelu: bool = False,
        resize_mode: str = "squash",
        backend: str = "auto",
        max_batch: int = 128,
    ):
        if model_name not in EMBED_DIMS:
            raise ValueError(f"model_name must be one of {sorted(EMBED_DIMS)}, got {model_name!r}")
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")
        if model_name == "RN50" and image_size != 224:
            # The attention-pool positional embedding is sized for a 7x7 grid, and
            # open_clip's resize_pos_embed only handles the ViT key.
            raise ValueError("RemoteCLIP RN50 only runs at image_size 224")
        if backend == "auto":
            backend = "open_clip" if _open_clip_available() else "timm"
        if backend == "timm" and model_name not in _TIMM_ARCHS:
            raise ValueError(f"the timm backend supports {sorted(_TIMM_ARCHS)}; RN50 needs open_clip")

        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.backend = backend
        self.max_batch = max_batch
        self.descriptor_dim = EMBED_DIMS[model_name]

        state_dict = _load_state_dict(checkpoint_path)
        build = _build_open_clip_visual if backend == "open_clip" else _build_timm_visual
        self.model = build(model_name, state_dict, image_size, quick_gelu).to(device).eval()

        self._mean = torch.tensor(_OPENAI_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_OPENAI_CLIP_STD, device=device).view(1, 3, 1, 1)

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if len({img.shape for img in images}) > 1 and self.resize_mode == "crop":
            # A common-size squash would defeat the crop, so go one image at a time.
            return np.concatenate([self._embed_chunk([img]) for img in images]).astype(np.float32)
        if len({img.shape for img in images}) > 1:
            # Same fallback as EarthLocRetriever: bring mixed shapes to a common
            # size on CPU so the batch stacks into one tensor.
            import cv2

            images = [
                cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                for img in images
            ]
        chunks = [
            self._embed_chunk(images[start : start + self.max_batch])
            for start in range(0, len(images), self.max_batch)
        ]
        return np.concatenate(chunks).astype(np.float32)

    def _resize(self, batch: torch.Tensor) -> torch.Tensor:
        size = self.image_size
        if self.resize_mode == "squash":
            return F.interpolate(
                batch, size=(size, size), mode="bicubic", align_corners=False, antialias=True
            )
        height, width = batch.shape[-2:]
        scale = size / min(height, width)
        batch = F.interpolate(
            batch,
            size=(max(size, round(height * scale)), max(size, round(width * scale))),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        top = (batch.shape[-2] - size) // 2
        left = (batch.shape[-1] - size) // 2
        return batch[..., top : top + size, left : left + size]

    def _embed_chunk(self, images: list[np.ndarray]) -> np.ndarray:
        batch = torch.from_numpy(np.stack(images)).to(self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (self._resize(batch).clamp(0.0, 1.0) - self._mean) / self._std

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    features = self.model(batch)
            else:
                features = self.model(batch)
        return F.normalize(features.float(), p=2, dim=-1).cpu().numpy()
