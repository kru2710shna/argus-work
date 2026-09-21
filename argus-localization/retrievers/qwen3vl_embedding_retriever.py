"""Zero-shot baseline: Qwen3-VL-Embedding as a Retriever.

Qwen3-VL-Embedding (arXiv 2601.04720, HF Qwen/Qwen3-VL-Embedding-{2B,8B}) is a
Qwen3-VL model trained for multimodal embeddings. Each input is wrapped in a
chat template (system message = task instruction, user message = the image),
and the embedding is the last token's final hidden state, L2-normalized. It has
never seen astronaut photos or this tile database, so its recall is zero-shot.
Nothing in its reported training data or benchmarks is remote sensing.

This reimplements the official helper (scripts/qwen3_vl_embedding.py in the HF
model repo) for image-only inputs on top of transformers' own Qwen3-VL classes,
so no trust_remote_code or qwen-vl-utils is needed. The pieces it has to match:
- instruction: default "Represent the user's input."; a custom one gets a "."
  appended when it doesn't already end in punctuation;
- resize: qwen_vl_utils.smart_resize with factor 32 (patch 16 x spatial merge
  2), a PIL resize, then the processor with do_resize=False;
- padding on the right, pooling at the last attention-mask position.

Images are squash-resized to image_size x image_size first, like
EarthLocRetriever, so every image becomes (image_size / 32)^2 visual tokens:
100 at 320 px.

dim < hidden size is Matryoshka (MRL) truncation: keep the leading dim
components, then re-normalize. The model card supports 64 up to the hidden size.
"""

import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.interfaces import Retriever

DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGE_FACTOR = 32  # patch size 16 x spatial merge 2


def format_instruction(instruction: str | None) -> str:
    """Mirrors Qwen3VLEmbedder.format_model_input's instruction handling."""
    instruction = (instruction or "").strip()
    if not instruction:
        return DEFAULT_INSTRUCTION
    if not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def pool_last_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state at each row's last attention-mask position (right padding)."""
    last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[rows, last]


def _load_model(model_path: str, dtype: torch.dtype, attn_implementation: str) -> torch.nn.Module:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        # Same shape as the official helper's class, so checkpoint keys
        # (model.visual.*, model.language_model.*) load as-is. The empty mapping
        # stops transformers applying legacy Qwen-VL key renames.
        _checkpoint_conversion_mapping: dict = {}

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

    wrapper = Qwen3VLForEmbedding.from_pretrained(
        model_path, dtype=dtype, attn_implementation=attn_implementation, local_files_only=True
    )
    return wrapper.model


class Qwen3VLEmbeddingRetriever(Retriever):
    descriptor_dim: int

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        image_size: int = 320,
        dim: int | None = None,
        instruction: str | None = None,
        batch_size: int = 32,
        attn_implementation: str = "sdpa",
    ):
        if image_size % IMAGE_FACTOR:
            raise ValueError(f"image_size must be a multiple of {IMAGE_FACTOR}, got {image_size}")
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = _load_model(model_path, dtype, attn_implementation).to(device).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path, padding_side="right", local_files_only=True
        )

        hidden_size = self.model.config.text_config.hidden_size
        if dim is not None and not 64 <= dim <= hidden_size:
            raise ValueError(f"dim must be in [64, {hidden_size}], got {dim}")
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.instruction = format_instruction(instruction)
        self.batch_size = batch_size
        self.descriptor_dim = dim or hidden_size

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {"role": "user", "content": [{"type": "image"}]},
        ]
        # Same instruction for every image, so the templated prompt is built once.
        self._prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        chunks = [
            self._embed_chunk(images[start : start + self.batch_size])
            for start in range(0, len(images), self.batch_size)
        ]
        return np.concatenate(chunks).astype(np.float32)

    def _embed_chunk(self, images: list[np.ndarray]) -> np.ndarray:
        size = (self.image_size, self.image_size)
        # smart_resize leaves a square multiple of 32 unchanged, so this one
        # resize is the whole of what qwen_vl_utils.fetch_image would do.
        pil_images = [
            Image.fromarray(np.ascontiguousarray(img)).convert("RGB").resize(size, Image.BICUBIC)
            for img in images
        ]
        inputs = self.processor(
            text=[self._prompt] * len(pil_images),
            images=pil_images,
            padding=True,
            do_resize=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        embeddings = pool_last_token(hidden, inputs["attention_mask"])[:, : self.descriptor_dim]
        return F.normalize(embeddings.float(), p=2, dim=-1).cpu().numpy()
