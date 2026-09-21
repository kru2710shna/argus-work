"""RemoteCLIP and Qwen3-VL-Embedding retrievers.

The pure helpers always run. The real-weight checks run only when the weights
are on disk; point these at them to enable:
    ARGUS_REMOTECLIP_CKPT=/path/RemoteCLIP-ViT-B-32.pt
    ARGUS_QWEN3VL_EMBEDDING_DIR=/path/Qwen3-VL-Embedding-2B
They run on CPU and never download anything.
"""

import os
import unittest

import numpy as np
import torch

from retrievers.qwen3vl_embedding_retriever import (
    DEFAULT_INSTRUCTION,
    format_instruction,
    pool_last_token,
)

REMOTECLIP_CKPT = os.environ.get("ARGUS_REMOTECLIP_CKPT")
QWEN_DIR = os.environ.get("ARGUS_QWEN3VL_EMBEDDING_DIR")


def sample_images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    # Smooth it so it looks less like noise; include a rotated (non-contiguous) view.
    image = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))
    return [image, np.rot90(image, 1), 255 - image]


class QwenHelpersTest(unittest.TestCase):
    def test_format_instruction_matches_official_helper(self):
        self.assertEqual(format_instruction(None), DEFAULT_INSTRUCTION)
        self.assertEqual(format_instruction("   "), DEFAULT_INSTRUCTION)
        self.assertEqual(format_instruction(" Find the same area "), "Find the same area.")
        self.assertEqual(format_instruction("Which tile matches?"), "Which tile matches?")

    def test_pool_last_token_with_right_padding(self):
        hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        pooled = pool_last_token(hidden, mask)
        torch.testing.assert_close(pooled[0], hidden[0, 3])
        torch.testing.assert_close(pooled[1], hidden[1, 1])


@unittest.skipUnless(REMOTECLIP_CKPT and os.path.exists(REMOTECLIP_CKPT), "set ARGUS_REMOTECLIP_CKPT")
class RemoteCLIPRealWeightsTest(unittest.TestCase):
    def test_contract(self):
        from retrievers.remoteclip_retriever import RemoteCLIPRetriever

        retriever = RemoteCLIPRetriever(REMOTECLIP_CKPT, model_name="ViT-B-32", device="cpu", backend="timm")
        images = sample_images()
        batch = retriever.embed_batch(images)
        self.assertEqual(batch.shape, (3, retriever.descriptor_dim))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(retriever.embed(images[1]), batch[1], atol=1e-5)

    def test_backends_agree(self):
        try:
            import open_clip  # noqa: F401
        except ImportError:
            self.skipTest("open_clip not installed")
        from retrievers.remoteclip_retriever import RemoteCLIPRetriever

        images = sample_images()
        for image_size in (224, 320):
            a = RemoteCLIPRetriever(REMOTECLIP_CKPT, device="cpu", backend="open_clip", image_size=image_size)
            b = RemoteCLIPRetriever(REMOTECLIP_CKPT, device="cpu", backend="timm", image_size=image_size)
            np.testing.assert_allclose(a.embed_batch(images), b.embed_batch(images), atol=1e-5)


@unittest.skipUnless(QWEN_DIR and os.path.isdir(QWEN_DIR), "set ARGUS_QWEN3VL_EMBEDDING_DIR")
class Qwen3VLEmbeddingRealWeightsTest(unittest.TestCase):
    def test_contract_and_mrl(self):
        from retrievers.qwen3vl_embedding_retriever import Qwen3VLEmbeddingRetriever

        retriever = Qwen3VLEmbeddingRetriever(QWEN_DIR, device="cpu", image_size=320, dim=512)
        self.assertIn("Represent the user's input.", retriever._prompt)
        batch = retriever.embed_batch(sample_images())
        self.assertEqual(batch.shape, (3, 512))
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
