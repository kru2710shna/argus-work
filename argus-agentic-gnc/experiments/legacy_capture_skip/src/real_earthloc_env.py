"""Real EarthLoc capture/skip environment for the Agentic GNC project.

The action is deliberately small: CAPTURE runs EarthLoc retrieval on the next
real ISS query image; SKIP avoids image/compute cost. Labels from the EarthLoc
intersection file are evaluation-only and are not returned in observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
import sys

import faiss
import numpy as np
from PIL import Image


class EarthLocAction(IntEnum):
    SKIP = 0
    CAPTURE = 1


@dataclass(frozen=True)
class RealEarthLocConfig:
    data_root: Path = Path("/Users/krushna/argus-earthloc-data")
    checkpoint_path: Path = Path("/Users/krushna/argus-earthloc-data/best_trained_model.pt")
    manifest_path: Path = Path("argus-agentic-gnc/configs/earthloc_real_episode.json")
    device: str = "cpu"
    top_k: int = 5
    image_cost: float = 0.08


class RealEarthLocCaptureEnv:
    """An episodic real-image retrieval environment with capture/skip actions."""

    def __init__(self, config: RealEarthLocConfig = RealEarthLocConfig()) -> None:
        self.config = config
        self._manifest = json.loads(config.manifest_path.read_text())
        self._queries = [Path(path) for path in self._manifest["queries"]]
        self._database_tiles = [
            Path(path)
            for path in [*self._manifest["database_tiles"], *self._manifest["distractor_tiles"]]
        ]
        self._model = None
        self._index = None
        self._frame = 0
        self._images_taken = 0
        self._frames_since_capture = 99
        self._last_similarity = 0.0
        self._last_top_paths: list[str] = []

    def reset(self) -> dict:
        """Start the fixed, real 12-frame ISS episode."""
        self._ensure_index()
        self._frame = 0
        self._images_taken = 0
        self._frames_since_capture = 99
        self._last_similarity = 0.0
        self._last_top_paths = []
        return self._observation()

    def step(self, action: int | EarthLocAction) -> tuple[dict, float, bool, dict]:
        if self._index is None:
            raise RuntimeError("Call reset() before step()")
        if self._frame >= len(self._queries):
            raise RuntimeError("Episode is complete; call reset()")

        action = EarthLocAction(action)
        query_relative_path = self._queries[self._frame]
        query_name = query_relative_path.name
        top_paths: list[str] = []
        top_scores: list[float] = []
        positive_rank: int | None = None

        if action == EarthLocAction.CAPTURE:
            descriptor = self._embed(self.config.data_root / query_relative_path)
            scores, indices = self._index.search(descriptor[None, :], self.config.top_k)
            top_paths = [str(self._database_tiles[index]) for index in indices[0]]
            top_scores = [float(score) for score in scores[0]]

            known_positive = set(self._manifest["positive_tiles_by_query"][query_name])
            positive_rank = next(
                (rank for rank, path in enumerate(top_paths, start=1) if path in known_positive),
                None,
            )
            self._images_taken += 1
            self._frames_since_capture = 0
            self._last_similarity = top_scores[0]
            self._last_top_paths = top_paths

            # Ground-truth labels shape only the training/evaluation reward.
            if positive_rank == 1:
                reward = 1.0
            elif positive_rank is not None:
                reward = 0.55
            else:
                reward = -0.20
            reward -= self.config.image_cost
        else:
            reward = -0.03
            self._frames_since_capture += 1

        self._frame += 1
        done = self._frame == len(self._queries)
        info = {
            "query_path": str(query_relative_path),
            "action": action.name,
            "images_taken": self._images_taken,
            "top_paths": top_paths,
            "top_scores": top_scores,
            "positive_rank": positive_rank,
        }
        return self._observation(), reward, done, info

    def _observation(self) -> dict:
        """Values a policy may use without access to ground-truth labels."""
        return {
            "frame": self._frame,
            "frames_remaining": len(self._queries) - self._frame,
            "images_taken": self._images_taken,
            "frames_since_capture": self._frames_since_capture,
            "last_top_similarity": self._last_similarity,
            "has_previous_retrieval": bool(self._last_top_paths),
        }

    def _ensure_index(self) -> None:
        if self._index is not None:
            return

        workspace_root = Path(__file__).resolve().parents[3]
        localization_root = workspace_root / "argus-localization"
        if str(localization_root) not in sys.path:
            sys.path.insert(0, str(localization_root))
        from retrievers.earthloc_retriever import EarthLocRetriever

        missing = [path for path in [*self._queries, *self._database_tiles] if not (self.config.data_root / path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing EarthLoc episode file: {missing[0]}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing EarthLoc checkpoint: {self.config.checkpoint_path}")

        print(f"Loading EarthLoc on {self.config.device} and indexing {len(self._database_tiles)} real tiles...")
        self._model = EarthLocRetriever(str(self.config.checkpoint_path), device=self.config.device)
        descriptors = np.stack([self._embed(self.config.data_root / path) for path in self._database_tiles])
        self._index = faiss.IndexFlatIP(descriptors.shape[1])
        self._index.add(descriptors.astype(np.float32))

    def _embed(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
        return np.asarray(self._model.embed(pixels), dtype=np.float32)
