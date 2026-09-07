"""Closed-loop capture/skip environment for agentic GNC experiments."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from agentic_gnc.contracts import ODBatch
from agentic_gnc.fsw_od_adapter import solve_with_fsw_python
from agentic_gnc.simulator import circular_orbit_truth, generate_synthetic_batch


class SensingAction(IntEnum):
    SKIP = 0
    CAPTURE = 1


@dataclass(frozen=True)
class EnvironmentConfig:
    steps: int = 8
    dt_s: float = 10.0
    landmarks_per_frame: int = 4
    bearing_noise_rad: float = 0.0
    image_cost: float = 0.01


class AgenticGNCEnvironment:
    """Simulator where an agent decides whether to capture each camera frame."""

    def __init__(self, config: EnvironmentConfig = EnvironmentConfig()) -> None:
        self.config = config
        self._truth = None
        self._full_batch = None
        self._selected_landmarks = []
        self._selected_frame_ids = set()
        self._frame = 0
        self._images_taken = 0
        self._last_result = None

    def reset(self, seed: int = 0) -> dict:
        """Start a new synthetic orbit scenario."""
        self._truth = circular_orbit_truth(
            steps=self.config.steps,
            dt_s=self.config.dt_s,
        )
        self._full_batch = generate_synthetic_batch(
            self._truth,
            landmarks_per_frame=self.config.landmarks_per_frame,
            bearing_noise_rad=self.config.bearing_noise_rad,
            seed=seed,
        )
        self._selected_landmarks = []
        self._selected_frame_ids = set()
        self._frame = 0
        self._images_taken = 0
        self._last_result = None
        return self._observation()

    def step(self, action: int | SensingAction) -> tuple[dict, float, bool, dict]:
        """Apply one capture/skip decision and advance the simulation."""
        if self._truth is None or self._full_batch is None:
            raise RuntimeError("Call reset() before step()")
        if self._frame >= self.config.steps:
            raise RuntimeError("Episode is complete; call reset()")

        action = SensingAction(action)
        current_frame = self._frame

        if action == SensingAction.CAPTURE:
            frame_landmarks = [
                item
                for item in self._full_batch.landmarks
                if item.frame_id == current_frame
            ]
            self._selected_landmarks.extend(frame_landmarks)
            self._selected_frame_ids.add(current_frame)
            self._images_taken += 1

        self._frame += 1
        solver_ran = False
        position_error_m = None

        # Need at least two observed frames: one image alone cannot determine
        # an orbit trajectory/velocity.
        if (
            action == SensingAction.CAPTURE
            and len(self._selected_landmarks) >= 4
            and len(self._selected_frame_ids) >= 3
        ):
            partial_batch = ODBatch(
                landmarks=tuple(self._selected_landmarks),
                gyros=tuple(
                    gyro
                    for frame_id, gyro in enumerate(self._full_batch.gyros[: self._frame])
                    if frame_id in self._selected_frame_ids
                ),
            )
            self._last_result = solve_with_fsw_python(partial_batch)
            solver_ran = True

            estimated_position_m = (
                np.asarray(self._last_result["positions"])[-1] * 1_000.0
            )
            true_position_m = self._truth.positions_eci_m[self._frame - 1]
            position_error_m = float(
                np.linalg.norm(estimated_position_m - true_position_m)
            )

        # Training reward may use simulator truth. The observation returned to
        # the agent below intentionally does not expose true error.
        if position_error_m is None:
            reward = -0.10
        else:
            reward = -(position_error_m / 1_000.0)

        if action == SensingAction.CAPTURE:
            reward -= self.config.image_cost

        done = self._frame >= self.config.steps
        info = {
            "action": action.name,
            "solver_ran": solver_ran,
            "images_taken": self._images_taken,
            "position_error_m": position_error_m,
        }
        return self._observation(), reward, done, info

    def _observation(self) -> dict:
        """Only information that an onboard policy may observe."""
        estimated_position_km = np.zeros(3)
        has_estimate = self._last_result is not None

        if has_estimate:
            estimated_position_km = np.asarray(self._last_result["positions"])[-1]

        return {
            "time_fraction": self._frame / self.config.steps,
            "images_taken": self._images_taken,
            "observed_frames": len(self._selected_frame_ids),
            "has_estimate": has_estimate,
            "estimated_position_km": estimated_position_km,
        }
