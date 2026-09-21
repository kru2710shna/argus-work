"""Shared VINSat-to-FSW measurement contract."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _vec3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite shape-(3,) vector")
    return vector


@dataclass(frozen=True)
class LandmarkObservation:
    """Visual landmark: time is J2000 seconds; ECI position is metres."""

    time_j2000_s: float
    frame_id: int
    bearing_body: np.ndarray
    landmark_eci_m: np.ndarray
    confidence: float
    bearing_sigma_rad: float = 0.009

    def __post_init__(self) -> None:
        bearing = _vec3(self.bearing_body, "bearing_body")
        landmark = _vec3(self.landmark_eci_m, "landmark_eci_m")

        if not np.isfinite(self.time_j2000_s) or self.frame_id < 0:
            raise ValueError("timestamp must be finite and frame_id non-negative")
        if not np.isclose(np.linalg.norm(bearing), 1.0, atol=1e-6):
            raise ValueError("bearing_body must have unit length")
        if np.linalg.norm(landmark) < 6.0e6:
            raise ValueError("landmark_eci_m must use Earth-scale metres")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        object.__setattr__(self, "bearing_body", bearing)
        object.__setattr__(self, "landmark_eci_m", landmark)


@dataclass(frozen=True)
class GyroObservation:
    """Body-frame angular velocity in rad/s."""

    time_j2000_s: float
    angular_velocity_body_rad_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "angular_velocity_body_rad_s",
            _vec3(self.angular_velocity_body_rad_s, "angular_velocity_body_rad_s"),
        )


@dataclass(frozen=True)
class ODBatch:
    """Time-ordered observations exported to FSW-Payload OD."""

    landmarks: tuple[LandmarkObservation, ...]
    gyros: tuple[GyroObservation, ...]

    def to_fsw_arrays(
        self, landmark_position_unit: Literal["m", "km"] = "m"
    ):
        """Return FSW arrays: landmarks, group starts, gyros, sigmas."""
        if len(self.landmarks) < 4:
            raise ValueError("FSW OD requires at least four landmarks")
        if not self.gyros:
            raise ValueError("FSW OD requires gyro measurements")

        scale = 1.0 if landmark_position_unit == "m" else 1e-3

        landmark_rows = np.array(
            [
                [
                    x.time_j2000_s,
                    *x.bearing_body,
                    *(x.landmark_eci_m * scale),
                ]
                for x in self.landmarks
            ],
            dtype=np.float64,
        )

        group_starts = np.zeros(len(self.landmarks), dtype=bool)
        previous_frame = None
        for index, observation in enumerate(self.landmarks):
            if observation.frame_id != previous_frame:
                group_starts[index] = True
                previous_frame = observation.frame_id

        gyro_rows = np.array(
            [[x.time_j2000_s, *x.angular_velocity_body_rad_s] for x in self.gyros],
            dtype=np.float64,
        )
        sigmas = np.array([x.bearing_sigma_rad for x in self.landmarks])

        return landmark_rows, group_starts, gyro_rows, sigmas