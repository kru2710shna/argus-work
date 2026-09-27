"""Convert verified image pixels into satellite body-frame bearings.

WHAT THIS FILE DOES
-------------------
This is the pixel-and-camera half of the Navigation Measurement Builder.

It converts:

    LightGlue matched image pixel + camera calibration
        -> unit viewing ray in the camera frame
        -> unit viewing ray in the satellite body frame

This body-frame ray is later combined with an ECI landmark position by:

    verified_visual_matches_to_fsw_landmark_observations.py

FRAME CONVENTIONS
-----------------
Camera frame:
    +X = image right
    +Y = image down
    +Z = camera optical axis / looking direction

Body frame:
    Defined by the calibrated camera-to-body rotation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value: object, name: str) -> np.ndarray:
    """Validate and return one finite 3D vector."""

    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite shape-(3,) vector")

    return vector


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    """Normalize a non-zero vector to unit length."""

    length = np.linalg.norm(vector)

    if length <= 0.0:
        raise ValueError(f"{name} must have non-zero length")

    return vector / length


def _rotation(value: object, name: str) -> np.ndarray:
    """Validate a proper 3D rotation matrix."""

    matrix = np.asarray(value, dtype=np.float64)

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite shape-(3, 3) matrix")

    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")

    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must have determinant +1")

    return matrix


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole-camera intrinsic calibration, measured in pixels."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)

        if not all(np.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")

        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("fx_px and fy_px must be positive")


def pixel_to_camera_bearing(
    pixel_uv: object,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert image pixel ``[u, v]`` to a unit camera-frame viewing ray.

    The center pixel ``[cx, cy]`` produces camera bearing ``[0, 0, 1]``.
    """

    pixel = np.asarray(pixel_uv, dtype=np.float64)

    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_uv must be a finite shape-(2,) vector")

    u_px, v_px = pixel

    ray_camera = np.array(
        [
            (u_px - intrinsics.cx_px) / intrinsics.fx_px,
            (v_px - intrinsics.cy_px) / intrinsics.fy_px,
            1.0,
        ],
        dtype=np.float64,
    )

    return _unit(ray_camera, "camera ray")


def camera_to_body_bearing(
    bearing_camera: object,
    rotation_camera_to_body: object,
) -> np.ndarray:
    """Rotate a unit camera-frame ray into the satellite body frame."""

    ray_camera = _unit(
        _vector(bearing_camera, "bearing_camera"),
        "bearing_camera",
    )
    rotation = _rotation(
        rotation_camera_to_body,
        "rotation_camera_to_body",
    )

    return _unit(rotation @ ray_camera, "body ray")