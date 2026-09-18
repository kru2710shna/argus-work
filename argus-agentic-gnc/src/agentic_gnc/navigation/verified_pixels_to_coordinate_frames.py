"""Camera and coordinate-frame utilities for navigation measurements.

WHAT THIS FILE DOES
-------------------
This is the first part of the Navigation Measurement Builder.

It converts:
    LightGlue matched image pixel + camera calibration
        -> unit viewing ray in the camera frame
        -> unit viewing ray in the satellite body frame

It also defines the safe handoff for:
    known map/landmark position in ECEF
        -> landmark position in ECI

The final FSW-Payload landmark observation needs:

    [timestamp,
     body_bearing_x, body_bearing_y, body_bearing_z,
     landmark_ECI_x, landmark_ECI_y, landmark_ECI_z]

WHAT THIS FILE DOES NOT DO YET
------------------------------
- It does not run EarthLoc.
- It does not run LightGlue.
- It does not choose or verify map tiles.
- It does not calculate ECEF-to-ECI rotation itself.
  AstroPy/SPICE will provide that timestamp-specific rotation later.
- It does not synchronize gyro/IMU data yet.

FRAME CONVENTIONS
-----------------
Camera frame:
    +X = image right
    +Y = image down
    +Z = camera optical axis / looking direction

Body frame:
    Defined by the spacecraft camera-to-body calibration matrix.

ECEF:
    Earth-fixed Cartesian coordinates, in metres.

ECI:
    Inertial Cartesian coordinates, in metres, using the same convention
    expected by FSW-Payload.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value: object, name: str) -> np.ndarray:
    """Validate and return one finite three-dimensional vector."""
    array = np.asarray(value, dtype=np.float64)

    # Navigation vectors must always be exactly [x, y, z].
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite shape-(3,) vector")

    return array


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    """Normalize a vector so that its length is exactly one."""
    length = np.linalg.norm(vector)

    # A zero-length vector has no physical direction.
    if length <= 0.0:
        raise ValueError(f"{name} must have non-zero length")

    return vector / length


def _rotation(value: object, name: str) -> np.ndarray:
    """Validate a proper 3D rotation matrix.

    A valid rotation matrix must:
    - have shape (3, 3),
    - be orthonormal: R.T @ R = identity,
    - have determinant +1.
    """
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
    """Pinhole camera calibration values, measured in pixels.

    fx_px, fy_px:
        focal lengths in horizontal and vertical pixel units.

    cx_px, cy_px:
        principal point: the image pixel aligned with the optical axis.
    """

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)

        if not all(np.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")

        # Focal length cannot be zero or negative.
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("fx_px and fy_px must be positive")


def pixel_to_camera_bearing(
    pixel_uv: object,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert one image pixel [u, v] into a unit camera-frame ray.

    Pinhole-camera projection is reversed:

        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_camera = normalize([x, y, 1])

    The center pixel [cx, cy] therefore produces [0, 0, 1].
    """
    pixel = np.asarray(pixel_uv, dtype=np.float64)

    # A pixel is exactly [horizontal_u, vertical_v].
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_uv must be a finite shape-(2,) vector")

    u_px, v_px = pixel

    # Convert pixel coordinates into normalized camera coordinates.
    ray_camera = np.array(
        [
            (u_px - intrinsics.cx_px) / intrinsics.fx_px,
            (v_px - intrinsics.cy_px) / intrinsics.fy_px,
            1.0,
        ]
    )

    # FSW expects a direction, so output must have unit length.
    return _unit(ray_camera, "camera ray")


def camera_to_body_bearing(
    bearing_camera: object,
    rotation_camera_to_body: object,
) -> np.ndarray:
    """Rotate a camera-frame ray into the satellite body frame.

    Input:
        bearing_camera: unit ray from pixel_to_camera_bearing()
        rotation_camera_to_body: calibrated 3x3 camera-to-body rotation

    Output:
        unit body-frame ray required by FSW-Payload.
    """
    ray_camera = _unit(_vector(bearing_camera, "bearing_camera"), "bearing_camera")
    rotation = _rotation(rotation_camera_to_body, "rotation_camera_to_body")

    return _unit(rotation @ ray_camera, "body ray")


def ecef_to_eci_position(
    position_ecef_m: object,
    rotation_ecef_to_eci: object,
) -> np.ndarray:
    """Convert a known Earth landmark from ECEF metres to ECI metres.

    Input:
        position_ecef_m:
            known map-tile or landmark position in Earth-fixed coordinates.

        rotation_ecef_to_eci:
            timestamp-specific rotation from AstroPy/SPICE.

    Output:
        landmark position in the ECI frame required by FSW-Payload.

    Important:
        This function deliberately does not approximate Earth rotation.
        We will connect AstroPy/SPICE in the next coordinate-integration layer.
    """
    position = _vector(position_ecef_m, "position_ecef_m")

    # Earth-surface positions should be approximately Earth-radius scale.
    if np.linalg.norm(position) < 6.0e6:
        raise ValueError("position_ecef_m must be an Earth-scale position in metres")

    rotation = _rotation(rotation_ecef_to_eci, "rotation_ecef_to_eci")

    return rotation @ position