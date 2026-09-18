"""Build FSW-ready landmark observations from verified visual matches.

WHAT THIS FILE DOES
-------------------
This is the central conversion step in the Navigation Measurement Builder.

It takes one verified visual match containing:
    - the matched query-image pixel from LightGlue,
    - the known Earth landmark/map position,
    - camera calibration,
    - camera-to-body orientation,
    - ECEF-to-ECI orientation at the image timestamp,

and produces:
    LandmarkObservation(
        time,
        frame_id,
        body-frame unit bearing,
        ECI landmark position,
        confidence,
        bearing uncertainty,
    )

That LandmarkObservation is the exact shared object exported to FSW-Payload.

IMPORTANT
---------
EarthLoc and LightGlue are not connected here yet. This file defines the
clean boundary they will call once they provide verified pixels and a
georeferenced Earth landmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_gnc.navigation.contracts import LandmarkObservation
from agentic_gnc.navigation.coordinate_frames import (
    CameraIntrinsics,
    camera_to_body_bearing,
    ecef_to_eci_position,
    pixel_to_camera_bearing,
)


@dataclass(frozen=True)
class VerifiedVisualMatch:
    """One image-to-Earth match accepted by geometric verification.

    query_pixel_uv:
        Matched [u, v] pixel in the satellite camera image.

    landmark_ecef_m:
        Known Earth location of the matched map feature, in ECEF metres.

    rotation_camera_to_body:
        3x3 calibrated rotation from camera coordinates to satellite body
        coordinates.

    rotation_ecef_to_eci:
        3x3 timestamp-specific Earth-fixed to inertial rotation, supplied
        later by AstroPy/SPICE.

    confidence:
        Confidence assigned after retrieval and geometric verification.

    bearing_sigma_rad:
        Estimated angular uncertainty of this viewing ray.
    """

    time_j2000_s: float
    frame_id: int
    query_pixel_uv: np.ndarray
    landmark_ecef_m: np.ndarray
    rotation_camera_to_body: np.ndarray
    rotation_ecef_to_eci: np.ndarray
    confidence: float
    bearing_sigma_rad: float = 0.009


def build_landmark_observation(match: VerifiedVisualMatch) -> LandmarkObservation:
    """Convert one verified visual match into an FSW-ready observation."""

    if not np.isfinite(match.time_j2000_s):
        raise ValueError("time_j2000_s must be finite")

    if match.frame_id < 0:
        raise ValueError("frame_id must be non-negative")

    if not 0.0 <= match.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    if not np.isfinite(match.bearing_sigma_rad) or match.bearing_sigma_rad <= 0.0:
        raise ValueError("bearing_sigma_rad must be finite and positive")

    # LightGlue pixel -> camera-frame unit viewing ray.
    bearing_camera = pixel_to_camera_bearing(
        match.query_pixel_uv,
        match.camera_intrinsics,
    )

    # Camera ray -> body-frame ray required by FSW-Payload.
    bearing_body = camera_to_body_bearing(
        bearing_camera,
        match.rotation_camera_to_body,
    )

    # Known map-feature position -> inertial ECI coordinates required by FSW.
    landmark_eci_m = ecef_to_eci_position(
        match.landmark_ecef_m,
        match.rotation_ecef_to_eci,
    )

    return LandmarkObservation(
        time_j2000_s=match.time_j2000_s,
        frame_id=match.frame_id,
        bearing_body=bearing_body,
        landmark_eci_m=landmark_eci_m,
        confidence=match.confidence,
        bearing_sigma_rad=match.bearing_sigma_rad,
    )