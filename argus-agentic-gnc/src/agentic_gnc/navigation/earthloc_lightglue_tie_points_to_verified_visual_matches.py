"""Convert real EarthLoc/LightGlue coordinate records into verified visual matches.

WHAT THIS FILE DOES
-------------------
argus-localization exports one JSON record per camera frame:

    frame status + LightGlue/RANSAC inlier count +
    tie points: image pixel (u, v) <-> Earth latitude/longitude

This module converts every verified tie point into a VerifiedVisualMatch.
The next navigation module converts each VerifiedVisualMatch into an
FSW-ready LandmarkObservation.

Important:
- EarthLoc/LightGlue provides real image-to-Earth correspondences.
- Latitude/longitude are converted to a WGS-84 Earth-fixed (ECEF) point.
- The ECEF-to-ECI rotation is supplied by the caller because it depends on
  the observation timestamp and will later come from AstroPy/SPICE.
- The inlier-count confidence is only a temporary normalized quality weight;
  it is not yet a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from agentic_gnc.navigation.verified_image_pixels_to_body_frame_bearings import (
    CameraIntrinsics,
)
from agentic_gnc.navigation.verified_visual_matches_to_fsw_landmark_observations import (
    VerifiedVisualMatch,
)

_WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
_WGS84_FLATTENING = 1.0 / 298.257_223_563


@dataclass(frozen=True)
class EarthLocFrameNavigationContext:
    """Known navigation metadata associated with one exported image frame."""

    frame_index: int
    time_j2000_s: float
    camera_intrinsics: CameraIntrinsics
    rotation_camera_to_body: np.ndarray
    rotation_ecef_to_eci: np.ndarray
    bearing_sigma_rad: float = 0.009


def earthloc_latlon_degrees_to_ecef_landmark_m(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0,
) -> np.ndarray:
    """Convert a WGS-84 geodetic Earth location to ECEF metres."""

    latitude_deg = _finite_float(latitude_deg, "latitude_deg")
    longitude_deg = _finite_float(longitude_deg, "longitude_deg")
    altitude_m = _finite_float(altitude_m, "altitude_m")

    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude_deg must be in [-90, 90]")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude_deg must be in [-180, 180]")

    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)

    eccentricity_squared = _WGS84_FLATTENING * (2.0 - _WGS84_FLATTENING)
    sin_latitude = np.sin(latitude_rad)
    prime_vertical_radius_m = _WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - eccentricity_squared * sin_latitude**2
    )

    return np.array(
        [
            (prime_vertical_radius_m + altitude_m)
            * np.cos(latitude_rad)
            * np.cos(longitude_rad),
            (prime_vertical_radius_m + altitude_m)
            * np.cos(latitude_rad)
            * np.sin(longitude_rad),
            (
                prime_vertical_radius_m * (1.0 - eccentricity_squared)
                + altitude_m
            )
            * sin_latitude,
        ],
        dtype=np.float64,
    )


def earthloc_lightglue_tie_points_to_verified_visual_matches(
    record: Mapping[str, Any],
    context: EarthLocFrameNavigationContext,
    *,
    inliers_for_full_confidence: float = 50.0,
) -> list[VerifiedVisualMatch]:
    """Convert one exported EarthLoc/LightGlue ``fix`` record into matches.

    A ``no_fix`` record intentionally produces no navigation measurements.
    """

    if context.frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    status = record.get("status")
    if status == "no_fix":
        return []
    if status != "fix":
        raise ValueError(f"record status must be 'fix' or 'no_fix', got {status!r}")

    inlier_count = _finite_float(
        record.get("confidence_num_inliers"),
        "confidence_num_inliers",
    )
    if inlier_count <= 0.0:
        raise ValueError("a fix record must contain a positive inlier count")

    inliers_for_full_confidence = _finite_float(
        inliers_for_full_confidence,
        "inliers_for_full_confidence",
    )
    if inliers_for_full_confidence <= 0.0:
        raise ValueError("inliers_for_full_confidence must be positive")

    tie_points = record.get("tie_points")
    if not isinstance(tie_points, list) or not tie_points:
        raise ValueError("a fix record must contain at least one tie point")

    # Temporary, documented quality weight. Replace with calibrated uncertainty
    # after we evaluate real matching failures across representative episodes.
    confidence = min(1.0, inlier_count / inliers_for_full_confidence)

    matches: list[VerifiedVisualMatch] = []
    for tie_point in tie_points:
        if not isinstance(tie_point, Mapping):
            raise ValueError("each tie point must be a mapping")

        query_pixel_uv = np.array(
            [
                _finite_float(tie_point.get("u"), "tie_point.u"),
                _finite_float(tie_point.get("v"), "tie_point.v"),
            ],
            dtype=np.float64,
        )
        landmark_ecef_m = earthloc_latlon_degrees_to_ecef_landmark_m(
            _finite_float(tie_point.get("lat"), "tie_point.lat"),
            _finite_float(tie_point.get("lon"), "tie_point.lon"),
        )

        matches.append(
            VerifiedVisualMatch(
                time_j2000_s=context.time_j2000_s,
                frame_id=context.frame_index,
                query_pixel_uv=query_pixel_uv,
                camera_intrinsics=context.camera_intrinsics,
                landmark_ecef_m=landmark_ecef_m,
                rotation_camera_to_body=context.rotation_camera_to_body,
                rotation_ecef_to_eci=context.rotation_ecef_to_eci,
                confidence=confidence,
                bearing_sigma_rad=context.bearing_sigma_rad,
            )
        )

    return matches


def _finite_float(value: Any, name: str) -> float:
    """Return a finite float or fail with an input-specific error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error

    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result