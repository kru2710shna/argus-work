"""Tests for real EarthLoc/LightGlue coordinate-record conversion."""

import numpy as np

from agentic_gnc.navigation.earthloc_lightglue_tie_points_to_verified_visual_matches import (
    EarthLocFrameNavigationContext,
    earthloc_latlon_degrees_to_ecef_landmark_m,
    earthloc_lightglue_tie_points_to_verified_visual_matches,
)
from agentic_gnc.navigation.verified_image_pixels_to_body_frame_bearings import (
    CameraIntrinsics,
)


def _context() -> EarthLocFrameNavigationContext:
    return EarthLocFrameNavigationContext(
        frame_index=4,
        time_j2000_s=123.0,
        camera_intrinsics=CameraIntrinsics(
            fx_px=1000.0,
            fy_px=1000.0,
            cx_px=640.0,
            cy_px=480.0,
        ),
        rotation_camera_to_body=np.eye(3),
        rotation_ecef_to_eci=np.eye(3),
    )


def test_latlon_equator_prime_meridian_becomes_wgs84_ecef_x_axis() -> None:
    landmark_ecef_m = earthloc_latlon_degrees_to_ecef_landmark_m(0.0, 0.0)

    np.testing.assert_allclose(
        landmark_ecef_m,
        np.array([6_378_137.0, 0.0, 0.0]),
        atol=1e-6,
    )


def test_real_coordinate_record_becomes_verified_visual_matches() -> None:
    record = {
        "status": "fix",
        "confidence_num_inliers": 40,
        "tie_points": [
            {"u": 640.0, "v": 480.0, "lat": 0.0, "lon": 0.0},
            {"u": 650.0, "v": 480.0, "lat": 0.0, "lon": 90.0},
        ],
    }

    matches = earthloc_lightglue_tie_points_to_verified_visual_matches(
        record,
        _context(),
    )

    assert len(matches) == 2
    assert matches[0].frame_id == 4
    assert matches[0].confidence == 0.8
    np.testing.assert_allclose(matches[0].query_pixel_uv, [640.0, 480.0])
    np.testing.assert_allclose(matches[0].landmark_ecef_m, [6_378_137.0, 0.0, 0.0])
    np.testing.assert_allclose(matches[1].landmark_ecef_m, [0.0, 6_378_137.0, 0.0], atol=1e-6)


def test_no_fix_record_produces_no_navigation_measurements() -> None:
    matches = earthloc_lightglue_tie_points_to_verified_visual_matches(
        {
            "status": "no_fix",
            "confidence_num_inliers": 0,
            "tie_points": [],
        },
        _context(),
    )

    assert matches == []