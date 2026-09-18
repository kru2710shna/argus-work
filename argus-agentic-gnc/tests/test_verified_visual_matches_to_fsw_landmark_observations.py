import numpy as np

from agentic_gnc.navigation.verified_image_pixels_to_body_frame_bearings import CameraIntrinsics
from agentic_gnc.navigation.verified_visual_matches_to_fsw_landmark_observations import (
    VerifiedVisualMatch,
    build_landmark_observation,
)


def test_verified_match_becomes_fsw_landmark_observation() -> None:
    match = VerifiedVisualMatch(
        time_j2000_s=123.0,
        frame_id=7,
        query_pixel_uv=np.array([640.0, 480.0]),
        camera_intrinsics=CameraIntrinsics(
            fx_px=1000.0,
            fy_px=1000.0,
            cx_px=640.0,
            cy_px=480.0,
        ),
        landmark_ecef_m=np.array([6_378_137.0, 0.0, 0.0]),
        rotation_camera_to_body=np.eye(3),
        rotation_ecef_to_eci=np.eye(3),
        confidence=0.95,
        bearing_sigma_rad=0.01,
    )

    observation = build_landmark_observation(match)

    assert observation.time_j2000_s == 123.0
    assert observation.frame_id == 7
    assert np.allclose(observation.bearing_body, [0.0, 0.0, 1.0])
    assert np.allclose(observation.landmark_eci_m, [6_378_137.0, 0.0, 0.0])
    assert observation.confidence == 0.95
    assert observation.bearing_sigma_rad == 0.01