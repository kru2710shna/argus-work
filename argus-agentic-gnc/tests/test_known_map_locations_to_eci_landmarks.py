import numpy as np

from agentic_gnc.navigation.known_map_locations_to_eci_landmarks import (
    known_map_location_ecef_to_eci_landmark,
)
from agentic_gnc.navigation.verified_image_pixels_to_body_frame_bearings import (
    CameraIntrinsics,
    camera_to_body_bearing,
    pixel_to_camera_bearing,
)


def test_identity_rotation_keeps_landmark_position() -> None:
    earth_point_ecef_m = np.array([6_378_137.0, 0.0, 0.0])

    earth_point_eci_m = known_map_location_ecef_to_eci_landmark(
        earth_point_ecef_m,
        np.eye(3),
    )

    assert np.allclose(earth_point_eci_m, earth_point_ecef_m)


def test_rotation_changes_landmark_frame() -> None:
    earth_point_ecef_m = np.array([6_378_137.0, 0.0, 0.0])

    rotation_ecef_to_eci = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    earth_point_eci_m = known_map_location_ecef_to_eci_landmark(
        earth_point_ecef_m,
        rotation_ecef_to_eci,
    )

    assert np.allclose(earth_point_eci_m, [0.0, 6_378_137.0, 0.0])