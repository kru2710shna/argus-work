import numpy as np

from agentic_gnc.navigation.verified_pixels_to_coordinate_frames.py import (
    CameraIntrinsics,
    camera_to_body_bearing,
    ecef_to_eci_position,
    pixel_to_camera_bearing,
)


def test_center_pixel_points_along_camera_z() -> None:
    camera = CameraIntrinsics(1000.0, 1000.0, 640.0, 480.0)

    bearing = pixel_to_camera_bearing([640.0, 480.0], camera)

    assert np.allclose(bearing, [0.0, 0.0, 1.0])


def test_right_pixel_has_positive_camera_x() -> None:
    camera = CameraIntrinsics(100.0, 100.0, 0.0, 0.0)

    bearing = pixel_to_camera_bearing([100.0, 0.0], camera)

    assert np.allclose(bearing, [1.0, 0.0, 1.0] / np.sqrt(2.0))


def test_identity_camera_to_body_rotation() -> None:
    bearing = camera_to_body_bearing([0.0, 0.0, 1.0], np.eye(3))

    assert np.allclose(bearing, [0.0, 0.0, 1.0])


def test_identity_ecef_to_eci_rotation() -> None:
    earth_point_ecef_m = np.array([6_378_137.0, 0.0, 0.0])

    earth_point_eci_m = ecef_to_eci_position(earth_point_ecef_m, np.eye(3))

    assert np.allclose(earth_point_eci_m, earth_point_ecef_m)