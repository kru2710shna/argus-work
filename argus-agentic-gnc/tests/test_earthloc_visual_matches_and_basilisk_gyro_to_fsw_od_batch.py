"""Tests for EarthLoc visual plus gyro FSW batch assembly."""

import numpy as np
import pytest

from agentic_gnc.navigation.earthloc_lightglue_tie_points_to_verified_visual_matches import (
    EarthLocFrameNavigationContext,
)
from agentic_gnc.navigation.earthloc_visual_matches_and_basilisk_gyro_to_fsw_od_batch import (
    earthloc_visual_matches_and_basilisk_gyro_to_fsw_od_batch,
)
from agentic_gnc.navigation.navigation_measurement_contracts import GyroObservation
from agentic_gnc.navigation.verified_image_pixels_to_body_frame_bearings import (
    CameraIntrinsics,
)


def _context(frame_index: int, time_j2000_s: float) -> EarthLocFrameNavigationContext:
    return EarthLocFrameNavigationContext(
        frame_index=frame_index,
        time_j2000_s=time_j2000_s,
        camera_intrinsics=CameraIntrinsics(
            fx_px=1000.0,
            fy_px=1000.0,
            cx_px=640.0,
            cy_px=480.0,
        ),
        rotation_camera_to_body=np.eye(3),
        rotation_ecef_to_eci=np.eye(3),
    )


def _fix_record(longitude_offset_deg: float) -> dict:
    return {
        "status": "fix",
        "confidence_num_inliers": 40,
        "tie_points": [
            {"u": 640.0, "v": 480.0, "lat": 0.0, "lon": longitude_offset_deg},
            {"u": 650.0, "v": 480.0, "lat": 1.0, "lon": longitude_offset_deg},
        ],
    }


def test_visual_matches_and_gyros_become_one_fsw_batch() -> None:
    batch = earthloc_visual_matches_and_basilisk_gyro_to_fsw_od_batch(
        earthloc_coordinate_records=[
            _fix_record(0.0),
            _fix_record(10.0),
        ],
        frame_navigation_contexts=[
            _context(frame_index=0, time_j2000_s=100.0),
            _context(frame_index=1, time_j2000_s=110.0),
        ],
        gyro_observations=[
            GyroObservation(110.0, np.array([0.01, 0.02, 0.03])),
            GyroObservation(100.0, np.array([0.01, 0.02, 0.03])),
        ],
    )

    assert len(batch.landmarks) == 4
    assert len(batch.gyros) == 2

    landmark_rows, group_starts, gyro_rows, sigmas = batch.to_fsw_arrays()

    assert landmark_rows.shape == (4, 7)
    assert group_starts.tolist() == [True, False, True, False]
    assert gyro_rows.shape == (2, 4)
    assert gyro_rows[:, 0].tolist() == [100.0, 110.0]
    assert sigmas.shape == (4,)


def test_record_and_context_counts_must_match() -> None:
    with pytest.raises(ValueError, match="same length"):
        earthloc_visual_matches_and_basilisk_gyro_to_fsw_od_batch(
            earthloc_coordinate_records=[_fix_record(0.0)],
            frame_navigation_contexts=[],
            gyro_observations=[
                GyroObservation(100.0, np.array([0.01, 0.02, 0.03])),
            ],
        )