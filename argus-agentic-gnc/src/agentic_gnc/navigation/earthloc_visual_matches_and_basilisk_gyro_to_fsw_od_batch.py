"""Assemble visual matches and gyro observations into one FSW OD batch.

WHAT THIS FILE DOES
-------------------
This is the final assembly step before FSW-Payload orbit determination.

    EarthLoc/LightGlue coordinate records
    -> verified visual matches
    -> FSW landmark observations

    Basilisk gyro output
    -> GyroObservation objects

    landmarks + gyros
    -> one time-ordered ODBatch for FSW-Payload

This file does not parse raw Basilisk messages yet. It accepts already
validated GyroObservation objects, which is the interface Basilisk will
provide through its future adapter.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from agentic_gnc.navigation.earthloc_lightglue_tie_points_to_verified_visual_matches import (
    EarthLocFrameNavigationContext,
    earthloc_lightglue_tie_points_to_verified_visual_matches,
)
from agentic_gnc.navigation.navigation_measurement_contracts import (
    GyroObservation,
    ODBatch,
)
from agentic_gnc.navigation.verified_visual_matches_to_fsw_landmark_observations import (
    build_landmark_observation,
)


def earthloc_visual_matches_and_basilisk_gyro_to_fsw_od_batch(
    earthloc_coordinate_records: Sequence[Mapping[str, Any]],
    frame_navigation_contexts: Sequence[EarthLocFrameNavigationContext],
    gyro_observations: Sequence[GyroObservation],
) -> ODBatch:
    """Build one FSW-ready ODBatch from visual and inertial measurements."""

    if len(earthloc_coordinate_records) != len(frame_navigation_contexts):
        raise ValueError(
            "earthloc_coordinate_records and frame_navigation_contexts "
            "must have the same length"
        )

    frame_ids = [context.frame_index for context in frame_navigation_contexts]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("every frame_navigation_context must have a unique frame_index")

    landmark_observations = []
    for record, context in zip(
        earthloc_coordinate_records,
        frame_navigation_contexts,
    ):
        verified_matches = (
            earthloc_lightglue_tie_points_to_verified_visual_matches(
                record,
                context,
            )
        )
        landmark_observations.extend(
            build_landmark_observation(match)
            for match in verified_matches
        )

    if not landmark_observations:
        raise ValueError("no verified visual landmarks were available for FSW OD")

    if not gyro_observations:
        raise ValueError("at least one gyro observation is required for FSW OD")

    if not all(np.isfinite(gyro.time_j2000_s) for gyro in gyro_observations):
        raise ValueError("every gyro observation must have a finite timestamp")

    ordered_landmarks = tuple(
        sorted(
            landmark_observations,
            key=lambda observation: (
                observation.time_j2000_s,
                observation.frame_id,
            ),
        )
    )
    ordered_gyros = tuple(
        sorted(
            gyro_observations,
            key=lambda observation: observation.time_j2000_s,
        )
    )

    return ODBatch(
        landmarks=ordered_landmarks,
        gyros=ordered_gyros,
    )