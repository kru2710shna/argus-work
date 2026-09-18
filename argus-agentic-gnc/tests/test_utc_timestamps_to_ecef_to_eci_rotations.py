"""Tests for UTC timestamp to ECEF-to-ECI conversion."""

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
from astropy.time import Time

from agentic_gnc.navigation.utc_timestamps_to_ecef_to_eci_rotations import (
    utc_timestamp_to_ecef_to_eci_rotation,
    utc_timestamp_to_j2000_seconds,
)


def test_timestamp_produces_a_proper_ecef_to_eci_rotation() -> None:
    rotation = utc_timestamp_to_ecef_to_eci_rotation("2020-04-11T00:00:00")

    assert rotation.shape == (3, 3)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_rotation_matches_astropy_coordinate_transform() -> None:
    timestamp = Time("2020-04-11T00:00:00", scale="utc")
    position_ecef_m = np.array([6_378_137.0, 0.0, 0.0])

    ecef = ITRS(
        CartesianRepresentation(*(position_ecef_m * u.m)),
        obstime=timestamp,
    )
    expected_eci_m = ecef.transform_to(
        GCRS(obstime=timestamp)
    ).cartesian.xyz.to_value(u.m)

    rotation = utc_timestamp_to_ecef_to_eci_rotation(timestamp)
    actual_eci_m = rotation @ position_ecef_m

    np.testing.assert_allclose(actual_eci_m, expected_eci_m, atol=1e-6)


def test_j2000_seconds_increase_with_utc_time() -> None:
    first_time_j2000_s = utc_timestamp_to_j2000_seconds("2020-04-11T00:00:00")
    second_time_j2000_s = utc_timestamp_to_j2000_seconds("2020-04-11T00:01:00")

    np.testing.assert_allclose(second_time_j2000_s - first_time_j2000_s, 60.0)