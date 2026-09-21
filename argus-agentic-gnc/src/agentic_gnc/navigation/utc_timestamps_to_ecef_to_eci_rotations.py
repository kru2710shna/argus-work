"""Convert UTC observation timestamps into ECEF-to-ECI rotations.

WHAT THIS FILE DOES
-------------------
The visual-localization pipeline gives Earth landmarks in ECEF: a coordinate
system fixed to the rotating Earth.

FSW-Payload needs those landmarks in an inertial frame. This module uses
AstroPy's public coordinate API to construct the timestamp-specific rotation:

    UTC camera timestamp
    -> Earth orientation at that time
    -> 3x3 ECEF/ITRS-to-ECI/GCRS rotation matrix

The returned matrix follows this convention:

    position_eci_m = rotation_ecef_to_eci @ position_ecef_m

Also included:

    UTC timestamp -> seconds since the J2000 TT epoch

That value is the time format used by the navigation measurement contract.

FRAME NOTE
----------
"ECI" is made explicit here as AstroPy GCRS. Before a flight-style integration,
we must confirm that FSW-Payload uses the same inertial-frame convention.
"""

from __future__ import annotations

from typing import Union

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
from astropy.time import Time

UtcTimestamp = Union[str, Time]

_J2000_EPOCH_TT = Time("2000-01-01T12:00:00", scale="tt")


def utc_timestamp_to_ecef_to_eci_rotation(
    timestamp_utc: UtcTimestamp,
) -> np.ndarray:
    """Return the 3x3 ECEF/ITRS-to-ECI/GCRS rotation at one UTC timestamp."""

    observation_time = _as_scalar_time(timestamp_utc)

    # Transform each ECEF basis vector with AstroPy's public ITRS -> GCRS API.
    # The transformed vectors become the columns of the rotation matrix.
    rotation_ecef_to_eci = np.column_stack(
        [
            _ecef_basis_vector_to_eci(
                np.array([1.0, 0.0, 0.0]),
                observation_time,
            ),
            _ecef_basis_vector_to_eci(
                np.array([0.0, 1.0, 0.0]),
                observation_time,
            ),
            _ecef_basis_vector_to_eci(
                np.array([0.0, 0.0, 1.0]),
                observation_time,
            ),
        ]
    )

    if not np.allclose(
        rotation_ecef_to_eci.T @ rotation_ecef_to_eci,
        np.eye(3),
        atol=1e-10,
    ):
        raise RuntimeError("AstroPy returned a non-orthonormal ECEF-to-ECI rotation")

    if not np.isclose(np.linalg.det(rotation_ecef_to_eci), 1.0, atol=1e-10):
        raise RuntimeError("AstroPy returned an improper ECEF-to-ECI rotation")

    return rotation_ecef_to_eci


def utc_timestamp_to_j2000_seconds(timestamp_utc: UtcTimestamp) -> float:
    """Convert one UTC timestamp to elapsed seconds since J2000 TT."""

    observation_time = _as_scalar_time(timestamp_utc)
    return float((observation_time.tt - _J2000_EPOCH_TT).to_value(u.s))


def _ecef_basis_vector_to_eci(
    vector_ecef_m: np.ndarray,
    observation_time: Time,
) -> np.ndarray:
    """Transform one ECEF basis vector to ECI/GCRS coordinates."""

    ecef = ITRS(
        CartesianRepresentation(*(vector_ecef_m * u.m)),
        obstime=observation_time,
    )
    eci = ecef.transform_to(GCRS(obstime=observation_time))
    return eci.cartesian.xyz.to_value(u.m)


def _as_scalar_time(timestamp_utc: UtcTimestamp) -> Time:
    """Parse a scalar UTC timestamp without changing its time information."""

    observation_time = (
        timestamp_utc
        if isinstance(timestamp_utc, Time)
        else Time(timestamp_utc, scale="utc")
    )

    if not observation_time.isscalar:
        raise ValueError("timestamp_utc must describe one timestamp")

    return observation_time