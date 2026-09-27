"""Convert known Earth map locations into FSW-ready ECI landmarks.

WHAT THIS FILE DOES
-------------------
This is the map-location half of the Navigation Measurement Builder.

It converts:

    known map-feature location in ECEF metres
    + ECEF-to-ECI rotation at the image timestamp
    -> known landmark position in ECI metres

FSW-Payload requires the landmark position in ECI coordinates.

WHAT PROVIDES THE INPUTS LATER
------------------------------
- EarthLoc identifies the likely map tile.
- LightGlue verifies a feature match.
- Map georeferencing gives the feature's ECEF position.
- AstroPy/SPICE gives the timestamp-specific ECEF-to-ECI rotation.

This file deliberately does not approximate Earth rotation itself.
"""

from __future__ import annotations

import numpy as np


def _earth_scale_vector(value: object, name: str) -> np.ndarray:
    """Validate a finite Earth-scale Cartesian position in metres."""
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite shape-(3,) vector")

    # Earth-surface locations should be approximately 6.37 million metres
    # from the Earth's center.
    if np.linalg.norm(vector) < 6.0e6:
        raise ValueError(f"{name} must be an Earth-scale position in metres")

    return vector


def _rotation_matrix(value: object, name: str) -> np.ndarray:
    """Validate a proper 3D rotation matrix."""
    matrix = np.asarray(value, dtype=np.float64)

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite shape-(3, 3) matrix")

    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")

    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must have determinant +1")

    return matrix


def known_map_location_ecef_to_eci_landmark(
    map_location_ecef_m: object,
    rotation_ecef_to_eci: object,
) -> np.ndarray:
    """Convert one known map location from ECEF metres to ECI metres.

    Input:
        map_location_ecef_m:
            Earth-fixed x/y/z location from map georeferencing.

        rotation_ecef_to_eci:
            Timestamp-specific 3x3 rotation supplied by AstroPy/SPICE.

    Output:
        ECI x/y/z landmark position in metres for FSW-Payload.
    """
    location_ecef_m = _earth_scale_vector(
        map_location_ecef_m,
        "map_location_ecef_m",
    )
    rotation = _rotation_matrix(
        rotation_ecef_to_eci,
        "rotation_ecef_to_eci",
    )

    return rotation @ location_ecef_m