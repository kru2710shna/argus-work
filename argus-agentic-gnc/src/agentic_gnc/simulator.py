"""Minimal VINSat-style synthetic orbit and landmark-observation generator."""

from dataclasses import dataclass

import numpy as np

from agentic_gnc.contracts import GyroObservation, LandmarkObservation, ODBatch

EARTH_RADIUS_M = 6_378_137.0
EARTH_MU_M3_S2 = 3.986_004_418e14


@dataclass(frozen=True)
class OrbitTruth:
    """Ground-truth state sequence in ECI coordinates."""

    times_j2000_s: np.ndarray
    positions_eci_m: np.ndarray
    velocities_eci_m_s: np.ndarray


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def circular_orbit_truth(
    start_j2000_s: float = 0.0,
    altitude_m: float = 420_000.0,
    inclination_deg: float = 51.6,
    steps: int = 20,
    dt_s: float = 10.0,
) -> OrbitTruth:
    """Generate a simple circular Earth orbit in ECI coordinates."""
    radius = EARTH_RADIUS_M + altitude_m
    speed = np.sqrt(EARTH_MU_M3_S2 / radius)
    mean_motion = speed / radius
    inclination = np.radians(inclination_deg)

    times = start_j2000_s + np.arange(steps) * dt_s
    phase = mean_motion * (times - start_j2000_s)

    positions = np.column_stack(
        [
            radius * np.cos(phase),
            radius * np.sin(phase) * np.cos(inclination),
            radius * np.sin(phase) * np.sin(inclination),
        ]
    )
    velocities = np.column_stack(
        [
            -speed * np.sin(phase),
            speed * np.cos(phase) * np.cos(inclination),
            speed * np.cos(phase) * np.sin(inclination),
        ]
    )

    return OrbitTruth(times, positions, velocities)


def body_to_eci_rotation(position_eci_m: np.ndarray) -> np.ndarray:
    """Nadir-pointing body axes expressed in ECI.

    Body +Z points toward Earth. Columns are body X, Y, Z in ECI.
    """
    z_body_eci = _unit(-position_eci_m)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, z_body_eci)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    x_body_eci = _unit(np.cross(reference, z_body_eci))
    y_body_eci = _unit(np.cross(z_body_eci, x_body_eci))
    return np.column_stack([x_body_eci, y_body_eci, z_body_eci])


def angular_velocity_body_rad_s(
    rotation_body_to_eci_now: np.ndarray,
    rotation_body_to_eci_next: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Estimate body-frame angular velocity from two attitude matrices."""
    relative_rotation = rotation_body_to_eci_now.T @ rotation_body_to_eci_next
    skew = (relative_rotation - relative_rotation.T) / (2.0 * dt_s)

    return np.array(
        [
            skew[2, 1],
            skew[0, 2],
            skew[1, 0],
        ]
    )


def ray_to_earth(
    satellite_eci_m: np.ndarray,
    bearing_eci: np.ndarray,
) -> np.ndarray:
    """Intersect a camera ray with a spherical Earth."""
    b = 2.0 * np.dot(satellite_eci_m, bearing_eci)
    c = np.dot(satellite_eci_m, satellite_eci_m) - EARTH_RADIUS_M**2
    discriminant = b**2 - 4.0 * c

    if discriminant <= 0:
        raise ValueError("Camera ray does not intersect Earth")

    distance = (-b - np.sqrt(discriminant)) / 2.0
    return satellite_eci_m + distance * bearing_eci


def generate_synthetic_batch(
    truth: OrbitTruth,
    landmarks_per_frame: int = 4,
    half_fov_deg: float = 20.0,
    bearing_noise_rad: float = 0.0,
    seed: int = 0,
) -> ODBatch:
    """Generate known Earth landmarks and body-frame camera bearings.

    This is a synthetic sensor baseline. It does not yet use VINSat imagery,
    YOLO, Earth rotation, or a calibrated real camera.
    """
    if landmarks_per_frame < 1:
        raise ValueError("landmarks_per_frame must be positive")

    rng = np.random.default_rng(seed)
    observations = []
    gyros = []

    for frame_id, (time_s, position) in enumerate(
        zip(truth.times_j2000_s, truth.positions_eci_m)
    ):
        rotation_body_to_eci = body_to_eci_rotation(position)

        # Placeholder gyro: later replaced by a finite-difference attitude model.
        if len(truth.times_j2000_s) < 2:
            raise ValueError("Synthetic IMU generation requires at least two orbit states")

        if frame_id < len(truth.positions_eci_m) - 1:
            next_rotation = body_to_eci_rotation(truth.positions_eci_m[frame_id + 1])
            gyro_body = angular_velocity_body_rad_s(
                rotation_body_to_eci,
                next_rotation,
                truth.times_j2000_s[frame_id + 1] - time_s,
            )
        else:
            previous_rotation = body_to_eci_rotation(truth.positions_eci_m[frame_id - 1])
            gyro_body = angular_velocity_body_rad_s(
                previous_rotation,
                rotation_body_to_eci,
                time_s - truth.times_j2000_s[frame_id - 1],
            )

        gyros.append(GyroObservation(time_s, gyro_body))

        for _ in range(landmarks_per_frame):
            angle = rng.uniform(0.0, np.radians(half_fov_deg))
            azimuth = rng.uniform(0.0, 2.0 * np.pi)

            bearing_body = np.array(
                [
                    np.sin(angle) * np.cos(azimuth),
                    np.sin(angle) * np.sin(azimuth),
                    np.cos(angle),
                ]
            )
            bearing_eci = rotation_body_to_eci @ bearing_body
            landmark_eci_m = ray_to_earth(position, bearing_eci)

            if bearing_noise_rad > 0.0:
                noise = rng.normal(0.0, bearing_noise_rad, size=3)
                noise -= np.dot(noise, bearing_body) * bearing_body
                bearing_body = _unit(bearing_body + noise)

            observations.append(
                LandmarkObservation(
                    time_j2000_s=float(time_s),
                    frame_id=frame_id,
                    bearing_body=bearing_body,
                    landmark_eci_m=landmark_eci_m,
                    confidence=1.0,
                    bearing_sigma_rad=max(bearing_noise_rad, 1e-6),
                )
            )

    return ODBatch(tuple(observations), tuple(gyros))
