"""
Common earth utilities.
"""

import math
from functools import cache

import brahe
import numpy as np
from brahe import Epoch
from brahe.constants import R_EARTH


# TODO: use brahe constants instead of hardcoding
def ecef_to_lat_lon(
    intersection_points: np.ndarray, a: float = 6378137.0, b: float = 6356752.314245
) -> np.ndarray:
    """
    Convert intersection points (ECEF) to latitude and longitude.

    Parameters:
        intersection_points: A numpy array of shape (..., 3) consisting of ECEF coordinates.

    Returns:
        A numpy array of shape (..., 2) consisting of latitudes and longitudes, or NaN for invalid points.
    """
    assert intersection_points.shape[-1] == 3, "Input must have shape (..., 3)"

    shape_prefix = intersection_points.shape[:-1]
    intersection_points_flat = intersection_points.reshape(-1, 3)

    valid_mask = ~np.isnan(intersection_points_flat).any(axis=1)

    lat_lon_flat = np.full((np.prod(shape_prefix), 2), np.nan)

    valid_points = intersection_points_flat[valid_mask]

    x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Longitude calculation (same for geodetic and geocentric)
    lon = np.degrees(np.arctan2(y, x))

    # Geodetic latitude calculation (iterative approach)
    e2 = (a**2 - b**2) / a**2  # First eccentricity squared
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared
    p = np.sqrt(x**2 + y**2)

    # Initial approximation of latitude
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    # Convert to degrees
    lat = np.degrees(lat)

    # Store results in flat array
    lat_lon_flat[valid_mask, 0] = lat
    lat_lon_flat[valid_mask, 1] = lon

    return lat_lon_flat.reshape(*shape_prefix, 2)


def lat_lon_to_ecef(
    lat_lon: np.ndarray, a: float = 6378137.0, b: float = 6356752.314245
) -> np.ndarray:
    """
    Convert latitude and longitude to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    Parameters:
        lat_lon: A numpy array of shape (..., 2) consisting of latitudes and longitudes.

    Returns:
        np.ndarray: A numpy array of shape (..., 3) consisting of ECEF coordinates.
    """
    assert lat_lon.shape[-1] == 2, "Input must have shape (..., 2)"

    shape_prefix = lat_lon.shape[:-1]
    lat_lon_flat = lat_lon.reshape(-1, 2)

    lat = lat_lon_flat[:, 0]
    lon = lat_lon_flat[:, 1]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # First eccentricity squared
    e2 = (a**2 - b**2) / a**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Assume height h = 0 (on the ellipsoid)
    x = N * np.cos(lat_rad) * np.cos(lon_rad)
    y = N * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2)) * np.sin(lat_rad)

    ecef_flat = np.column_stack((x, y, z))
    return ecef_flat.reshape(*shape_prefix, 3)


def get_nadir_rotation(state: np.ndarray, nadir_axis: str = "x+") -> np.ndarray:
    """
    Get the rotation matrix that points the specified body frame axis towards the center of the Earth.
    The body z-axis will point along the orbital angular momentum vector, the specified body frame axis will point
    towards the center of the Earth, and the third axis will complete the right-handed orthonormal basis.

    This function is agnostic to the frame of reference of the input state. The output rotation matrix
    will be from the frame of reference of the input state to the body frame.

    Parameters:
        state: A numpy array of shape (6,) containing the position and velocity of the satellite.
        nadir_axis: The body frame axis that should point towards the center of the Earth.
                    Must be one of "x+", "y+", "x-", "y-". Defaults to "x+".

    Returns:
        A numpy array of shape (3, 3) representing the rotation matrix from the body frame to the input state frame.
    """
    assert state.shape == (6,), "state must have shape (6,)"
    assert nadir_axis in (
        "x+",
        "y+",
        "x-",
        "y-",
    ), 'nadir_axis must be one of "x+", "y+", "x-", "y-"'

    pos, vel = state[:3], state[3:]
    angular_momentum_dir = np.cross(pos, vel)

    nadir_dir = -pos / np.linalg.norm(pos)
    z_plus_dir = angular_momentum_dir / np.linalg.norm(angular_momentum_dir)

    is_nadir_axis_x = nadir_axis[0] == "x"
    is_nadir_axis_plus = nadir_axis[1] == "+"
    if is_nadir_axis_x:
        x_plus_dir = nadir_dir if is_nadir_axis_plus else -nadir_dir
        y_plus_dir = np.cross(z_plus_dir, x_plus_dir)
    else:
        y_plus_dir = nadir_dir if is_nadir_axis_plus else -nadir_dir
        x_plus_dir = np.cross(y_plus_dir, z_plus_dir)

    return np.column_stack([x_plus_dir, y_plus_dir, z_plus_dir])


def intersect_ellipsoid(
    ray_directions: np.ndarray,
    satellite_position: np.ndarray,
    a: float = 6378137.0,
    b: float = 6356752.314245,
) -> np.ndarray:
    """
    Vectorized computation of ray intersections with the WGS84 ellipsoid.

    Parameters:
        ray_directions: A numpy array of shape (..., 3) containing ray directions.
        satellite_position: Satellite position in ECEF as a numpy array of shape (3,).
        a: Semi-major axis of the WGS84 ellipsoid (meters).
        b: Semi-minor axis of the WGS84 ellipsoid (meters).

    Returns:
        The resulting intersection points as a numpy array of shape (..., 3), or NaN for rays that miss.
    """
    assert ray_directions.shape[-1] == 3, "ray_directions must have shape (..., 3)"
    assert np.allclose(np.linalg.norm(ray_directions, axis=-1), 1), "ray_dirs must be normalized"

    ray_directions_flat = ray_directions.reshape(-1, 3)

    aab_squared = np.array([a, a, b]) ** 2
    A = np.sum(ray_directions_flat**2 / aab_squared, axis=1)
    B = 2 * ray_directions_flat @ (satellite_position / aab_squared)
    C = np.sum(satellite_position**2 / aab_squared) - 1
    discriminant = B**2 - 4 * A * C

    # Initialize intersection points as NaN
    intersection_points_flat = np.full_like(ray_directions_flat, np.nan)

    valid_mask = discriminant >= 0
    if np.any(valid_mask):
        # Compute roots of the quadratic equation
        sqrt_discriminant = np.sqrt(discriminant[valid_mask])
        t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
        t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])

        # Choose the smallest positive t
        t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
        t = np.where(t > 0, t, np.nan)  # Filter out negative t values

        # Calculate intersection points
        valid_ray_directions = ray_directions_flat[valid_mask]
        intersection_points_flat[valid_mask] = (
            t[:, None] * valid_ray_directions + satellite_position
        )

    return intersection_points_flat.reshape(ray_directions.shape)


# Define MGRS latitude bands and UTM exceptions
# TODO: consolidate functionality between this and the get_MGRS_grid function
mgrs_latitude_bands = [
    {"name": b"C", "min_lat": -80, "max_lat": -72},
    {"name": b"D", "min_lat": -72, "max_lat": -64},
    {"name": b"E", "min_lat": -64, "max_lat": -56},
    {"name": b"F", "min_lat": -56, "max_lat": -48},
    {"name": b"G", "min_lat": -48, "max_lat": -40},
    {"name": b"H", "min_lat": -40, "max_lat": -32},
    {"name": b"J", "min_lat": -32, "max_lat": -24},
    {"name": b"K", "min_lat": -24, "max_lat": -16},
    {"name": b"L", "min_lat": -16, "max_lat": -8},
    {"name": b"M", "min_lat": -8, "max_lat": 0},
    {"name": b"N", "min_lat": 0, "max_lat": 8},
    {"name": b"P", "min_lat": 8, "max_lat": 16},
    {"name": b"Q", "min_lat": 16, "max_lat": 24},
    {"name": b"R", "min_lat": 24, "max_lat": 32},
    {"name": b"S", "min_lat": 32, "max_lat": 40},
    {"name": b"T", "min_lat": 40, "max_lat": 48},
    {"name": b"U", "min_lat": 48, "max_lat": 56},
    {"name": b"V", "min_lat": 56, "max_lat": 64},
    {"name": b"W", "min_lat": 64, "max_lat": 72},
    {"name": b"X", "min_lat": 72, "max_lat": 84},  # X spans 12° latitude
]

mgrs_utm_exceptions = [
    {"zone": 32, "min_lon": 3, "max_lon": 12, "bands": [b"V"]},  # Norway
    {"zone": 31, "min_lon": 0, "max_lon": 9, "bands": [b"X"]},  # Svalbard
    {"zone": 33, "min_lon": 9, "max_lon": 21, "bands": [b"X"]},  # Svalbard
    {"zone": 35, "min_lon": 21, "max_lon": 33, "bands": [b"X"]},  # Svalbard
    {"zone": 37, "min_lon": 33, "max_lon": 42, "bands": [b"X"]},  # Svalbard
]


def calculate_mgrs_zones(lat_lon: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of MGRS region identifiers for given latitude and longitude coordinates.

    Note that the output will have a dtype of "S3", meaning each element is an ascii encoded string with a maximum
    length of 3 bytes. You can convert individual elements to regular strings using str(element, encoding="ascii") or
    convert the entire array to regular strings using np.char.decode(mgrs_regions, encoding="ascii").

    Parameters:
        lat_lon: A numpy array of shape (..., 2) containing latitudes and longitudes in degrees. May contain np.nan.

    Returns:
        A numpy array of MGRS region identifiers with dtype "S3" and shape lat_lon.shape[:-1], unless
        lat_lon.shape == (2,) in which case the output will be shape (1,). Output elements will be b"" if either the
        latitude or longitude is np.nan, or if the coordinates correspond to polar regions that are not covered by the
        MGRS.
    """
    assert lat_lon.shape[-1] == 2, "Input must have shape (..., 2)"
    if len(lat_lon.shape) == 1:
        # Special case for single lat/lon pair
        return calculate_mgrs_zones(lat_lon[np.newaxis, :])[0, :]

    # Create lookup tables for vectorized latitude band calculation
    latitude_band_names = np.array([band["name"] for band in mgrs_latitude_bands])
    latitude_band_edges = np.array(
        [[band["min_lat"], band["max_lat"]] for band in mgrs_latitude_bands]
    )

    # Filter out invalid coordinates and flatten
    valid_indices = (
        np.all(~np.isnan(lat_lon), axis=-1) & (lat_lon[..., 0] >= -80) & (lat_lon[..., 0] < 84)
    )
    lat_flat, lon_flat = lat_lon[valid_indices, :].T

    # Determine latitude bands
    lat_bands = np.empty(len(lat_flat), dtype="S1")
    seen_mask = np.zeros(len(lat_flat), dtype=bool)
    for name, (min_lat, max_lat) in zip(latitude_band_names, latitude_band_edges):
        mask = (lat_flat >= min_lat) & (lat_flat < max_lat)
        lat_bands[mask] = name
        assert ~np.any(seen_mask & mask)
        seen_mask |= mask
    assert np.all(seen_mask)

    # Determine UTM zones (default calculation)
    utm_zones = ((lon_flat + 180) // 6 + 1).astype(int)
    for exception in mgrs_utm_exceptions:
        mask = (
            (lon_flat >= exception["min_lon"])
            & (lon_flat < exception["max_lon"])
            & np.isin(lat_bands, exception["bands"])
        )
        utm_zones[mask] = exception["zone"]
    assert np.all(utm_zones >= 1) and np.all(utm_zones <= 60)
    utm_zones = np.char.zfill(utm_zones.astype("S2"), 2)

    mgrs_regions = np.full(valid_indices.shape, b"", dtype="S3")
    mgrs_regions[valid_indices] = np.char.add(utm_zones, lat_bands)
    return mgrs_regions


@cache
def get_MGRS_grid() -> dict[str, tuple[float, float, float, float]]:
    """
    Generate a grid of MGRS (Military Grid Reference System) coordinates.
    Note that keys corresponding to single digit region identifiers have a leading zero (e.g. "01C").

    Returns:
        A dictionary mapping MGRS region identifiers to a tuple containing (min_lon, min_lat, max_lon, max_lat).
    """
    LON_STEP = 6
    LAT_STEP = 8
    lons = np.arange(-180, 180, LON_STEP)
    lats = np.arange(-80, 80, LAT_STEP)
    lon_labels = np.arange(1, 61)
    lat_labels = list("CDEFGHJKLMNPQRSTUVWX")
    mgrs_grid = {}
    for i, lat_label in enumerate(lat_labels):
        for j, lon_label in enumerate(lon_labels):
            mgrs_grid[str(lon_label).zfill(2) + lat_label] = (
                lons[j],
                lats[i],
                lons[j] + LON_STEP,
                lats[i] + LAT_STEP,
            )

    for i in lon_labels:
        idx = str(i).zfill(2) + "X"
        mgrs_grid[idx] = (lons[i - 1], 72, lons[i - 1] + LON_STEP, 84)
    mgrs_grid["31V"] = (0, 56, 3, 64)
    mgrs_grid["32V"] = (3, 56, 12, 64)
    mgrs_grid["31X"] = (0, 72, 9, 84)
    mgrs_grid["33X"] = (9, 72, 21, 84)
    mgrs_grid["35X"] = (21, 72, 33, 84)
    mgrs_grid["37X"] = (33, 72, 42, 84)
    del mgrs_grid["32X"]
    del mgrs_grid["34X"]
    del mgrs_grid["36X"]
    return mgrs_grid


def get_mgrs_region_dimensions(region_id: str) -> tuple[float, float, float]:
    """
    Get the height, the width of the top, and the width of the bottom of a specified MGRS region, in meters.
    Note that this approximates the Earth as a sphere.

    Parameters:
        region_id: The MGRS region identifier.

    Returns:
        A tuple containing (region_height, region_top_width, region_bottom_width) for the specified region.
    """
    min_lon, min_lat, max_lon, max_lat = get_MGRS_grid()[region_id]

    region_height = (np.abs(max_lat - min_lat) / 360) * 2 * np.pi * R_EARTH
    region_top_width = (np.abs(max_lon - min_lon) / 360) * 2 * np.pi * R_EARTH * np.cos(np.deg2rad(max_lat))
    region_bottom_width = (np.abs(max_lon - min_lon) / 360) * 2 * np.pi * R_EARTH * np.cos(np.deg2rad(min_lat))
    return region_height, region_top_width, region_bottom_width


def get_mgrs_region_area(region_id: str) -> float:
    """
    Get the area of a specified MGRS region, in square meters.
    Note that this approximates the Earth as a sphere.

    Parameters:
        region_id: The MGRS region identifier.

    Returns:
        The area of the specified region in square meters.
    """
    min_lon, min_lat, max_lon, max_lat = get_MGRS_grid()[region_id]

    # Taking the definite integral of R_EARTH**2 * cos(lat) dlat dlon over the region yields the following formula.
    return R_EARTH**2 * np.deg2rad(max_lon - min_lon) * (np.sin(np.deg2rad(max_lat)) - np.sin(np.deg2rad(min_lat)))


def noisy_bearing_measurement(vec: np.ndarray, sigma: float = np.sqrt(0.0005)) -> np.ndarray:
    """
    Add Gaussian noise to a bearing measurement.
    Parameters:
        vec (np.ndarray): The original bearing vector.
        sigma (float): The standard deviation of the noise.

    Returns:
        np.ndarray: The noisy bearing vector.
    """

    # Check if at least one of the first two components is nonzero and choose the
    # arbitrary vector accordingly
    n = vec.shape[0]
    cond = (np.abs(vec[:, 0]) <= np.abs(vec[:, 2]))[:, None]  # shape (n,1) for broadcasting
    arbitrary = np.where(cond, np.array([1, 0, 0]), np.array([0, 0, 1]))  # shape (n,3)

    perp_arb = arbitrary - np.sum(arbitrary * vec, axis=1, keepdims=True) * vec
    perp_arb = perp_arb / np.linalg.norm(perp_arb, axis=1, keepdims=True)
    third_vec = np.cross(vec, perp_arb)

    theta = np.random.uniform(0, 2 * np.pi, size=(n, 1))

    noise_direction = np.cos(theta) * perp_arb + np.sin(theta) * third_vec

    # Add the noise to the original vector and renormalize each vector to maintain unit length.
    new_vec = vec + sigma * noise_direction
    new_vec = new_vec / np.linalg.norm(new_vec, axis=1, keepdims=True)

    return new_vec


def density_harris_priester(x: np.ndarray, epoch: Epoch) -> float:
    """
    Harris-Priester atmospheric density model.

    Parameters:
        x: Satellite state vector.
        epoch: Epoch of the satellite state vector.

    Returns:
        Density [kg/m^3].
    """

    # Harris-Priester Constants
    HP_UPPER_LIMIT = 1000.0  # Upper height limit [km]
    HP_LOWER_LIMIT = 100.0  # Lower height limit [km]
    HP_RA_LAG = 0.523599  # Right ascension lag [rad]
    HP_N_PRM = 3  # Harris-Priester parameter
    # 2(6) low(high) inclination
    HP_N = 50  # Number of coefficients

    # Height [km]
    hp_h = [
        100.0,
        120.0,
        130.0,
        140.0,
        150.0,
        160.0,
        170.0,
        180.0,
        190.0,
        200.0,
        210.0,
        220.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        300.0,
        320.0,
        340.0,
        360.0,
        380.0,
        400.0,
        420.0,
        440.0,
        460.0,
        480.0,
        500.0,
        520.0,
        540.0,
        560.0,
        580.0,
        600.0,
        620.0,
        640.0,
        660.0,
        680.0,
        700.0,
        720.0,
        740.0,
        760.0,
        780.0,
        800.0,
        840.0,
        880.0,
        920.0,
        960.0,
        1000.0,
    ]

    # Minimum density [g/km^3]
    hp_c_min = [
        4.974e05,
        2.490e04,
        8.377e03,
        3.899e03,
        2.122e03,
        1.263e03,
        8.008e02,
        5.283e02,
        3.617e02,
        2.557e02,
        1.839e02,
        1.341e02,
        9.949e01,
        7.488e01,
        5.709e01,
        4.403e01,
        3.430e01,
        2.697e01,
        2.139e01,
        1.708e01,
        1.099e01,
        7.214e00,
        4.824e00,
        3.274e00,
        2.249e00,
        1.558e00,
        1.091e00,
        7.701e-01,
        5.474e-01,
        3.916e-01,
        2.819e-01,
        2.042e-01,
        1.488e-01,
        1.092e-01,
        8.070e-02,
        6.012e-02,
        4.519e-02,
        3.430e-02,
        2.632e-02,
        2.043e-02,
        1.607e-02,
        1.281e-02,
        1.036e-02,
        8.496e-03,
        7.069e-03,
        4.680e-03,
        3.200e-03,
        2.210e-03,
        1.560e-03,
        1.150e-03,
    ]

    # Maximum density [g/km^3]
    hp_c_max = [
        4.974e05,
        2.490e04,
        8.710e03,
        4.059e03,
        2.215e03,
        1.344e03,
        8.758e02,
        6.010e02,
        4.297e02,
        3.162e02,
        2.396e02,
        1.853e02,
        1.455e02,
        1.157e02,
        9.308e01,
        7.555e01,
        6.182e01,
        5.095e01,
        4.226e01,
        3.526e01,
        2.511e01,
        1.819e01,
        1.337e01,
        9.955e00,
        7.492e00,
        5.684e00,
        4.355e00,
        3.362e00,
        2.612e00,
        2.042e00,
        1.605e00,
        1.267e00,
        1.005e00,
        7.997e-01,
        6.390e-01,
        5.123e-01,
        4.121e-01,
        3.325e-01,
        2.691e-01,
        2.185e-01,
        1.779e-01,
        1.452e-01,
        1.190e-01,
        9.776e-02,
        8.059e-02,
        5.741e-02,
        4.210e-02,
        3.130e-02,
        2.360e-02,
        1.810e-02,
    ]

    # Satellite height
    r_sun = brahe.ephemerides.sun_position(epc=epoch)

    # Transforming eci -> ecef -> geod conversion
    x = brahe.frames.sECItoECEF(epc=epoch, x=x[0:6])
    geod = brahe.coordinates.sECEFtoGEOD(x[:3], use_degrees=True)
    height = geod[2] / 1.0e3  # height in [km]

    # Exit with zero density above height model limit
    if height > HP_UPPER_LIMIT:
        return 0.0

    # Set height to lower limit if below model limit
    if height < HP_LOWER_LIMIT:
        height = HP_LOWER_LIMIT

    # Sun right ascension, declination
    ra_sun = math.atan2(r_sun[1] , r_sun[0])
    if ra_sun < 0:
        # Adjust for negative angles to ensure ra_sun is in [0, 2*pi]
        ra_sun += 2 * math.pi
    # declination of the sun in radians
    dec_sun =  math.asin(r_sun[2]/np.linalg.norm(r_sun))  

    # Unit vector u towards the apex of the diurnal bulge
    # in inertial geocentric coordinates
    c_dec = math.cos(dec_sun)
    u = np.array(
        [
            c_dec * math.cos(ra_sun + HP_RA_LAG),
            c_dec * np.sin(ra_sun + HP_RA_LAG),
            math.sin(dec_sun),
        ]
    )

    # Cosine of half angle between satellite position vector and
    # apex of diurnal bulge
    c_psi2 = 0.5 + 0.5 * np.dot(x[:3], u) / np.linalg.norm(x[:3])

    # Height index search and exponential density interpolation
    ih = 0  # section index reset
    for i in range(HP_N):  # loop over N_Coef height regimes
        if height >= hp_h[i] and height < hp_h[i + 1]:
            ih = i  # ih identifies height section
            break

    h_min = (hp_h[ih] - hp_h[ih + 1]) / math.log(hp_c_min[ih + 1] / hp_c_min[ih])
    h_max = (hp_h[ih] - hp_h[ih + 1]) / math.log(hp_c_max[ih + 1] / hp_c_max[ih])

    d_min = hp_c_min[ih] * math.exp((hp_h[ih] - height) / h_min)
    d_max = hp_c_max[ih] * math.exp((hp_h[ih] - height) / h_max)

    # Density computation
    density = d_min + (d_max - d_min) * c_psi2**HP_N_PRM

    # Convert from g/km^3 to kg/m^3
    density *= 1.0e-12

    # Finished
    return density
