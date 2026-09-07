"""
Prepares the training data for the specified MGRS regions for training YOLO models (GPU-accelerated).

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - 00000.png
    - 00000_lat_lon.npz
    - ...
    - bounding_boxes.csv

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - /LD_training
      - dataset.yaml
      - /train
        - /images
          - 00000.png (symlink)
          - ...
        - /labels
          - 00000.txt
          - ...
      - /test
        - ...
      - /val
        - ...
"""

import argparse
import os
from dataclasses import dataclass
from functools import partial
from itertools import starmap
from multiprocessing import Pool, cpu_count
from typing import ClassVar, Generator, Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available. Falling back to CPU computation.")
    print("Install CuPy for GPU acceleration: pip install cupy-cuda11x (replace 11x with your CUDA version)")

from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import lat_lon_to_ecef
from utils.function_utils import unpack_and_call

LD_TRAINING_DIR_NAME = "LD_training"
YOLO_CONFIG_FILE_NAME = "dataset.yaml"
SPLIT_DIR_NAMES = ["train", "test", "val"]
BOUNDING_BOXES_VISUALIZATION_FILE_NAME = "bounding_boxes.png"


# ==================== Ported Dependencies ====================

@dataclass
class GeotaggedImage:
    """
    A class representing an image alongside lat/lon coordinates for each pixel.

    Attributes:
        image: A numpy array containing the RGB image.
        lat_lon: A numpy array containing the latitudes and longitudes for each
                 pixel, or np.nan if the pixel does not intersect the Earth.
    """

    IMAGE_SUFFIX: ClassVar[str] = ".png"
    LAT_LON_SUFFIX: ClassVar[str] = "_lat_lon.npz"

    image: np.ndarray
    lat_lon: np.ndarray

    @staticmethod
    def load(region: str, file_prefix: str, delete_bad_files: bool = True) -> "GeotaggedImage":
        """
        Load the image and lat/lon coordinates from the specified region and file prefix.

        :param region: The MGRS region.
        :param file_prefix: The prefix for the output files.
        :param delete_bad_files: Whether to delete the files if they are malformed.
        :return: A GeotaggedImage object containing the loaded image and lat/lon coordinates.
        """
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region)

        img_path = os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.IMAGE_SUFFIX}")
        lat_lon_path = os.path.join(region_dir, f"{file_prefix}{GeotaggedImage.LAT_LON_SUFFIX}")

        if not os.path.exists(img_path) or not os.path.exists(lat_lon_path):
            raise FileNotFoundError(
                f"Image or lat/lon file not found for region {region} with prefix {file_prefix}."
            )

        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            with np.load(lat_lon_path) as data:
                lat_lon = data["lat_lon"]

            return GeotaggedImage(image, lat_lon)
        except Exception:
            if delete_bad_files:
                print(f"Warning: malformed image or lat/lon file for {region=}, {file_prefix=}. Deleting.")
                os.remove(img_path)
                os.remove(lat_lon_path)
            raise


def get_common_file_name_prefixes(input_dir: str, ignore_names: Iterable[str] = ()) -> List[str]:
    """
    Get the common file name prefixes for PNG and .npz lat/lon files in the input directory.
    A warning is printed if there are PNG files without corresponding lat/lon files, or vice versa.

    :param input_dir: The directory containing the PNGs and .npz files.
    :param ignore_names: An iterable of file names to ignore.
    :return: A list of common file name prefixes.
    """
    file_names = sorted(os.listdir(input_dir))
    image_file_prefixes = {
        name[: -len(GeotaggedImage.IMAGE_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.IMAGE_SUFFIX) and name not in ignore_names
    }
    lat_lon_file_prefixes = {
        name[: -len(GeotaggedImage.LAT_LON_SUFFIX)]
        for name in file_names
        if name.endswith(GeotaggedImage.LAT_LON_SUFFIX) and name not in ignore_names
    }
    common_file_prefixes = list(image_file_prefixes & lat_lon_file_prefixes)

    if len(image_file_prefixes) != len(common_file_prefixes):
        print(
            f"Warning: Some PNG files do not have corresponding lat/lon files: "
            f"{list(image_file_prefixes - lat_lon_file_prefixes)}"
        )
    if len(lat_lon_file_prefixes) != len(common_file_prefixes):
        print(
            f"Warning: Some lat/lon files do not have corresponding PNG files: "
            f"{list(lat_lon_file_prefixes - image_file_prefixes)}"
        )
    return common_file_prefixes


# ==================== Standalone LandmarkDetector Functions ====================

def get_region_bounding_boxes_relative_path(region_id: str) -> str:
    """
    Get the relative path to the bounding box lat/lon coordinates file for a specific MGRS region.

    :param region_id: The MGRS region ID to get the bounding box coordinates relative path for.
    :return: The relative path to the bounding box coordinates file.
    """
    return os.path.join(region_id, "bounding_boxes.csv")


def load_ground_truth(ground_truth_path: str) -> np.ndarray:
    """
    Loads ground truth bounding box coordinates from a CSV file.

    :param ground_truth_path: Path to the ground truth CSV file.
    :return: A numpy array of shape (N, 6) containing the following for each landmark:
             (centroid_lat, centroid_lon, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon).
    """
    return np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)


# ==================== Standalone Logger ====================

def log_message(level: str, message: str) -> None:
    """
    Simple standalone logging function.

    :param level: The log level (INFO, WARNING, ERROR, DEBUG).
    :param message: The message to log.
    """
    print(f"[{level}] {message}")


# ==================== Main Functions ====================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepares the training data for the specified MGRS regions for training YOLO models (GPU-accelerated)."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions for which to prepare training data for training YOLO models.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory if it exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume preparing yolo data for all requests that failed in the previous run.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.8 * cpu_count()),
        help="Number of processes to use to prepare training data for training YOLO models "
        "in parallel across the specified regions.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of images to use for testing.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of images to use for validation.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=GPU_AVAILABLE,
        help="Use GPU acceleration with CuPy (default: True if CuPy is available).",
    )
    parser.add_argument(
        "--gpu_batch_size",
        type=int,
        default=10000,
        help="Number of pixels to process per GPU batch (adjust based on GPU memory).",
    )
    parser.add_argument(
        "--dark_threshold",
        type=int,
        default=10,
        help="Pixels with sum of RGB values below this are considered dark/invalid (default: 10).",
    )
    parser.add_argument(
        "--min_data_threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of bbox that must have valid (non-dark) pixels (default: 0.5).",
    )
    parser.add_argument(
        "--no_center_check",
        action="store_true",
        help="Disable checking if bbox center is in a non-dark region.",
    )
    return parser.parse_args()


def setup_LD_training_directory(
    region_id: str, overwrite: bool, resume: bool, test_fraction: float, val_fraction: float
) -> bool:
    """
    Set up the LD_training directory for the prepared YOLO training data.
    This includes creating the directories, writing the dataset.yaml file, performing the train/test/val split, and
    creating symlinks to the image files.

    :param region_id: The MGRS region ID to set up the LD_training directory for.
    :param overwrite: Whether to overwrite the output files if they exist. Cannot be True if resume is also True.
    :param resume: Whether to resume preparing YOLO data for all requests that failed in the previous run. Cannot be
                   True if overwrite is also True.
    :param test_fraction: The fraction of images to use for testing.
    :param val_fraction: The fraction of images to use for validation.
    :return: True if LD_training is now a directory that is ready for preparing YOLO training data, False otherwise.
    """
    assert not (resume and overwrite), "resume and overwrite cannot be used together."
    train_fraction = 1 - test_fraction - val_fraction
    assert 0 <= train_fraction <= 1, "test_fraction + val_fraction must be less than or equal to 1."
    assert 0 <= test_fraction <= 1, "test_fraction must be in the range [0, 1]."
    assert 0 <= val_fraction <= 1, "val_fraction must be in the range [0, 1]."

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    LD_training_dir = os.path.join(training_dir, region_id, LD_TRAINING_DIR_NAME)

    if not os.path.exists(LD_training_dir):
        os.makedirs(LD_training_dir, exist_ok=True)

    if not os.path.isdir(LD_training_dir):
        if not overwrite:
            return False
        os.remove(LD_training_dir)
        os.makedirs(LD_training_dir, exist_ok=True)

    yolo_config_path = os.path.join(LD_training_dir, YOLO_CONFIG_FILE_NAME)
    write_yolo_config = False
    if not os.path.exists(yolo_config_path):
        write_yolo_config = True
    else:
        if not overwrite and not resume:
            return False
        if overwrite:
            os.remove(yolo_config_path)
            write_yolo_config = True

    if write_yolo_config:
        bounding_boxes_lat_lon = load_ground_truth(
            os.path.join(
                training_dir, get_region_bounding_boxes_relative_path(region_id)
            )
        )
        num_classes = bounding_boxes_lat_lon.shape[0]
        yolo_config = {
            "path": os.path.abspath(LD_training_dir),
            "train": "train/images",
            "test": "test/images",
            "val": "val/images",
            "nc": num_classes,
            "names": [str(i) for i in range(num_classes)],
        }

        with open(yolo_config_path, "w", encoding="utf-8") as yolo_config_file:
            yaml.dump(yolo_config, yolo_config_file, default_flow_style=False)

    split_img_file_names_list: List[List[str]] = []
    split_label_file_names_list: List[List[str]] = []
    for split_dir_name in SPLIT_DIR_NAMES:
        split_dir = os.path.join(LD_training_dir, split_dir_name)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        split_img_file_names = [
            file_name for file_name in os.listdir(images_dir) if file_name.endswith(".png")
        ]
        split_label_file_names = [
            file_name for file_name in os.listdir(labels_dir) if file_name.endswith(".txt")
        ]

        if not overwrite and not resume:
            if len(split_img_file_names) > 0 or len(split_label_file_names) > 0:
                return False
        if overwrite:
            for img_file_name in split_img_file_names:
                os.remove(os.path.join(images_dir, img_file_name))
            for label_file_name in split_label_file_names:
                os.remove(os.path.join(labels_dir, label_file_name))
            split_img_file_names = []
            split_label_file_names = []

        split_img_file_names_list.append(split_img_file_names)
        split_label_file_names_list.append(split_label_file_names)

    file_prefixes = get_common_file_name_prefixes(
        os.path.join(training_dir, region_id), ignore_names=[BOUNDING_BOXES_VISUALIZATION_FILE_NAME]
    )
    if not resume:
        num_files = len(file_prefixes)
        train_cutoff = int(num_files * train_fraction)
        test_cutoff = int(num_files * (train_fraction + test_fraction))

        all_indices = np.arange(num_files)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:train_cutoff]
        test_indices = all_indices[train_cutoff:test_cutoff]
        val_indices = all_indices[test_cutoff:]

        for split_indices, split_dir_name in zip(
            [train_indices, test_indices, val_indices], SPLIT_DIR_NAMES
        ):
            for i in split_indices:
                file_prefix = file_prefixes[i]
                os.symlink(
                    os.path.join(training_dir, region_id, f"{file_prefix}.png"),
                    os.path.join(LD_training_dir, split_dir_name, "images", f"{file_prefix}.png"),
                )
    else:
        seen_img_file_prefixes = set()
        for split_dir_name, split_img_file_names, split_label_file_names in zip(
            SPLIT_DIR_NAMES, split_img_file_names_list, split_label_file_names_list
        ):
            split_img_file_prefixes = {
                file_name[: -len(".png")] for file_name in split_img_file_names
            }
            split_label_file_prefixes = {
                file_name[: -len(".txt")] for file_name in split_label_file_names
            }
            if not (split_label_file_prefixes <= split_img_file_prefixes):
                raise ValueError(
                    f"Label files are not a subset of image files for {os.path.join(LD_training_dir, split_dir_name)}"
                )

            if len(seen_img_file_prefixes & split_img_file_prefixes) > 0:
                raise ValueError(
                    f"Duplicate image files found in different splits for {LD_training_dir}"
                )
            seen_img_file_prefixes |= split_img_file_prefixes

        if seen_img_file_prefixes != set(file_prefixes):
            raise ValueError(
                f"Image files do not match the expected image files for {LD_training_dir}"
            )

    return True


def is_bbox_center_valid(
    image: np.ndarray,
    closest_us: np.ndarray,
    closest_vs: np.ndarray,
    dark_threshold: int = 10,
) -> np.ndarray:
    """
    Check if the center of each bounding box falls in a non-dark region.

    This is a fast pre-filter to reject bounding boxes whose centers fall in
    dark/black regions (e.g., areas outside the satellite's field of view).

    :param image: The RGB image to check against.
    :param closest_us: A numpy array of shape (N, 4) containing u-coordinates of bbox corners.
    :param closest_vs: A numpy array of shape (N, 4) containing v-coordinates of bbox corners.
    :param dark_threshold: Pixels with sum of RGB values below this are considered dark.
    :return: A boolean numpy array of shape (N,) indicating which bbox centers are valid.
    """
    # Compute center of each bounding box
    center_us = np.mean(closest_us, axis=1).astype(int)
    center_vs = np.mean(closest_vs, axis=1).astype(int)

    height, width = image.shape[:2]

    # Clamp to image bounds
    center_us = np.clip(center_us, 0, width - 1)
    center_vs = np.clip(center_vs, 0, height - 1)

    # Check if center pixel is not dark
    center_pixels = image[center_vs, center_us]  # Shape: (N, 3)
    center_brightness = np.sum(center_pixels, axis=1)  # Sum of RGB

    return center_brightness > dark_threshold


def get_valid_bounding_boxes(
    image: np.ndarray,
    closest_us: np.ndarray,
    closest_vs: np.ndarray,
    minimum_data_threshold: float = 0.5,
    check_center: bool = True,
    dark_threshold: int = 10,
) -> np.ndarray:
    """
    Compute a boolean mask indicating which bounding boxes are considered valid and should be used for training.

    A bounding box is considered valid if:
    1. None of the four corners' closest pixels are on the boundary of the image
    2. The center of the bounding box is not in a dark region (if check_center=True)
    3. The fraction of the bounding box containing nonzero data is above the specified threshold

    :param image: The image to check the bounding boxes against.
    :param closest_us: A numpy array of shape (N, 4) containing the u-coordinates of the closest pixel to the top left,
                       top right, bottom right, and bottom left corners of the bounding box, respectively.
    :param closest_vs: A numpy array of shape (N, 4) containing the v-coordinates of the closest pixel to the top left,
                       top right, bottom right, and bottom left corners of the bounding box, respectively.
    :param minimum_data_threshold: The minimum fraction of the bounding box that must contain nonzero data for it to be
                                   considered valid. Must be in the range [0, 1].
    :param check_center: Whether to check if the bbox center is in a non-dark region.
    :param dark_threshold: Pixels with sum of RGB values below this are considered dark.
    :return: A boolean numpy array of shape (N,) indicating which bounding boxes are valid.
    """
    assert image.ndim == 3, f"Expected image to have 3 dimensions, but got {image.ndim}."
    assert (
        closest_us.shape == closest_vs.shape
    ), f"Expected closest_us and closest_vs to have the same shape, but got {closest_us.shape} and {closest_vs.shape}."
    assert (
        closest_us.ndim == 2
    ), f"Expected closest_us and closest_vs to have 2 dimensions, but got {closest_us.ndim}."
    assert (
        closest_us.shape[1] == 4
    ), f"Expected closest_us and closest_vs to have 4 columns, but got {closest_us.shape[1]}."

    # reject any bounding boxes that have a corner whose closest pixel is on the boundary of the image
    height, width = image.shape[:2]
    valid_bounding_boxes = np.all(
        (closest_us > 0) & (closest_us < width - 1) & (closest_vs > 0) & (closest_vs < height - 1),
        axis=1,
    )

    # Check if bbox center is in a non-dark region (fast pre-filter)
    if check_center:
        center_valid = is_bbox_center_valid(image, closest_us, closest_vs, dark_threshold)
        valid_bounding_boxes &= center_valid

    has_data = np.any(image > 0, axis=2)
    for i in range(len(valid_bounding_boxes)):
        if not valid_bounding_boxes[i]:
            continue

        # since the bounding box corners are mapped from lat/lon to pixel coordinates, they could form any quadrilateral
        # cv2.fillPoly does not support numpy arrays with dtype=bool, so we need to use uint8 instead
        quadrilateral_mask = np.zeros(has_data.shape, dtype=np.uint8)
        cv2.fillPoly(
            quadrilateral_mask,
            # OpenCV expects the vertices to be in the form (num_polygons, num_points, 2)
            np.column_stack((closest_us[i, :], closest_vs[i, :]))[np.newaxis, ...],
            color=1,
        )
        quadrilateral_mask = quadrilateral_mask.astype(bool)

        if (
            np.sum(has_data[quadrilateral_mask]) / np.sum(quadrilateral_mask)
            < minimum_data_threshold
        ):
            valid_bounding_boxes[i] = False

    return valid_bounding_boxes


def generate_yolo_label_gpu(
    region_id: str,
    split_dir_name: str,
    file_prefix: str,
    gpu_batch_size: int = 10000,
    dark_threshold: int = 10,
    min_data_threshold: float = 0.5,
    check_center: bool = True,
) -> None:
    """
    Generate a YOLO label .txt file for the specified region and file prefix using GPU acceleration.

    The .txt file contains one line for each bounding box that is entirely contained within the image.
    The line is formatted as follows, with all coordinates normalized to the range [0, 1]:
    <class_id> <u_center> <v_center> <box_width> <box_height>

    :param region_id: The MGRS region ID to generate YOLO label files for.
    :param split_dir_name: The name of the split directory that the file prefix belongs to. Must be an element of
                           SPLIT_DIR_NAMES.
    :param file_prefix: The common prefix of the PNG and lat/lon .npz files to process and the .txt YOLO label file to
                        generate.
    :param gpu_batch_size: The number of pixels to process in each GPU batch when finding the closest pixel to each
                          bounding box corner. Smaller values use less GPU memory but may be slower.
    :param dark_threshold: Pixels with sum of RGB below this are considered dark/invalid.
    :param min_data_threshold: Minimum fraction of bbox that must have valid pixels.
    :param check_center: Whether to check if bbox center is in a non-dark region.
    """
    assert split_dir_name in SPLIT_DIR_NAMES, f"Invalid split directory name: {split_dir_name}"

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    bounding_boxes_lat_lon = load_ground_truth(
        os.path.join(
            training_dir, get_region_bounding_boxes_relative_path(region_id)
        )
    )
    num_classes = bounding_boxes_lat_lon.shape[0]

    # CSV columns are: [centroid_lon, centroid_lat, tl_lon, tl_lat, br_lon, br_lat]
    # But lat_lon_to_ecef expects [lat, lon], so we need to swap the order
    top_left_lat_lon = bounding_boxes_lat_lon[:, [3, 2]]  # [tl_lat, tl_lon]
    bottom_right_lat_lon = bounding_boxes_lat_lon[:, [5, 4]]  # [br_lat, br_lon]
    top_right_lat_lon = np.column_stack((top_left_lat_lon[:, 0], bottom_right_lat_lon[:, 1]))  # [tl_lat, br_lon]
    bottom_left_lat_lon = np.column_stack((bottom_right_lat_lon[:, 0], top_left_lat_lon[:, 1]))  # [br_lat, tl_lon]

    stacked_corners_lat_lon = np.concatenate(
        # must be in a circular order for cv2.fillPoly to work correctly
        (top_left_lat_lon, top_right_lat_lon, bottom_right_lat_lon, bottom_left_lat_lon), axis=0
    )
    stacked_corners_ecef = lat_lon_to_ecef(stacked_corners_lat_lon)

    try:
        geotagged_image = GeotaggedImage.load(region_id, file_prefix, delete_bad_files=False)
    except AssertionError:
        # Skip dimension assertion for real satellite images - load manually
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region_id)

        img_path = os.path.join(region_dir, f"{file_prefix}.png")
        lat_lon_path = os.path.join(region_dir, f"{file_prefix}_lat_lon.npz")

        # Check if files exist
        if not os.path.exists(img_path) or not os.path.exists(lat_lon_path):
            log_message("WARNING", f"Missing file for: {region_id=}, {file_prefix=}.")
            return

        # Try to load image
        image = cv2.imread(img_path)
        if image is None:
            log_message("WARNING", f"Corrupted image file: {img_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            with np.load(lat_lon_path) as data:
                lat_lon = data["lat_lon"]
        except Exception as e:
            log_message("WARNING", f"Failed to load lat/lon for {file_prefix}: {e}")
            return

        # Create object without strict assertions
        class FlexibleGeotaggedImage:
            def __init__(self, img, ll):
                self.image = img
                self.lat_lon = ll

        geotagged_image = FlexibleGeotaggedImage(image, lat_lon)

    except Exception as e:
        log_message("WARNING", f"Failed to load geotagged image for: {region_id=}, {file_prefix=}. Error: {e}")
        return

    height, width = geotagged_image.image.shape[:2]
    pixel_coordinates_ecef = lat_lon_to_ecef(geotagged_image.lat_lon).reshape(-1, 3)

    # GPU-accelerated distance computation
    if GPU_AVAILABLE:
        try:
            # Transfer data to GPU
            stacked_corners_ecef_gpu = cp.asarray(stacked_corners_ecef)

            closest_pixel_indices = cp.empty(4 * num_classes, dtype=int)
            minimum_distances = cp.full(4 * num_classes, cp.inf)

            for start_pixel_idx in range(0, height * width, gpu_batch_size):
                end_pixel_idx = min(start_pixel_idx + gpu_batch_size, height * width)
                pixel_slice = slice(start_pixel_idx, end_pixel_idx)

                # Transfer batch to GPU
                pixel_batch_gpu = cp.asarray(pixel_coordinates_ecef[pixel_slice])

                # Compute distances on GPU
                x_distances = cp.subtract.outer(
                    stacked_corners_ecef_gpu[:, 0], pixel_batch_gpu[:, 0]
                )
                y_distances = cp.subtract.outer(
                    stacked_corners_ecef_gpu[:, 1], pixel_batch_gpu[:, 1]
                )
                z_distances = cp.subtract.outer(
                    stacked_corners_ecef_gpu[:, 2], pixel_batch_gpu[:, 2]
                )

                distances = cp.linalg.norm(
                    cp.stack((x_distances, y_distances, z_distances), axis=0), axis=0
                )

                batch_closest_pixel_indices = cp.argmin(distances, axis=1)
                batch_minimum_distances = distances[cp.arange(4 * num_classes), batch_closest_pixel_indices]

                closer_mask = batch_minimum_distances < minimum_distances
                closest_pixel_indices[closer_mask] = (
                    batch_closest_pixel_indices[closer_mask] + start_pixel_idx
                )
                minimum_distances[closer_mask] = batch_minimum_distances[closer_mask]

            # Transfer results back to CPU
            closest_pixel_indices = cp.asnumpy(closest_pixel_indices)

        except Exception as e:
            log_message("WARNING", f"GPU processing failed, falling back to CPU: {e}")
            # Fall back to CPU computation
            closest_pixel_indices = np.empty(4 * num_classes, dtype=int)
            minimum_distances = np.full(4 * num_classes, np.inf)

            for start_pixel_idx in range(0, height * width, gpu_batch_size):
                end_pixel_idx = min(start_pixel_idx + gpu_batch_size, height * width)
                pixel_slice = slice(start_pixel_idx, end_pixel_idx)

                x_distances = np.subtract.outer(
                    stacked_corners_ecef[:, 0], pixel_coordinates_ecef[pixel_slice, 0]
                )
                y_distances = np.subtract.outer(
                    stacked_corners_ecef[:, 1], pixel_coordinates_ecef[pixel_slice, 1]
                )
                z_distances = np.subtract.outer(
                    stacked_corners_ecef[:, 2], pixel_coordinates_ecef[pixel_slice, 2]
                )

                distances = np.linalg.norm(
                    np.stack((x_distances, y_distances, z_distances), axis=0), axis=0
                )

                batch_closest_pixel_indices = np.argmin(distances, axis=1)
                batch_minimum_distances = distances[np.arange(4 * num_classes), batch_closest_pixel_indices]

                closer_mask = batch_minimum_distances < minimum_distances
                closest_pixel_indices[closer_mask] = (
                    batch_closest_pixel_indices[closer_mask] + start_pixel_idx
                )
                minimum_distances[closer_mask] = batch_minimum_distances[closer_mask]
    else:
        # CPU computation
        closest_pixel_indices = np.empty(4 * num_classes, dtype=int)
        minimum_distances = np.full(4 * num_classes, np.inf)

        for start_pixel_idx in range(0, height * width, gpu_batch_size):
            end_pixel_idx = min(start_pixel_idx + gpu_batch_size, height * width)
            pixel_slice = slice(start_pixel_idx, end_pixel_idx)

            x_distances = np.subtract.outer(
                stacked_corners_ecef[:, 0], pixel_coordinates_ecef[pixel_slice, 0]
            )
            y_distances = np.subtract.outer(
                stacked_corners_ecef[:, 1], pixel_coordinates_ecef[pixel_slice, 1]
            )
            z_distances = np.subtract.outer(
                stacked_corners_ecef[:, 2], pixel_coordinates_ecef[pixel_slice, 2]
            )

            distances = np.linalg.norm(
                np.stack((x_distances, y_distances, z_distances), axis=0), axis=0
            )

            batch_closest_pixel_indices = np.argmin(distances, axis=1)
            batch_minimum_distances = distances[np.arange(4 * num_classes), batch_closest_pixel_indices]

            closer_mask = batch_minimum_distances < minimum_distances
            closest_pixel_indices[closer_mask] = (
                batch_closest_pixel_indices[closer_mask] + start_pixel_idx
            )
            minimum_distances[closer_mask] = batch_minimum_distances[closer_mask]

    closest_vs, closest_us = np.unravel_index(closest_pixel_indices, (height, width))
    closest_us = closest_us.reshape(4, num_classes).T
    closest_vs = closest_vs.reshape(4, num_classes).T

    valid_bounding_boxes = get_valid_bounding_boxes(
        geotagged_image.image,
        closest_us,
        closest_vs,
        minimum_data_threshold=min_data_threshold,
        check_center=check_center,
        dark_threshold=dark_threshold,
    )
    closest_us = closest_us[valid_bounding_boxes, :]
    closest_vs = closest_vs[valid_bounding_boxes, :]
    class_ids = np.arange(num_classes)[valid_bounding_boxes]

    # widen bounding boxes to the smallest axis-aligned bounding box that contains all 4 corners
    min_us = np.min(closest_us, axis=1)
    max_us = np.max(closest_us, axis=1)
    min_vs = np.min(closest_vs, axis=1)
    max_vs = np.max(closest_vs, axis=1)

    # convert to YOLO format
    u_center = (min_us + max_us) / (2 * width)
    v_center = (min_vs + max_vs) / (2 * height)
    box_width = (max_us - min_us) / width
    box_height = (max_vs - min_vs) / height

    yolo_label_path = os.path.join(
        training_dir, region_id, LD_TRAINING_DIR_NAME, split_dir_name, "labels", f"{file_prefix}.txt"
    )
    with open(yolo_label_path, "w") as yolo_label_file:
        for class_id, u, v, w, h in zip(class_ids, u_center, v_center, box_width, box_height):
            yolo_label_file.write(f"{class_id} {u} {v} {w} {h}\n")


def generate_yolo_label_cpu(
    region_id: str,
    split_dir_name: str,
    file_prefix: str,
    pixel_batch_size: int = 1000,
    dark_threshold: int = 10,
    min_data_threshold: float = 0.5,
    check_center: bool = True,
) -> None:
    """
    Generate a YOLO label .txt file for the specified region and file prefix (CPU version).

    The .txt file contains one line for each bounding box that is entirely contained within the image.
    The line is formatted as follows, with all coordinates normalized to the range [0, 1]:
    <class_id> <u_center> <v_center> <box_width> <box_height>

    :param region_id: The MGRS region ID to generate YOLO label files for.
    :param split_dir_name: The name of the split directory that the file prefix belongs to. Must be an element of
                           SPLIT_DIR_NAMES.
    :param file_prefix: The common prefix of the PNG and lat/lon .npz files to process and the .txt YOLO label file to
                        generate.
    :param pixel_batch_size: The number of pixels to process in each batch when finding the closest pixel to each
                             bounding box corner. Smaller values will use less memory but may be slower.
    :param dark_threshold: Pixels with sum of RGB below this are considered dark/invalid.
    :param min_data_threshold: Minimum fraction of bbox that must have valid pixels.
    :param check_center: Whether to check if bbox center is in a non-dark region.
    """
    assert split_dir_name in SPLIT_DIR_NAMES, f"Invalid split directory name: {split_dir_name}"

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    bounding_boxes_lat_lon = load_ground_truth(
        os.path.join(
            training_dir, get_region_bounding_boxes_relative_path(region_id)
        )
    )
    num_classes = bounding_boxes_lat_lon.shape[0]

    # CSV columns are: [centroid_lon, centroid_lat, tl_lon, tl_lat, br_lon, br_lat]
    # But lat_lon_to_ecef expects [lat, lon], so we need to swap the order
    top_left_lat_lon = bounding_boxes_lat_lon[:, [3, 2]]  # [tl_lat, tl_lon]
    bottom_right_lat_lon = bounding_boxes_lat_lon[:, [5, 4]]  # [br_lat, br_lon]
    top_right_lat_lon = np.column_stack((top_left_lat_lon[:, 0], bottom_right_lat_lon[:, 1]))  # [tl_lat, br_lon]
    bottom_left_lat_lon = np.column_stack((bottom_right_lat_lon[:, 0], top_left_lat_lon[:, 1]))  # [br_lat, tl_lon]

    stacked_corners_lat_lon = np.concatenate(
        # must be in a circular order for cv2.fillPoly to work correctly
        (top_left_lat_lon, top_right_lat_lon, bottom_right_lat_lon, bottom_left_lat_lon), axis=0
    )
    stacked_corners_ecef = lat_lon_to_ecef(stacked_corners_lat_lon)

    try:
        geotagged_image = GeotaggedImage.load(region_id, file_prefix, delete_bad_files=False)
    except AssertionError:
        # Skip dimension assertion for real satellite images - load manually
        training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
        region_dir = os.path.join(training_dir, region_id)

        img_path = os.path.join(region_dir, f"{file_prefix}.png")
        lat_lon_path = os.path.join(region_dir, f"{file_prefix}_lat_lon.npz")

        # Check if files exist
        if not os.path.exists(img_path) or not os.path.exists(lat_lon_path):
            log_message("WARNING", f"Missing file for: {region_id=}, {file_prefix=}.")
            return

        # Try to load image
        image = cv2.imread(img_path)
        if image is None:
            log_message("WARNING", f"Corrupted image file: {img_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            with np.load(lat_lon_path) as data:
                lat_lon = data["lat_lon"]
        except Exception as e:
            log_message("WARNING", f"Failed to load lat/lon for {file_prefix}: {e}")
            return

        # Create object without strict assertions
        class FlexibleGeotaggedImage:
            def __init__(self, img, ll):
                self.image = img
                self.lat_lon = ll

        geotagged_image = FlexibleGeotaggedImage(image, lat_lon)

    except Exception as e:
        log_message("WARNING", f"Failed to load geotagged image for: {region_id=}, {file_prefix=}. Error: {e}")
        return

    height, width = geotagged_image.image.shape[:2]
    pixel_coordinates_ecef = lat_lon_to_ecef(geotagged_image.lat_lon).reshape(-1, 3)

    closest_pixel_indices = np.empty(4 * num_classes, dtype=int)
    minimum_distances = np.full(4 * num_classes, np.inf)
    for start_pixel_idx in range(0, height * width, pixel_batch_size):
        end_pixel_idx = min(start_pixel_idx + pixel_batch_size, height * width)
        pixel_slice = slice(start_pixel_idx, end_pixel_idx)

        x_distances = np.subtract.outer(
            stacked_corners_ecef[:, 0], pixel_coordinates_ecef[pixel_slice, 0]
        )
        y_distances = np.subtract.outer(
            stacked_corners_ecef[:, 1], pixel_coordinates_ecef[pixel_slice, 1]
        )
        z_distances = np.subtract.outer(
            stacked_corners_ecef[:, 2], pixel_coordinates_ecef[pixel_slice, 2]
        )
        # surprisingly axis=0 is actually faster than axis=-1 here, probably because of the overhead in
        # creating the stacked array
        distances = np.linalg.norm(
            np.stack((x_distances, y_distances, z_distances), axis=0), axis=0
        )
        assert distances.shape == (4 * num_classes, end_pixel_idx - start_pixel_idx)

        batch_closest_pixel_indices = np.argmin(distances, axis=1)
        batch_minimum_distances = distances[np.arange(4 * num_classes), batch_closest_pixel_indices]

        closer_mask = batch_minimum_distances < minimum_distances
        closest_pixel_indices[closer_mask] = (
            batch_closest_pixel_indices[closer_mask] + start_pixel_idx
        )
        minimum_distances[closer_mask] = batch_minimum_distances[closer_mask]

    closest_vs, closest_us = np.unravel_index(closest_pixel_indices, (height, width))
    closest_us = closest_us.reshape(4, num_classes).T
    closest_vs = closest_vs.reshape(4, num_classes).T

    valid_bounding_boxes = get_valid_bounding_boxes(
        geotagged_image.image,
        closest_us,
        closest_vs,
        minimum_data_threshold=min_data_threshold,
        check_center=check_center,
        dark_threshold=dark_threshold,
    )
    closest_us = closest_us[valid_bounding_boxes, :]
    closest_vs = closest_vs[valid_bounding_boxes, :]
    class_ids = np.arange(num_classes)[valid_bounding_boxes]

    # widen bounding boxes to the smallest axis-aligned bounding box that contains all 4 corners
    min_us = np.min(closest_us, axis=1)
    max_us = np.max(closest_us, axis=1)
    min_vs = np.min(closest_vs, axis=1)
    max_vs = np.max(closest_vs, axis=1)

    # convert to YOLO format
    u_center = (min_us + max_us) / (2 * width)
    v_center = (min_vs + max_vs) / (2 * height)
    box_width = (max_us - min_us) / width
    box_height = (max_vs - min_vs) / height

    yolo_label_path = os.path.join(
        training_dir, region_id, LD_TRAINING_DIR_NAME, split_dir_name, "labels", f"{file_prefix}.txt"
    )
    with open(yolo_label_path, "w") as yolo_label_file:
        for class_id, u, v, w, h in zip(class_ids, u_center, v_center, box_width, box_height):
            yolo_label_file.write(f"{class_id} {u} {v} {w} {h}\n")


def main():
    args = parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")

    if args.use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU requested but CuPy is not available. Falling back to CPU.")
        args.use_gpu = False

    regions = sorted(set(args.regions) - set(args.skip_regions))

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    for region in tqdm(regions, desc="Setting up region directories"):
        if not setup_LD_training_directory(
            region, args.overwrite, args.resume, args.test_fraction, args.val_fraction
        ):
            print(
                f"Output directory for {region} could not be emptied. Set --overwrite to clear any existing data."
            )
            return

    def get_requests_generator() -> Generator[Tuple[str, str, str], None, None]:
        """
        :return: A generator that yields tuples of (region_id, split_dir_name, file_prefix) for each YOLO label file to
                 be generated.
        """
        for region in regions:
            LD_training_dir = os.path.join(training_dir, region, LD_TRAINING_DIR_NAME)
            for split_dir_name in SPLIT_DIR_NAMES:
                split_dir = os.path.join(LD_training_dir, split_dir_name)
                images_dir = os.path.join(split_dir, "images")

                file_prefixes_generator = (
                    file_name[: -len(".png")]
                    for file_name in os.listdir(images_dir)
                    if file_name.endswith(".png")
                )
                if args.resume:
                    labels_dir = os.path.join(split_dir, "labels")
                    file_prefixes_generator = (
                        file_prefix
                        for file_prefix in file_prefixes_generator
                        if not os.path.exists(os.path.join(labels_dir, f"{file_prefix}.txt"))
                    )

                yield from (
                    (region, split_dir_name, file_prefix) for file_prefix in file_prefixes_generator
                )

    # Choose the appropriate function based on GPU availability
    check_center = not args.no_center_check
    if args.use_gpu:
        print(f"Using GPU acceleration with batch size {args.gpu_batch_size}")
        print(f"Dark region filtering: center_check={check_center}, dark_threshold={args.dark_threshold}, min_data={args.min_data_threshold}")
        generate_func = partial(
            generate_yolo_label_gpu,
            gpu_batch_size=args.gpu_batch_size,
            dark_threshold=args.dark_threshold,
            min_data_threshold=args.min_data_threshold,
            check_center=check_center,
        )
    else:
        print("Using CPU computation")
        print(f"Dark region filtering: center_check={check_center}, dark_threshold={args.dark_threshold}, min_data={args.min_data_threshold}")
        generate_func = partial(
            generate_yolo_label_cpu,
            dark_threshold=args.dark_threshold,
            min_data_threshold=args.min_data_threshold,
            check_center=check_center,
        )

    if args.num_processes > 1:
        total_requests = sum(1 for _ in get_requests_generator())
        with Pool(args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(unpack_and_call, generate_func), get_requests_generator()
                    ),
                    total=total_requests,
                    desc="Generating YOLO label files",
                )
            )
    else:
        requests = list(get_requests_generator())
        for req in tqdm(requests, desc="Generating YOLO label files"):
            generate_func(*req)


if __name__ == "__main__":
    main()
