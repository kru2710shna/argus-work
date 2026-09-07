#!/usr/bin/env python3
"""
Saliency Analysis with Glacier Filtering and Bounding Box Generation

Computes saliency maps from Blue Marble imagery, filters glaciers using brightness
and saturation thresholds, and generates non-overlapping bounding boxes.

Outputs:
- /training_directory/{region}/saliency_map.tif
- /training_directory/{region}/glacier_mask.tif
- /training_directory/{region}/bounding_boxes.csv
- /training_directory/{region}/bounding_boxes.png
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rasterio

from utils.config_utils import USER_CONFIG_PATH, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Saliency analysis with glacier filtering and bounding box generation"
    )
    
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        required=True,
        help="MGRS regions to process",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--glacier_threshold",
        type=float,
        default=0.7,
        help="Brightness threshold for glaciers (0-1)",
    )
    parser.add_argument(
        "--saturation_threshold",
        type=float,
        default=0.15,
        help="Max saturation for glaciers (0-1, lower=whiter)",
    )
    parser.add_argument(
        "--no_glacier_filter",
        action="store_true",
        help="Disable glacier filtering",
    )
    parser.add_argument(
        "--buffer_pixels",
        type=int,
        default=50,
        help="Buffer zone around glaciers in pixels",
    )
    parser.add_argument(
        "--num_boxes",
        type=int,
        default=100,
        help="Number of bounding boxes to generate",
    )
    parser.add_argument(
        "--min_box_size",
        type=int,
        default=15,
        help="Minimum box size in pixels",
    )
    parser.add_argument(
        "--max_box_size",
        type=int,
        default=75,
        help="Maximum box size in pixels",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.05,
        help="Max IoU overlap for boxes (0-1)",
    )
    parser.add_argument(
        "--enable_coastline_detection",
        action="store_true",
        help="Enable coastline detection to boost saliency near water/land boundaries",
    )
    parser.add_argument(
        "--coastline_strength",
        type=float,
        default=0.5,
        help="Strength of coastline boost (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.8 * cpu_count()),
        help="Number of parallel processes",
    )
    
    return parser.parse_args()


def compute_saliency(rgb_image: np.ndarray) -> np.ndarray:
    """
    Compute saliency map using OpenCV Fine-Grained algorithm.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
    
    Returns:
        Saliency map as float array (H, W) with values in [0, 1]
    """
    saliency_computer = cv2.saliency.StaticSaliencyFineGrained_create()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, saliency_map = saliency_computer.computeSaliency(bgr_image)
    
    if not success:
        raise RuntimeError("Failed to compute saliency map")
    
    return saliency_map


def detect_glaciers(
    rgb_image: np.ndarray,
    brightness_threshold: float = 0.7,
    saturation_threshold: float = 0.15,
    buffer_pixels: int = 50
) -> np.ndarray:
    """
    Detect glaciers using brightness + low saturation (whiteness).
    
    Glaciers are WHITE (high brightness + low saturation), not just bright.
    This avoids false positives from deserts, beaches, and sand.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
        brightness_threshold: Minimum brightness (0-1)
        saturation_threshold: Maximum saturation (0-1)
        buffer_pixels: Buffer zone around glaciers (pixels)
    
    Returns:
        Boolean array (H, W), True = glacier or buffer zone
    """
    # Normalize to [0, 1]
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    
    # Calculate brightness and saturation
    brightness = rgb_norm.max(axis=2)
    rgb_min = rgb_norm.min(axis=2)
    rgb_max = rgb_norm.max(axis=2)
    
    saturation = np.zeros_like(brightness)
    mask = rgb_max > 0
    saturation[mask] = (rgb_max[mask] - rgb_min[mask]) / rgb_max[mask]
    
    # Glaciers = HIGH brightness AND LOW saturation
    glacier_mask = (brightness > brightness_threshold) & (saturation < saturation_threshold)
    
    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    glacier_mask = cv2.morphologyEx(glacier_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    glacier_mask = cv2.morphologyEx(glacier_mask, cv2.MORPH_OPEN, kernel)
    
    # Add buffer zone
    if buffer_pixels > 0:
        buffer_kernel = np.ones((buffer_pixels * 2, buffer_pixels * 2), np.uint8)
        glacier_mask = cv2.dilate(glacier_mask, buffer_kernel, iterations=1)
    
    return glacier_mask.astype(bool)


def apply_glacier_mask(saliency_map: np.ndarray, glacier_mask: np.ndarray) -> np.ndarray:
    """
    Set glacier regions to zero in saliency map.
    
    Args:
        saliency_map: Float array (H, W) with values in [0, 1]
        glacier_mask: Boolean array (H, W), True = glacier
    
    Returns:
        Filtered saliency map with glaciers set to 0
    """
    filtered_map = saliency_map.copy()
    filtered_map[glacier_mask] = 0.0
    return filtered_map


def detect_coastlines(
    rgb_image: np.ndarray,
    water_brightness_threshold: float = 0.3,
    edge_strength: float = 0.5
) -> np.ndarray:
    """
    Detect coastlines by finding edges between water and land.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
        water_brightness_threshold: Max brightness for water detection (0-1)
        edge_strength: Strength of coastline boost (0-1)
    
    Returns:
        Float array (H, W) with coastline edges emphasized
    """
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Detect water (dark regions with low brightness)
    brightness = rgb_image.astype(np.float32).mean(axis=2) / 255.0
    water_mask = brightness < water_brightness_threshold
    
    # Find edges using Canny
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    edges_float = edges.astype(np.float32) / 255.0
    
    # Dilate water mask slightly to catch nearby edges
    kernel = np.ones((5, 5), np.uint8)
    water_dilated = cv2.dilate(water_mask.astype(np.uint8), kernel, iterations=2)
    
    # Keep only edges near water (coastlines)
    coastline_edges = edges_float * water_dilated.astype(np.float32)
    
    # Dilate coastline edges to make them more prominent
    coastline_kernel = np.ones((3, 3), np.uint8)
    coastline_edges = cv2.dilate(coastline_edges, coastline_kernel, iterations=2)
    
    # Scale by edge strength
    return coastline_edges * edge_strength


def combine_saliency_with_coastlines(
    saliency_map: np.ndarray,
    coastline_map: np.ndarray
) -> np.ndarray:
    """
    Combine saliency map with coastline detection.
    
    Args:
        saliency_map: Float array (H, W) with values in [0, 1]
        coastline_map: Float array (H, W) with coastline edges
    
    Returns:
        Combined map with coastlines boosted
    """
    combined = saliency_map + coastline_map
    # Clip to [0, 1] range
    combined = np.clip(combined, 0.0, 1.0)
    return combined


def filter_overlapping_boxes(
    boxes_with_scores: List[Tuple[Tuple[int, int, int, int], float]],
    overlap_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """
    Non-Maximum Suppression to filter overlapping boxes.
    
    Args:
        boxes_with_scores: List of ((x, y, w, h), score) tuples
        overlap_threshold: Maximum allowed IoU (0-1)
    
    Returns:
        List of non-overlapping boxes
    """
    if len(boxes_with_scores) == 0:
        return []
    
    boxes_with_scores.sort(key=lambda x: x[1], reverse=True)
    selected_boxes = []
    
    for (box, score) in boxes_with_scores:
        x1, y1, w1, h1 = box
        overlaps = False
        
        for (x2, y2, w2, h2) in selected_boxes:
            # Calculate IoU
            ix = max(x1, x2)
            iy = max(y1, y2)
            iw = min(x1 + w1, x2 + w2) - ix
            ih = min(y1 + h1, y2 + h2) - iy
            
            if iw > 0 and ih > 0:
                intersection = iw * ih
                union = w1 * h1 + w2 * h2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > overlap_threshold:
                    overlaps = True
                    break
        
        if not overlaps:
            selected_boxes.append(box)
    
    return selected_boxes


def subdivide_large_region(
    x: int, y: int, w: int, h: int, target_size: int = 40
) -> List[Tuple[int, int, int, int]]:
    """
    Subdivide large region into smaller boxes.
    
    Args:
        x, y, w, h: Original bounding box
        target_size: Target size for subdivided boxes
    
    Returns:
        List of (x, y, w, h) tuples for subdivided boxes
    """
    boxes = []
    num_x = max(1, w // target_size)
    num_y = max(1, h // target_size)
    box_w = w // num_x
    box_h = h // num_y
    
    for i in range(num_x):
        for j in range(num_y):
            sub_x = x + i * box_w
            sub_y = y + j * box_h
            boxes.append((sub_x, sub_y, box_w, box_h))
    
    return boxes


def generate_bounding_boxes(
    saliency_map: np.ndarray,
    glacier_mask: np.ndarray,
    num_boxes: int = 100,
    min_box_size: int = 15,
    max_box_size: int = 75,
    overlap_threshold: float = 0.05
) -> List[Tuple[int, int, int, int]]:
    """
    Generate bounding boxes from saliency map, filtering glacier overlaps.
    
    Args:
        saliency_map: Float array (H, W) with values in [0, 1]
        glacier_mask: Boolean array (H, W), True = glacier zone
        num_boxes: Number of boxes to generate
        min_box_size: Minimum box size in pixels
        max_box_size: Maximum box size in pixels
        overlap_threshold: Maximum IoU for NMS
    
    Returns:
        List of (x, y, w, h) tuples for top-N non-glacier boxes
    """
    # Find connected components
    threshold = 0.2
    binary_map = (saliency_map > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    
    # Collect valid boxes with scores
    box_data = []
    
    for i in range(1, num_labels):  # Skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        min_dim = min(w, h)
        max_dim = max(w, h)
        
        # Skip if too small
        if min_dim < min_box_size:
            continue
        
        # Subdivide if too large
        if w > max_box_size or h > max_box_size:
            sub_boxes = subdivide_large_region(x, y, w, h, target_size=max_box_size // 2)
            for (sub_x, sub_y, sub_w, sub_h) in sub_boxes:
                # Check glacier overlap
                box_region = glacier_mask[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
                if box_region.any():
                    continue
                # Score sub-box
                sub_region_mask = saliency_map[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
                mean_score = sub_region_mask.mean()
                box_data.append(((sub_x, sub_y, sub_w, sub_h), mean_score))
            continue
        
        # Skip extreme aspect ratios
        aspect_ratio = max_dim / max(min_dim, 1)
        if aspect_ratio > 10:
            continue
        
        # Check valid size range
        if not ((min_box_size <= w <= max_box_size) or (min_box_size <= h <= max_box_size)):
            continue
        
        # Check glacier overlap
        box_region = glacier_mask[y:y+h, x:x+w]
        if box_region.any():
            continue
        
        # Score box
        region_mask = (labels == i)
        mean_score = saliency_map[region_mask].mean()
        box_data.append(((x, y, w, h), mean_score))
    
    # Sort and filter overlaps
    box_data.sort(key=lambda x: x[1], reverse=True)
    filtered_boxes = filter_overlapping_boxes(box_data, overlap_threshold=overlap_threshold)
    
    return filtered_boxes[:num_boxes]


def save_bounding_boxes_csv(
    boxes: List[Tuple[int, int, int, int]],
    reference_tif: str,
    output_path: str
) -> None:
    """
    Save bounding boxes to CSV with lat/lon coordinates.
    
    Args:
        boxes: List of (x, y, w, h) tuples in pixel coordinates
        reference_tif: Path to reference GeoTIFF
        output_path: Output CSV file path
    """
    with rasterio.open(reference_tif) as src:
        transform = src.transform
        bbox_data = []
        
        for (x, y, w, h) in boxes:
            centroid_x, centroid_y = x + w / 2, y + h / 2
            
            # Convert to lat/lon
            centroid_lon, centroid_lat = transform * (centroid_x, centroid_y)
            top_left_lon, top_left_lat = transform * (x, y)
            bottom_right_lon, bottom_right_lat = transform * (x + w, y + h)
            
            bbox_data.append([
                centroid_lat, centroid_lon,
                top_left_lat, top_left_lon,
                bottom_right_lat, bottom_right_lon
            ])
    
    bbox_array = np.array(bbox_data)
    np.savetxt(
        output_path,
        bbox_array,
        delimiter=',',
        header='Centroid Latitude,Centroid Longitude,Top-Left Latitude,Top-Left Longitude,Bottom-Right Latitude,Bottom-Right Longitude',
        comments=''
    )


def create_bounding_boxes_png(
    saliency_map: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    output_path: str
) -> None:
    """
    Create PNG visualization of boxes on saliency map.
    
    Args:
        saliency_map: Saliency map array (H, W)
        boxes: List of (x, y, w, h) tuples
        output_path: Output PNG file path
    """
    visualization = np.rint(255 * saliency_map).astype(np.uint8)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
    
    for (x, y, w, h) in boxes:
        cv2.rectangle(
            visualization,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color=(255, 0, 0),  # Blue boxes (BGR)
            thickness=2
        )
    
    cv2.imwrite(output_path, visualization)


def save_geotiff(
    data: np.ndarray,
    reference_tif: str,
    output_path: str,
    dtype: str = 'float32'
) -> None:
    """Save data as GeoTIFF using reference for georeferencing."""
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=dtype, compress='lzw')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            if dtype == 'uint8':
                dst.write(data.astype(np.uint8), 1)
            else:
                dst.write(data.astype(np.float32), 1)


def process_region(
    region: str,
    overwrite: bool,
    glacier_threshold: Optional[float],
    saturation_threshold: float,
    buffer_pixels: int,
    num_boxes: int,
    min_box_size: int,
    max_box_size: int,
    overlap_threshold: float,
    no_glacier_filter: bool,
    enable_coastline_detection: bool = False,
    coastline_strength: float = 0.5
) -> None:
    """Process a single MGRS region."""
    config = load_config(USER_CONFIG_PATH)
    training_dir = config['training_directory']
    
    # Build paths
    region_dir = os.path.join(training_dir, region)
    input_tif = os.path.join(region_dir, 'blue_marble.tif')
    output_saliency = os.path.join(region_dir, 'saliency_map.tif')
    output_glacier = os.path.join(region_dir, 'glacier_mask.tif')
    output_csv = os.path.join(region_dir, 'bounding_boxes.csv')
    output_png = os.path.join(region_dir, 'bounding_boxes.png')
    
    # Check inputs
    if not os.path.exists(input_tif):
        print(f"ERROR: {region}: Input not found: {input_tif}")
        return
    
    if os.path.exists(output_saliency) and not overwrite:
        print(f"Skipping {region}: Outputs exist (use --overwrite)")
        return
    
    print(f"Processing {region}...")
    
    # Read RGB
    with rasterio.open(input_tif) as src:
        rgb = src.read()
        rgb = np.transpose(rgb, (1, 2, 0))
    
    # Compute saliency
    saliency_map = compute_saliency(rgb)
    
    # Detect coastlines if enabled
    if enable_coastline_detection:
        coastline_map = detect_coastlines(rgb, edge_strength=coastline_strength)
        saliency_map = combine_saliency_with_coastlines(saliency_map, coastline_map)
        coastline_pixels = (coastline_map > 0).sum()
        print(f"  {region}: Coastline pixels: {coastline_pixels:,}")
    
    # Detect glaciers
    if no_glacier_filter:
        glacier_mask = np.zeros(rgb.shape[:2], dtype=bool)
    else:
        glacier_mask = detect_glaciers(
            rgb, glacier_threshold, saturation_threshold, buffer_pixels
        )
        glacier_percent = 100 * glacier_mask.mean()
        print(f"  {region}: Glaciers: {glacier_percent:.1f}%")
    
    # Apply glacier mask
    filtered_saliency = apply_glacier_mask(saliency_map, glacier_mask)
    
    # Generate bounding boxes
    boxes = generate_bounding_boxes(
        saliency_map, glacier_mask, num_boxes,
        min_box_size, max_box_size, overlap_threshold
    )
    
    print(f"  {region}: Boxes: {len(boxes)}/{num_boxes}")
    
    # Save outputs
    save_geotiff(filtered_saliency, input_tif, output_saliency, dtype='float32')
    
    if not no_glacier_filter:
        save_geotiff(glacier_mask, input_tif, output_glacier, dtype='uint8')
    
    save_bounding_boxes_csv(boxes, input_tif, output_csv)
    create_bounding_boxes_png(filtered_saliency, boxes, output_png)
    
    print(f"  {region}: Complete")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    
    regions = sorted(set(args.regions) - set(args.skip_regions))
    args.num_processes = min(args.num_processes, len(regions))
    
    print("="*70)
    print("Saliency Analysis with Glacier Filtering")
    print("="*70)
    print(f"Regions: {len(regions)}")
    if args.no_glacier_filter:
        print(f"Glacier filtering: DISABLED")
    else:
        print(f"Glacier filtering: ENABLED")
        print(f"  Brightness: >{args.glacier_threshold}")
        print(f"  Saturation: <{args.saturation_threshold}")
        print(f"  Buffer: {args.buffer_pixels} px")
    if args.enable_coastline_detection:
        print(f"Coastline detection: ENABLED")
        print(f"  Strength: {args.coastline_strength}")
    print(f"Bounding boxes: {args.num_boxes}")
    print(f"Box size: {args.min_box_size}-{args.max_box_size} px")
    print(f"Processes: {args.num_processes}")
    print("="*70)
    
    func = partial(
        process_region,
        overwrite=args.overwrite,
        glacier_threshold=args.glacier_threshold,
        saturation_threshold=args.saturation_threshold,
        buffer_pixels=args.buffer_pixels,
        num_boxes=args.num_boxes,
        min_box_size=args.min_box_size,
        max_box_size=args.max_box_size,
        overlap_threshold=args.overlap_threshold,
        no_glacier_filter=args.no_glacier_filter,
        enable_coastline_detection=args.enable_coastline_detection,
        coastline_strength=args.coastline_strength
    )
    
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            pool.map(func, regions)
    else:
        list(map(func, regions))
    
    print("="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
