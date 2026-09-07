#!/usr/bin/env python3
"""
Batch convert PNG files to georeferenced GeoTIFFs for all MGRS regions
Saves to training_directory/{region_id}/blue_marble.tif as specified in user_config.yaml
"""

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import os
import glob
import sys
from pathlib import Path

# Import configuration utilities and earth utilities
from utils.config_utils import load_config, USER_CONFIG_PATH
from utils.earth_utils import get_MGRS_grid


def create_geotiff_from_image(input_image_path, region_id, bbox, training_directory):
    """
    Convert a manually downloaded image to georeferenced GeoTIFF
    
    Args:
        input_image_path: Path to downloaded PNG/JPEG
        region_id: MGRS region identifier (e.g., '05V')
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
        training_directory: Base training directory from config
    """
    # Create region-specific subdirectory
    region_dir = os.path.join(training_directory, region_id)
    os.makedirs(region_dir, exist_ok=True)
    
    # Output path is always blue_marble.tif in the region directory
    output_path = os.path.join(region_dir, 'blue_marble.tif')
    
    print(f"\nProcessing {region_id}:")
    print(f"  Input: {input_image_path}")
    print(f"  Output dir: {region_dir}")
    
    # Open image
    img = Image.open(input_image_path)
    
    # Convert to RGB if needed
    if img.mode == 'P':  # Palette mode
        img = img.convert('RGB')
    elif img.mode == 'RGBA':  # RGBA mode
        print("  Converting RGBA to RGB...")
        img = img.convert('RGB')
    elif img.mode != 'RGB' and img.mode != 'L':  # Not RGB or Grayscale
        print(f"  Converting {img.mode} to RGB...")
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    print(f"  Image shape: {img_array.shape}")
    print(f"  Image dtype: {img_array.dtype}")
    
    # Get dimensions
    if len(img_array.shape) == 3:
        height, width, channels = img_array.shape
        print(f"  Channels: {channels}")
    else:
        height, width = img_array.shape
        channels = 1
        print("  Grayscale image")
    
    # Calculate GSD
    center_lat = (bbox[1] + bbox[3]) / 2
    lon_extent = bbox[2] - bbox[0]
    lat_extent = bbox[3] - bbox[1]
    
    meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
    meters_per_degree_lat = 110540
    
    gsd_x = (lon_extent / width) * meters_per_degree_lon
    gsd_y = (lat_extent / height) * meters_per_degree_lat
    
    print(f"  Bounds: {bbox}")
    print(f"  Image size: {width} x {height} pixels")
    print(f"  GSD X: {gsd_x:.1f} m/pixel")
    print(f"  GSD Y: {gsd_y:.1f} m/pixel")
    
    # Create affine transform
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
    
    # Save as GeoTIFF
    print(f"  Saving: {output_path}")
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=channels,
        dtype=img_array.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        if channels == 1:
            dst.write(img_array, 1)
        else:
            for i in range(channels):
                dst.write(img_array[:, :, i], i + 1)
    
    # Verify
    with rasterio.open(output_path) as src:
        print(f"  ✓ Created GeoTIFF:")
        print(f"    CRS: {src.crs}")
        print(f"    Bounds: {src.bounds}")
        print(f"    Size: {src.width} x {src.height}")
        print(f"    Bands: {src.count}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    File size: {file_size:.2f} MB")
    
    return output_path


def process_all_regions(input_dir='downloaded_regions', config_path=USER_CONFIG_PATH):
    """
    Process all PNG files in the input directory
    
    Args:
        input_dir: Directory containing PNG files named like '05V.png'
        config_path: Path to user_config.yaml (defaults to USER_CONFIG_PATH from config_utils)
    """
    print("="*70)
    print("BATCH PROCESSING MGRS REGIONS")
    print("="*70)
    
    # Load configuration
    config = load_config(config_path)
    training_directory = config.get('training_directory')
    
    if not training_directory:
        print("✗ Error: 'training_directory' not found in user_config.yaml")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Training directory: {training_directory}")
    print(f"  Input directory: {input_dir}")
    
    # Get MGRS grid
    mgrs_grid = get_MGRS_grid()
    print(f"\nLoaded MGRS grid with {len(mgrs_grid)} regions")
    
    # Find all PNG files
    png_files = glob.glob(os.path.join(input_dir, '*.png'))
    
    if not png_files:
        print(f"\n✗ No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    # Process each PNG file
    successful = 0
    failed = 0
    skipped = 0
    
    for png_file in sorted(png_files):
        # Extract region ID from filename
        filename = os.path.basename(png_file)
        region_id = os.path.splitext(filename)[0]
        
        # Check if output already exists
        output_file = os.path.join(training_directory, region_id, 'blue_marble.tif')
        if os.path.exists(output_file):
            print(f"\n⊘ Skipping {region_id} (already exists): {output_file}")
            skipped += 1
            continue
        
        # Get bounding box from MGRS grid
        if region_id not in mgrs_grid:
            print(f"\n✗ Region {region_id} not found in MGRS grid")
            failed += 1
            continue
        
        bbox = mgrs_grid[region_id]
        
        # Process the image
        try:
            create_geotiff_from_image(png_file, region_id, bbox, training_directory)
            successful += 1
        except Exception as e:
            print(f"\n✗ Error processing {region_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Total: {len(png_files)}")
    print(f"\nOutput location: {training_directory}/{{region_id}}/blue_marble.tif")
    print("="*70 + "\n")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage:")
            print("  python convert_png_geotiff_all.py [input_directory] [config_file]")
            print("\nDefault input directory: 'downloaded_regions'")
            print("Default config file: user_config.yaml (from config_utils.USER_CONFIG_PATH)")
            print("\nThis script will:")
            print("  1. Read training_directory from user_config.yaml")
            print("  2. Find all PNG files in the input directory")
            print("  3. Extract the MGRS region ID from each filename (e.g., '05V.png' -> '05V')")
            print("  4. Look up the bounding box from the MGRS grid")
            print("  5. Create directory: training_directory/{region_id}/")
            print("  6. Convert each PNG to: training_directory/{region_id}/blue_marble.tif")
            print("\nExample directory structure:")
            print("  /mnt/sda2/training/")
            print("  ├── 05V/")
            print("  │   └── blue_marble.tif")
            print("  ├── 09V/")
            print("  │   └── blue_marble.tif")
            print("  └── 10S/")
            print("      └── blue_marble.tif")
            return
        else:
            input_dir = sys.argv[1]
            config_path = sys.argv[2] if len(sys.argv) > 2 else USER_CONFIG_PATH
    else:
        input_dir = 'downloaded_regions'
        config_path = USER_CONFIG_PATH
    
    if not os.path.exists(input_dir):
        print(f"✗ Error: Directory not found: {input_dir}")
        return
    
    process_all_regions(input_dir, config_path)


if __name__ == '__main__':
    main()
