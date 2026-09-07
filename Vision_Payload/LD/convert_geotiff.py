"""
Convert GeoTIFF files to PNG and NPZ format for YOLO training.

This script converts satellite imagery GeoTIFFs to the format expected by prepare_yolo_data.py:
- {prefix}.png - RGB image (8-bit)
- {prefix}_lat_lon.npz - Array of lat/lon coordinates for each pixel

Usage:
    python convert_geotiff.py --input_dir /path/to/geotiffs --output_dir /path/to/output
    python convert_geotiff.py --input_file /path/to/image.tif --output_dir /path/to/output
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import rasterio
from rasterio.transform import xy
from PIL import Image
from tqdm import tqdm


def convert_geotiff(input_path: str, output_dir: str, normalize: bool = True) -> bool:
    """
    Convert a GeoTIFF file to PNG and NPZ format.

    :param input_path: Path to the input GeoTIFF file.
    :param output_dir: Directory to save the output files.
    :param normalize: Whether to normalize the image to 0-255 range.
    :return: True if conversion was successful, False otherwise.
    """
    try:
        with rasterio.open(input_path) as src:
            # Read image data
            # GeoTIFFs can have various band configurations
            num_bands = src.count

            if num_bands >= 3:
                # Read RGB bands (usually bands 1, 2, 3 for RGB or R, G, B order)
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                image = np.stack([r, g, b], axis=-1)
            elif num_bands == 1:
                # Grayscale - replicate to 3 channels
                gray = src.read(1)
                image = np.stack([gray, gray, gray], axis=-1)
            else:
                print(f"Warning: Unexpected number of bands ({num_bands}) in {input_path}")
                return False

            # Get image dimensions
            height, width = image.shape[:2]

            # Generate lat/lon coordinates for each pixel
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

            # Convert pixel coordinates to geographic coordinates
            # xy() returns (x, y) which is (lon, lat) for geographic CRS
            xs, ys = xy(src.transform, rows.flatten(), cols.flatten())

            # Check if CRS is geographic (lat/lon) or projected
            if src.crs and src.crs.is_geographic:
                lons = np.array(xs).reshape(height, width)
                lats = np.array(ys).reshape(height, width)
            else:
                # If projected CRS, we need to transform to lat/lon
                from rasterio.warp import transform
                from rasterio.crs import CRS

                lons, lats = transform(src.crs, CRS.from_epsg(4326), xs, ys)
                lons = np.array(lons).reshape(height, width)
                lats = np.array(lats).reshape(height, width)

            # Stack lat/lon into the expected format (height, width, 2)
            lat_lon = np.stack([lats, lons], axis=-1)

            # Normalize image to 8-bit if needed
            if normalize:
                if image.dtype != np.uint8:
                    # Check the data range and normalize accordingly
                    img_min = np.min(image)
                    img_max = np.max(image)

                    if img_max > 255 or image.dtype in [np.float32, np.float64]:
                        # Normalize to 0-255
                        if img_max > img_min:
                            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            image = np.zeros_like(image, dtype=np.uint8)
                    else:
                        image = image.astype(np.uint8)

            # Generate output file names
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            png_path = os.path.join(output_dir, f"{base_name}.png")
            npz_path = os.path.join(output_dir, f"{base_name}_lat_lon.npz")

            # Save PNG
            pil_image = Image.fromarray(image)
            pil_image.save(png_path)

            # Save lat/lon as NPZ
            np.savez_compressed(npz_path, lat_lon=lat_lon)

            return True

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def convert_single(args_tuple):
    """Wrapper for multiprocessing."""
    input_path, output_dir, normalize = args_tuple
    return convert_geotiff(input_path, output_dir, normalize)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF files to PNG and NPZ format for YOLO training."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing GeoTIFF files to convert."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single GeoTIFF file to convert."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save converted PNG and NPZ files."
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip normalization (assume image is already 8-bit)."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(0.8 * cpu_count()),
        help="Number of parallel processes for batch conversion."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for finding GeoTIFF files (default: *.tif)."
    )

    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        parser.error("Either --input_dir or --input_file must be specified.")

    os.makedirs(args.output_dir, exist_ok=True)
    normalize = not args.no_normalize

    if args.input_file:
        # Single file conversion
        success = convert_geotiff(args.input_file, args.output_dir, normalize)
        if success:
            print(f"Successfully converted {args.input_file}")
        else:
            print(f"Failed to convert {args.input_file}")
    else:
        # Batch conversion
        import glob

        pattern = os.path.join(args.input_dir, args.pattern)
        tif_files = glob.glob(pattern)

        # Also check for .tiff extension
        if args.pattern == "*.tif":
            tif_files.extend(glob.glob(os.path.join(args.input_dir, "*.tiff")))

        if not tif_files:
            print(f"No GeoTIFF files found matching {pattern}")
            return

        print(f"Found {len(tif_files)} GeoTIFF files to convert")

        if args.num_processes > 1 and len(tif_files) > 1:
            # Parallel processing
            tasks = [(f, args.output_dir, normalize) for f in tif_files]
            with Pool(args.num_processes) as pool:
                results = list(tqdm(
                    pool.imap(convert_single, tasks),
                    total=len(tif_files),
                    desc="Converting GeoTIFFs"
                ))

            success_count = sum(results)
            print(f"Successfully converted {success_count}/{len(tif_files)} files")
        else:
            # Sequential processing
            success_count = 0
            for tif_file in tqdm(tif_files, desc="Converting GeoTIFFs"):
                if convert_geotiff(tif_file, args.output_dir, normalize):
                    success_count += 1
            print(f"Successfully converted {success_count}/{len(tif_files)} files")


if __name__ == "__main__":
    main()
