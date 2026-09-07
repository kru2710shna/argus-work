"""                                                                                                                                                                                                                
Train landmark detection YOLO models for the specified MGRS regions (standalone version).                                                                                                                          

This script expects to find the following contents in the training directory:
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

This script will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - yolo_model_weights.pt
    - yolo_training_results_{region}_{timestamp}/
      - ...
"""

import argparse
import os
from time import time

import torch
from tqdm import tqdm
from ultralytics import YOLO

from utils.config_utils import USER_CONFIG_PATH, load_config

# Constants (copied from prepare_yolo_data_gpu.py to avoid dependencies)
LD_TRAINING_DIR_NAME = "LD_training"
YOLO_CONFIG_FILE_NAME = "dataset.yaml"
TRAINING_LOG_DIR_PREFIX = "yolo_training_results"
MODEL_WEIGHTS_FILE_NAME = "yolo_model_weights.pt"


def parse_args():
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train landmark detection YOLO models for the specified MGRS regions."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to train landmark detection YOLO models for.",
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
        help="Whether to overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="yolov8s",
        help="The YOLO version to use (default: yolov8s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size for training (default: 2).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=4608,
        help="Image size for training - images will be padded to square (default: 4608).",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=10.0,
        help="Rotation augmentation in degrees (default: 10).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    return parser.parse_args()


def train_yolo(
    training_dir: str,
    region: str,
    overwrite: bool,
    version: str,
    epochs: int,
    batch: int,
    imgsz: int,
    degrees: float,
    resume: bool,
) -> None:
    """
    Train a YOLO model for a specific region.

    :param training_dir: Path to the training directory.
    :param region: The MGRS region to train the model for.
    :param overwrite: Whether to overwrite the output files if they exist.
    :param version: The YOLO model version to use.
    :param epochs: The number of epochs for training.
    :param batch: Batch size for training.
    :param imgsz: Image size for training.
    :param degrees: Rotation augmentation in degrees.
    :param resume: Whether to resume training from the last checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device={device} for training region={region}")

    region_dir = os.path.join(training_dir, region)
    output_file = os.path.join(region_dir, MODEL_WEIGHTS_FILE_NAME)

    # Check if output already exists
    if os.path.exists(output_file):
        if not overwrite:
            raise FileExistsError(f"Output file {output_file} already exists. Use --overwrite to replace.")
        os.remove(output_file)

    # Verify dataset.yaml exists
    yolo_config_path = os.path.join(region_dir, LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME)
    if not os.path.exists(yolo_config_path):
        raise FileNotFoundError(
            f"YOLO config not found at {yolo_config_path}. "
            f"Run prepare_yolo_data_gpu.py first to generate training data."
        )

    # Load YOLO model
    model = YOLO(f"{version}.pt")

    # Train the model
    results = model.train(
        data=yolo_config_path,
        # The result files are saved relative to current directory
        project=f"{TRAINING_LOG_DIR_PREFIX}_{region}",
        name=f"{TRAINING_LOG_DIR_PREFIX}_{region}_{int(time())}",
        # Image augmentation parameters
        degrees=degrees,  # rotation augmentation for non-nadir camera angles
        scale=0,
        fliplr=0,
        mosaic=0,
        perspective=0,
        # Training parameters
        imgsz=imgsz,
        rect=True,  # rectangular training (preserves aspect ratio)
        batch=batch,
        plots=True,
        save=True,
        resume=resume,
        epochs=epochs,
        device=device,
        patience=0,  # Disable early stopping
        conf=0.5,  # Higher confidence threshold to reduce false positives
    )

    # Copy the final weights to the expected output location
    # Using last.pt since custom ultralytics selects "best" based on buggy MSE metric
    import shutil
    last_weights = results.save_dir / "weights" / "last.pt"
    if last_weights.exists():
        shutil.copy(last_weights, output_file)
        print(f"Saved last weights to {output_file}")
    else:
        print(f"Warning: Could not find trained weights in {results.save_dir}")


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]

    # Validate training directory exists
    if not os.path.isdir(training_dir):
        raise NotADirectoryError(f"Training directory does not exist: {training_dir}")

    regions = sorted(set(args.regions) - set(args.skip_regions))

    if not regions:
        print("No regions to train. Exiting.")
        return

    print(f"Training YOLO models for {len(regions)} region(s): {regions}")
    print(f"Model: {args.version}, Epochs: {args.epochs}, Batch: {args.batch}, Image size: {args.imgsz}")

    for region in tqdm(regions, desc="Training YOLO models"):
        try:
            train_yolo(
                training_dir=training_dir,
                region=region,
                overwrite=args.overwrite,
                version=args.version,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                degrees=args.degrees,
                resume=args.resume,
            )
        except FileExistsError as e:
            print(f"Skipping {region}: {e}")
            continue
        except FileNotFoundError as e:
            print(f"Skipping {region}: {e}")
            continue
        except Exception as e:
            print(f"Error training YOLO model for {region}: {e}")
            continue


if __name__ == "__main__":
    main()
