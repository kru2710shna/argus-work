# Landmark Detection Training Pipeline

Complete workflow for training YOLO landmark detection models on satellite imagery.

## Environment

```bash
conda activate sunac-eedl
```

## Configuration

Edit `user_config.yaml`:
```yaml
training_directory: /mnt/sda2/training/
geotiff_folder: /mnt/sda2/geotiffs/
earth_engine_project_name: argus-cubesat
```

## Complete Workflow

### Step 1: Blue Marble Base Images

**Already done.** Cloud-free, picture-perfect Earth images should be in `training_directory` from `user_config.yaml`. One per MGRS region.

Location: `training_directory/{REGION}/blue_marble.tif`

---

### Step 2: Download Training Images

Download 500+ mosaic images using Earth Engine.

**Required: Mosaic Images**
```bash
cd images/

# Edit download_mosaics.sh with your region
MGRS_REGION="05V"
NUM_IMAGES=500

# Run download
bash download_mosaics.sh

# Monitor at: https://code.earthengine.google.com/tasks
```

**Optional: Composite Images** (for secondary testing)
```bash
# Edit download_composites.sh
MGRS_REGION="05V"
NUM_IMAGES=25

# Run download
bash download_composites.sh
```

Images export to Google Drive, then manually move to `training_directory`.

---

### Step 3: Extract Salient Features (Run While Waiting for Downloads)

While images are downloading (12-24 hours), extract salient features.

```bash
cd LD/

# For most regions
python run_saliency_analysis.py --regions 05V --top_n 75

# For snowy regions (like 05V)
python run_saliency_analysis.py --regions 05V --top_n 75 --use_glacier_detection
```

**Important:** Manually verify `bounding_boxes.png` to confirm salient features look correct.

Location: Check output in `training_directory/{REGION}/`

---

### Step 4: Move Training Images

Once downloads complete:

```bash
# Move from Google Drive to geotiff_folder
mv ~/GoogleDrive/05V_images/*.tif /mnt/sda2/geotiffs/05V/
```

Note: Move to `geotiff_folder` (not `training_directory`).

---

### Step 5: Prepare YOLO Dataset

```bash
cd LD/

python prepare_yolo_data.py --regions 05V --overwrite
```

Creates YOLO training dataset in `training_directory/{REGION}/LD_training/`

---

### Step 6: Train Model

```bash
cd LD/

python train_yolo.py --regions 05V --epochs 100 --version yolov8s
```

Trained model saved to: `training_directory/{REGION}/yolo_model_weights.pt`

---

## Quick Reference

```bash
# Complete pipeline for region 05V
conda activate sunac-eedl

# Step 1: Already done (blue marble images)

# Step 2: Download training images
cd images/
bash download_mosaics.sh

# Step 3: Extract salient features (while waiting)
cd ../LD/
python run_saliency_analysis.py --regions 05V --top_n 75 --use_glacier_detection

# Verify bounding_boxes.png manually

# Step 4: Move downloaded images (after download completes)
mv ~/GoogleDrive/05V_images/*.tif /mnt/sda2/geotiffs/05V/

# Step 5: Prepare dataset
python prepare_yolo_data.py --regions 05V --overwrite

# Step 6: Train
python train_yolo.py --regions 05V --epochs 100
```

## Directory Structure

```
/mnt/sda2/
├── training/                           # training_directory
│   └── 05V/
│       ├── blue_marble/                # Step 1 (already done)
│       ├── LD_training/                # Step 5 (created)
│       └── yolo_model_weights.pt       # Step 6 (output)
│
└── geotiffs/                           # geotiff_folder
    └── 05V/                            # Step 4 (move images here)
        ├── l8_05V_00000.tif
        └── ... (500+ files)
```

## Notes

- Use `--use_glacier_detection` for snowy regions (like 05V)
- Always verify `bounding_boxes.png` before proceeding
- Downloads take 12-24 hours
- Move images to `geotiff_folder`, not `training_directory`
- Composites are optional, for secondary testing only
