# LD Pipeline - Quick Reference Card

## Quick Start (3 Steps)

```bash
# Activate environment
conda activate sunac-eedl

# Step 1: Extract salient features (50-100 features recommended)
python run_saliency_analysis.py --regions 05V --top_n 75

# Step 2: Prepare YOLO dataset
python prepare_yolo_data.py --regions 05V --overwrite

# Step 3: Train YOLO model
python train_yolo.py --regions 05V --epochs 100 --version yolov8s
```

## Common Commands

### Single Region Pipeline
```bash
REGION="05V"
python run_saliency_analysis.py --regions $REGION --top_n 75
python prepare_yolo_data.py --regions $REGION --overwrite
python train_yolo.py --regions $REGION --epochs 100
```

### Multiple Regions Pipeline
```bash
REGIONS="05V 04V 06V"
python run_saliency_analysis.py --regions $REGIONS --top_n 50
python prepare_yolo_data.py --regions $REGIONS --overwrite
python train_yolo.py --regions $REGIONS --epochs 100 --overwrite
```

## Key Parameters

### Saliency Analysis
- `--top_n`: Number of features (50-100 recommended)

### YOLO Training
- `--version`: Model size (yolov8n/s/m/l/x)
  - **yolov8s**: Default, good balance
  - **yolov8m**: More accuracy, slower (not recommended)
  - **yolov8n**: Faster, less accurate
- `--epochs`: Training iterations (100 default)
- `--overwrite`: Replace existing files

## 📁 Output Locations

```
{training_directory}/{region}/
├── LD_training/dataset.yaml          # YOLO config (from prepare)
├── yolo_model_weights.pt             # Final model (from train)
└── yolo_training_results_{region}/   # Training logs
```

