# Earth Engine Image Download Pipeline

This directory contains scripts for downloading satellite imagery from Google Earth Engine for training landmark detection models and other computer vision tasks.

## Overview

Two main scripts for downloading different types of satellite imagery:

1. **eedl.py** - Download individual mosaic images (500+ recommended per region)
2. **composite-eedl-mgrs.py** - Download composite images (10-50 per region for verification)

## Prerequisites

### Environment Setup

Activate the required conda environment:
```bash
conda activate sunac-eedl
```

### Configuration File

**IMPORTANT**: Before running any download scripts, verify your configuration file:

```bash
# Check your configuration
cat /path/to/user_config.yaml
```

**Key Configuration Parameters:**

```yaml
# Path to geotiff files (for image simulation and reference data)
geotiff_folder: /mnt/sda2/geotiffs/

# Earth Engine project name
earth_engine_project_name: argus-cubesat

# Training directory (where downloaded images are stored for training)
training_directory: /mnt/sda2/training/

# Models directory (where trained models are saved/loaded)
models_directory: /home/argus/sunac/Vision_Payload/training_directory

# Output directory (for results and analysis)
output_directory: /home/argus/sunac/Vision_Payload
```

### Understanding Directory Structure

**Where do downloaded images go?**

- **Mosaic Images** (from `eedl.py`): 
  - Downloaded to google drive
  - Purpose: Raw training data for model training
  - Use: Primary dataset for landmark detection
  - Download from google drive to geotiffs directory 

- **Composite Images** (from `composite-eedl-mgrs.py`):
  - Downloaded to google drive
  - Purpose: Clean, cloud-free reference images
  - Use: Verification, validation, and quality assessment
  - Download from google drive to geotiffs directory 

- **GeoTIFF Folder** (`geotiff_folder`):
  - Used for: Image simulation and reference data
  - Needs to be manually managed

**Confirmation**: Downloaded images go to google drive, NOT `geotiff_folder` or `training_directory`.

## Download Workflow

### Step 1: Download Mosaic Images (Primary Dataset)

Download at least **500 mosaic images** per MGRS region for training.

```bash
# Run the mosaic download script
bash download_mosaics.sh
```

**What this does:**
- Downloads 500+ individual satellite images
- Images are mosaics from different dates
- Used as primary training dataset
- Includes natural variation in cloud cover, lighting, seasons

**Script Configuration** (`download_mosaics.sh`):

```bash
#!/bin/bash
# download_mosaics.sh

# Configuration
MGRS_REGION="09V"              # MGRS grid region
START_DATE="2017"              # Start date (YYYY or YYYY-MM-DD)
END_DATE="2024"                # End date (YYYY or YYYY-MM-DD)
NUM_IMAGES=10                  # Number of images (increase to 500+)
SENSOR="l8"                    # Sensor: l8, l9, or s2
GSD=175.0                      # Ground Sample Distance in meters
WIDTH_PIXELS=4608              # Image width in pixels
HEIGHT_PIXELS=2592             # Image height in pixels
OUTPUT_DIR="09V_images"        # Output directory on Google Drive

# Run the downloader with pixel dimensions
python eedl.py \
    -g "$MGRS_REGION" \
    -i "$START_DATE" \
    -f "$END_DATE" \
    -m "$NUM_IMAGES" \
    -se "$SENSOR" \
    -s "$GSD" \
    -wp "$WIDTH_PIXELS" \
    -hp "$HEIGHT_PIXELS" \
    -o "$OUTPUT_DIR" \
    -cm True \
    -gd True \
    -ba B4 B3 B2

echo ""
echo "Tasks submitted to Google Earth Engine!"
echo "Check status at: https://code.earthengine.google.com/tasks"
```

**Key Parameters:**
- `-g`: MGRS grid region (e.g., 05V, 09V)
- `-i`, `-f`: Date range for imagery
- `-m`: Number of images (set to 500+)
- `-se`: Sensor (l8=Landsat 8, l9=Landsat 9, s2=Sentinel-2)
- `-s`: Ground Sample Distance (GSD) in meters
- `-wp`, `-hp`: Image dimensions in pixels
- `-o`: Output directory name
- `-cm`: Custom mosaics (True)
- `-gd`: Google Drive export (True)
- `-ba`: Bands (B4 B3 B2 = RGB)

---

### Step 2: Download Composite Images (Verification Dataset)

Download **10-50 composite images** for verification and quality assessment.

```bash
# Run the composite download script
bash download_composites.sh
```

**What this does:**
- Downloads cloud-free composite images
- Images are temporally aggregated (cleaner, less variation)
- Used for verification and validation
- Helps assess model performance on ideal conditions

**Script Configuration** (`download_composites.sh`):

```bash
#!/bin/bash
# download_composites.sh

# Configuration
MGRS_REGION="09V"                   # MGRS grid region
START_DATE="2017"                   # Start date for imagery
END_DATE="2024"                     # End date for imagery
NUM_IMAGES=10                       # Number of composite images (10-50 recommended)
SENSOR="l8"                         # Sensor: l8 or l9
GSD=175.0                           # Ground Sample Distance in meters
WIDTH_PIXELS=4608                   # Image width in pixels
HEIGHT_PIXELS=2592                  # Image height in pixels
OUTPUT_DIR="09V_composite_images"   # Output directory on Google Drive

# Run the composite downloader
python composite-eedl-mgrs.py \
    -g "$MGRS_REGION" \
    -i "$START_DATE" \
    -f "$END_DATE" \
    -np "$NUM_IMAGES" \
    -se "$SENSOR" \
    -s "$GSD" \
    -wp "$WIDTH_PIXELS" \
    -hp "$HEIGHT_PIXELS" \
    -o "$OUTPUT_DIR" \
    -ba B4 B3 B2

echo ""
echo "Tasks submitted to Google Earth Engine!"
echo "Check status at: https://code.earthengine.google.com/tasks"
echo "Once complete, images will be in Google Drive folder: $OUTPUT_DIR"
```

**Key Parameters:**
- `-g`: MGRS grid region
- `-i`, `-f`: Date range for composite
- `-np`: Number of points/images (10-50)
- `-se`: Sensor (l8 or l9)
- `-s`: Ground Sample Distance in meters
- `-wp`, `-hp`: Image dimensions in pixels
- `-o`: Output directory name
- `-ba`: Bands (RGB)

## 📊 Complete Download Pipeline

### For a New Region (e.g., 05V)

```bash
# 1. Update configuration in download scripts
# Edit download_mosaics.sh
MGRS_REGION="05V"
NUM_IMAGES=500
OUTPUT_DIR="05V_images"

# Edit download_composites.sh
MGRS_REGION="05V"
NUM_IMAGES=25
OUTPUT_DIR="05V_composite_images"

# 2. Download mosaic images (primary dataset)
bash download_mosaics.sh

# 3. Download composite images (verification)
bash download_composites.sh

# 4. Monitor progress
# Go to: https://code.earthengine.google.com/tasks

# 5. Once complete, verify downloads
ls ~/GoogleDrive/05V_images/ | wc -l          # Should show ~500
ls ~/GoogleDrive/05V_composite_images/ | wc -l  # Should show ~25
```

### Processing Multiple Regions

```bash
# Create a batch script
cat > download_all_regions.sh << 'EOF'
#!/bin/bash
REGIONS=("05V" "04V" "06V" "09V")

for REGION in "${REGIONS[@]}"; do
    echo "Processing region: $REGION"
    
    # Update and run mosaic download
    sed -i "s/MGRS_REGION=\".*\"/MGRS_REGION=\"$REGION\"/" download_mosaics.sh
    sed -i "s/OUTPUT_DIR=\".*\"/OUTPUT_DIR=\"${REGION}_images\"/" download_mosaics.sh
    bash download_mosaics.sh
    
    # Update and run composite download
    sed -i "s/MGRS_REGION=\".*\"/MGRS_REGION=\"$REGION\"/" download_composites.sh
    sed -i "s/OUTPUT_DIR=\".*\"/OUTPUT_DIR=\"${REGION}_composite_images\"/" download_composites.sh
    bash download_composites.sh
    
    echo "Tasks submitted for region $REGION"
    echo "---"
done

echo "All regions processed!"
EOF

chmod +x download_all_regions.sh
bash download_all_regions.sh
```

## ⚙️ Parameter Reference

### Image Dimensions

**Pixel Dimensions** (`-wp`, `-hp`):
- Defines exact output image size
- Default: 4608 × 2592 pixels

**Ground Sample Distance** (`-s`, GSD):
- 175m: Default (and represents the actual camera specs) 

### Sensor Options

| Sensor | Description | Bands | Resolution |
|--------|-------------|-------|------------|
| l8 | Landsat 8 | B4,B3,B2 (RGB) | 30m native |
| l9 | Landsat 9 | B4,B3,B2 (RGB) | 30m native |
| s2 | Sentinel-2 | B4,B3,B2 (RGB) | 10m native |

**Note**: GSD parameter determines final resolution through resampling.

### Date Ranges

**Format Options:**
- Year only: `"2017"` to `"2024"`
- Full date: `"2017-01-01"` to `"2024-12-31"`
- Month: `"2017-06"` to `"2024-06"`

**Recommendations:**
- **Training**: Longer date ranges (5-10 years) for diversity
- **Composites**: Seasonal or annual composites work well

## Recommended Settings

## Script Reference

### eedl.py Parameters

```
-g, --grid_key: MGRS region (e.g., 05V)
-i, --idate: Start date (YYYY or YYYY-MM-DD)
-f, --fdate: End date (YYYY or YYYY-MM-DD)
-m, --maxims: Number of images to download
-se, --sensor: Sensor (l8, l9, s2)
-s, --scale: Ground Sample Distance in meters
-wp, --width_pixels: Image width in pixels
-hp, --height_pixels: Image height in pixels
-o, --outpath: Output directory name
-cm, --custom_mosaics: Use custom mosaics (True/False)
-gd, --gdrive: Export to Google Drive (True/False)
-ba, --bands: Spectral bands (e.g., B4 B3 B2)
-c, --crs: Coordinate reference system (default: EPSG:4326)
```

### composite-eedl-mgrs.py Parameters

```
-g, --grid_key: MGRS region
-i, --idate: Start date
-f, --fdate: End date
-np, --num_points: Number of composite images
-se, --sensor: Sensor (l8, l9)
-s, --scale: GSD in meters
-wp, --width_pixels: Image width in pixels
-hp, --height_pixels: Image height in pixels
-o, --outpath: Output directory name
-ba, --bands: Spectral bands
-p, --percentile: Composite percentile (default: 50)
-csr, --cloud_score_range: Cloud score range (default: 10)
-md, --max_depth: Maximum depth for composite (default: 40)
```

## 💡 Tips

1. **Start Small**: Test with 10 images before downloading 500+
2. **Monitor Storage**: Large downloads require significant space
3. **Use Composites Wisely**: Great for verification, not primary training
4. **Date Ranges**: Longer ranges = more variety = better training
5. **GSD Selection**: Balance between detail and coverage area
6. **Batch Processing**: Process multiple regions sequentially
7. **Backup**: Keep copies of downloaded images

## 🆘 Getting Help

### Resources

- Earth Engine Tasks: https://code.earthengine.google.com/tasks
- Earth Engine Docs: https://developers.google.com/earth-engine
- Landsat Info: https://www.usgs.gov/landsat-missions
- MGRS Grid: https://mgrs-mapper.com/

### Before Asking for Help

1. Check Earth Engine tasks page for error messages
2. Verify configuration file is correct
3. Confirm authentication is working
4. Test with small batch (10 images)
5. Check logs for specific error messages

---

**Last Updated**: January 2026  
**Conda Environment**: sunac-eedl  
**Earth Engine Project**: argus-cubesat
