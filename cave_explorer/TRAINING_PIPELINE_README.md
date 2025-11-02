# Training Pipeline Setup (Following Easha's Pattern)

This document explains how to set up the complete training pipeline that connects Task 1 (dataset collection) with Task 2 (artifact detection using trained model).

## Overview

The code now follows the same pattern as Easha's implementation:

1. **Task 1**: Collects images during exploration and saves them to `/home/student/ros_ws/dataset_images/`
2. **Task 2**: Uses a trained YOLOv5 model (trained on Task 1 data) for artifact detection

## Setup Instructions

### 1. Install Dependencies

```bash
# Install PyTorch
pip install torch torchvision

# Clone YOLOv5
cd /home/student/ros_ws/
git clone https://github.com/ultralytics/yolov5.git CV_Model/yolov5
cd CV_Model/yolov5
pip install -r requirements.txt
```

### 2. Collect Dataset (Task 1)

Run your robot to collect images:
```bash
# The robot will automatically save images every 3 seconds to:
# /home/student/ros_ws/dataset_images/
```

### 3. Label Dataset

You need to manually label the collected images:

**Option A: Using LabelImg (Recommended - Always Works)**
```bash
# LabelImg is already installed from source at /home/student/ros_ws/labelImg/
# Run LabelImg from source (this version works properly)
cd /home/student/labelImg
python3 labelImg.py /home/student/ros_ws/dataset_images/
```
**LabelImg Instructions:**
1. Open LabelImg with the command above
2. Click "Open Dir" and select `/home/student/ros_ws/dataset_images/`
3. Change save format to "YOLO" (not PascalVOC) - this is important!
4. For each image, draw bounding boxes around artifacts
5. Label each box with exact names: green_crystal, green_alien, stop_sign, mushrooms, formation, ice_wall, white_sphere
6. Save each annotation (creates .txt files automatically)
7. After labeling all images, you'll have .txt files next to each .jpg file

**Option B: Using Roboflow (If Available)**
Roboflow interface varies by account type. Try these navigation paths:
- Look for: **Annotate** → **Export** or **Download**
- Or: **Projects** → [Your Project] → **Generate** → **Export**
- Or: Main project page → **Export** button
- Or: **Versions** → **Export**

If Roboflow doesn't work easily, **use LabelImg instead** - it's more reliable and works offline.

### 4. Organize Dataset for Training

**If you used LabelImg:**
You need to organize your data into train/valid splits and create a data.yaml file.

```bash
# Create the dataset structure
cd /home/student/ros_ws
mkdir -p labeled_dataset/{train,valid}/{images,labels}

# Move ~80% of images to train, ~20% to valid (adjust numbers as needed)
# Example: if you have 100 images (000-099), put 000-079 in train, 080-099 in valid
cp dataset_images/cave_image_000*.jpg labeled_dataset/train/images/
cp dataset_images/cave_image_000*.txt labeled_dataset/train/labels/
cp dataset_images/cave_image_008*.jpg labeled_dataset/valid/images/
cp dataset_images/cave_image_008*.txt labeled_dataset/valid/labels/
cp dataset_images/cave_image_009*.jpg labeled_dataset/valid/images/
cp dataset_images/cave_image_009*.txt labeled_dataset/valid/labels/
```

**Create data.yaml file:**
```bash
cat > /home/student/ros_ws/labeled_dataset/data.yaml << EOF
# Dataset configuration for YOLOv5
path: /home/student/ros_ws/labeled_dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')

# Classes
nc: 6  # number of classes
names: ['green_crystal', 'green_alien', 'stop_sign', 'mushrooms', 'formation', 'white_sphere']
EOF
```

### 5. Train YOLOv5 Model

**Important: Use the virtual environment Python to avoid NumPy compatibility issues**

```bash
cd /home/student/ros_ws/CV_Model/yolov5

# Install YOLOv5 requirements in virtual environment (one-time setup)
/home/student/ros_ws/src/cave_explorer/.venv/bin/pip install -r requirements.txt

# Train the model using your labeled dataset (use virtual environment Python)
# Use smaller batch size if you get memory errors
/home/student/ros_ws/src/cave_explorer/.venv/bin/python train.py --img 640 --batch 8 --epochs 300 --data /home/student/ros_ws/labeled_dataset/data.yaml --weights yolov5s.pt

# After training, copy the best model (check which exp folder was created):
# List the experiment folders to see which one was created:
ls runs/train/

# Copy from the correct experiment folder (e.g., if you see exp2):
cp runs/train/exp2/weights/best.pt ./best.pt

# Or if you see exp:
cp runs/train/exp/weights/best.pt ./best.pt
```

**Training notes:**
- Reduce `--batch` to 8 or 4 if you get out-of-memory errors
- Increase `--epochs` to 200+ for better results if you have time
- The model will be saved as `best.pt` and `last.pt` in the `runs/train/exp/weights/` folder

### 6. Verify Setup

Your directory structure should look like:
```
/home/student/ros_ws/
├── dataset_images/           # Raw images from Task 1
├── labeled_dataset/          # Organized dataset for training
│   ├── data.yaml            # Dataset config (class names, paths)
│   ├── train/images/        # 80% of your labeled images
│   ├── train/labels/        # Corresponding .txt label files
│   ├── valid/images/        # 20% of your labeled images  
│   └── valid/labels/        # Corresponding .txt label files
├── CV_Model/
│   └── yolov5/
│       ├── best.pt          # Trained model
│       └── runs/train/exp/  # Training results
└── src/cave_explorer/
    └── cave_explorer/
        └── cave_explorer.py  # Uses trained model in Task 2
```

## Code Behavior

### Class Name Integration:
The artifact labels you use in Roboflow must match exactly what's defined in the code:
```python
# In cave_explorer.py line ~87:
artifact_labels = ['green_crystal', 'green_alien', 'stop_sign', 'mushrooms', 'formation', 'ice_wall', 'white_sphere']
```

**Important:** The order in your `data.yaml` should match this order, or update the code accordingly.

### With Trained Model Available:
- Task 2 will use the trained YOLOv5 model (like Easha's code)
- Higher accuracy and better detection
- Falls back to traditional CV methods if model fails

### Without Trained Model:
- Task 2 uses traditional computer vision methods
- Color detection, edge detection, template matching
- Lower accuracy but still functional

## Model Training Tips

1. **Collect diverse images**: Different lighting, angles, distances
2. **Label accurately**: Use tight bounding boxes
3. **Balance dataset**: Equal examples of each artifact type
4. **Augmentation**: YOLOv5 applies augmentation automatically
5. **Validation**: Keep 20% of data for validation

## Troubleshooting

### LabelImg issues:
- **✅ FIXED**: LabelImg is now installed from source and working properly
- **To run**: `cd /home/student/ros_ws/labelImg && python3 labelImg.py /home/student/ros_ws/dataset_images/`
- Make sure to select "YOLO" format (not PascalVOC) before starting
- Each .jpg file should have a corresponding .txt file with the same name
- If no .txt files are created, check that you selected YOLO format

### Only seeing "aggregated results" and "individual results" options:
- You're trying to export from **Workflow** or batch job results (wrong place)
- Go to: **Dataset** → **Versions** → **Export** (correct place)
- Workflow results are predictions from running a model, not training data
- You need the labeled dataset, not prediction results

### "Could not load custom model" error:
- Check if `/home/student/ros_ws/CV_Model/yolov5/best.pt` exists
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Check YOLOv5 installation in the correct path

### NumPy compatibility error during training:
- **Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
- **Fix**: Downgrade NumPy to version 1.x: `pip3 install "numpy<2"`
- **Alternative**: Upgrade matplotlib: `pip3 install --upgrade matplotlib`
- This is a common issue with YOLOv5 and NumPy 2.x compatibility

### Class name mismatch errors:
- Ensure your Roboflow labels match exactly: green_crystal, green_alien, stop_sign, mushrooms, formation, ice_wall, white_sphere
- Check the `data.yaml` file class names section
- Verify the order matches the code (or update `artifact_labels` in cave_explorer.py)

### Training killed/out of memory errors:
- **Error**: Training shows "Killed" message or stops unexpectedly
- **Fix**: Reduce batch size: use `--batch 8` or `--batch 4` instead of `--batch 16`
- **Alternative**: Reduce image size: use `--img 320` instead of `--img 640`
- This is common on systems with limited RAM (less than 8GB available)

### Poor detection performance:
- Collect more training images (500+ per class recommended)
- Improve labeling quality
- Train for more epochs
- Adjust confidence threshold in code

## Integration with Original Functionality

The code maintains backward compatibility:
- If no trained model is available, it uses traditional CV methods
- All original detection methods are preserved as fallbacks
- Performance should be equal or better than before
