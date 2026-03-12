# License Plate Detection with YOLO

A complete YOLO-based license plate detection system that detects and draws bounding boxes around license plates in images.

## Project Structure

```
insem lab obj/
├── data_config.yaml          # YOLO dataset configuration
├── train_model.py           # Training script
├── detect_plate.py          # Inference/detection script
├── README.md                # This file
├── data/                    # Dataset directory
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   └── labels/             # YOLO format annotations
│       ├── train/
│       └── val/
└── models/                  # Trained models
    └── license_plate_detector/
        └── weights/
            └── best.pt     # Best trained model
```

## Quick Start

### Option 1: Run Demo (No Training Required)

```bash
python detect_plate.py
```

This will create a sample test image and run detection on it using a pretrained YOLO model.

### Option 2: Train Your Own Model

```bash
python train_model.py
```

This will:
1. Create synthetic training data (50 images for training, 10 for validation)
2. Train YOLOv8 nano model for 50 epochs
3. Save the trained model to `models/license_plate_detector/weights/best.pt`

### Option 3: Detect on Your Own Image

```bash
python detect_plate.py path/to/your/image.jpg
```

This will:
1. Load the trained model
2. Detect license plates in the image
3. Draw bounding boxes around detected plates
4. Save the result with `_detected` suffix

## Requirements

All requirements are already installed:
- Python 3.12.8
- PyTorch 2.2.2
- Ultralytics 8.3.220
- OpenCV Python
- Pillow

## Usage Examples

### Basic Detection

```python
from detect_plate import load_model, detect_and_draw_boxes
import cv2

# Load model
model = load_model()

# Detect and draw bounding boxes
result = detect_plate("path/to/image.jpg", model)

# Save result
cv2.imwrite("output.jpg", result)
```

### Using YOLO Directly

```python
from ultralytics import YOLO

# Load model
model = YOLO("models/license_plate_detector/weights/best.pt")

# Run inference
results = model("path/to/image.jpg", conf=0.5)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        print(f"Confidence: {box.conf[0]:.3f}")
        print(f"Box: {box.xyxy[0]}")
```

## Training Customization

To customize training, modify parameters in `train_model.py`:

```python
results = model.train(
    data="data_config.yaml",
    epochs=50,           # Increase for better accuracy
    imgsz=640,           # Image size
    batch=16,            # Batch size (reduce if out of memory)
    device="cuda",       # or "cpu"
    project="models",
    name="license_plate_detector",
    exist_ok=True,
    verbose=True,
    plots=True,
    save=True,
)
```

## For Better Results

1. **Use Real Data**: Replace synthetic data with real license plate images
2. **More Training Data**: Use CCPD dataset (~200k images)
3. **More Epochs**: Train for 100+ epochs for production
4. **Larger Model**: Try yolov8s, yolov8m for better accuracy

## CCPD Dataset

For production use, download the full CCPD dataset:

```bash
# CCPD dataset download (requires ~500MB)
# Visit: https://github.com/ultralytics/yolov8/releases/download/v1.0/
```

## Output

The detection script outputs:
- Green bounding boxes around detected license plates
- Confidence score label on each box
- Saves result as `{original_name}_detected.jpg`

