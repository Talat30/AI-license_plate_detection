"""
YOLO License Plate Detection Model Training Script
This script trains a YOLOv8 model to detect license plates in images.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from ultralytics import YOLO
import torch

# Set working directory
WORK_DIR = Path("c:/Users/ADMIN/OneDrive/Desktop/insem lab obj")
DATA_DIR = WORK_DIR / "data"
MODEL_DIR = WORK_DIR / "models"

def download_sample_dataset():
    """
    Download a sample license plate dataset for training.
    Using a subset of CCPD dataset for demonstration.
    """
    print("Setting up dataset...")
    
    # Create train/val directories
    train_img_dir = DATA_DIR / "images" / "train"
    val_img_dir = DATA_DIR / "images" / "val"
    train_label_dir = DATA_DIR / "labels" / "train"
    val_label_dir = DATA_DIR / "labels" / "val"
    
    for directory in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have images
    existing_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    if len(existing_images) > 0:
        print(f"Found {len(existing_images)} existing training images.")
        return True
    
    print("Downloading sample license plate dataset...")
    
    # Download CCPD dataset (using a public subset)
    # Note: For production, you would use the full CCPD dataset
    url = "https://github.com/ultralytics/yolov8/releases/download/v1.0/car-plate.jpg"
    
    # For demonstration, we'll create sample synthetic data
    # In production, download full CCPD dataset
    print("Creating sample training data...")
    
    # Create a sample image with a license plate region
    # This is a placeholder - in practice, use real labeled data
    try:
        # Try to download a sample car image
        sample_url = "https://raw.githubusercontent.com/ultralytics/yolov8/main/tests/car.jpg"
        dest_path = train_img_dir / "sample_car.jpg"
        urllib.request.urlretrieve(sample_url, dest_path)
        
        # Create a label file (YOLO format: class x_center y_center width height)
        # For the sample car image, we'll assume the license plate is in the lower portion
        label_path = train_label_dir / "sample_car.txt"
        with open(label_path, 'w') as f:
            # License plate class (0), centered at bottom, normalized coordinates
            f.write("0 0.5 0.85 0.3 0.1\n")
        
        print(f"Downloaded sample image to {dest_path}")
        
        # Copy to validation set
        import shutil
        shutil.copy(dest_path, val_img_dir / "sample_car.jpg")
        shutil.copy(label_path, val_label_dir / "sample_car.txt")
        
        return True
        
    except Exception as e:
        print(f"Error downloading sample: {e}")
        return False


def create_synthetic_dataset():
    """
    Create synthetic training data for demonstration purposes.
    This generates random images with annotated license plate regions.
    """
    import cv2
    import numpy as np
    
    print("Creating synthetic training dataset...")
    
    train_img_dir = DATA_DIR / "images" / "train"
    val_img_dir = DATA_DIR / "images" / "val"
    train_label_dir = DATA_DIR / "labels" / "train"
    val_label_dir = DATA_DIR / "labels" / "val"
    
    # Create directories if they don't exist
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training images directory: {train_img_dir}")
    print(f"Training labels directory: {train_label_dir}")
    
    # Create 50 synthetic training images
    for i in range(50):
        # Create random car-like image (gray rectangle with some features)
        img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
        
        # Add some car-like features (rectangle in center)
        cv2.rectangle(img, (150, 200), (490, 450), (80, 80, 80), -1)
        
        # Add license plate region (bright rectangle at bottom)
        plate_x, plate_y = 220, 420
        plate_w, plate_h = 200, 60
        cv2.rectangle(img, (plate_x, plate_y), 
                     (plate_x + plate_w, plate_y + plate_h), 
                     (255, 255, 200), -1)
        
        # Add plate characters
        cv2.putText(img, "ABC-123", (plate_x + 30, plate_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save image
        img_path = train_img_dir / f"plate_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Convert to YOLO label format (normalized coordinates)
        x_center = (plate_x + plate_w/2) / 640
        y_center = (plate_y + plate_h/2) / 640
        width = plate_w / 640
        height = plate_h / 640
        
        label_path = train_label_dir / f"plate_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create 10 validation images
    for i in range(10):
        img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 200), (490, 450), (80, 80, 80), -1)
        
        plate_x, plate_y = 220 + np.random.randint(-30, 30), 420 + np.random.randint(-20, 20)
        plate_w, plate_h = 200, 60
        cv2.rectangle(img, (plate_x, plate_y), 
                     (plate_x + plate_w, plate_y + plate_h), 
                     (255, 255, 200), -1)
        cv2.putText(img, "XYZ-456", (plate_x + 30, plate_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        img_path = val_img_dir / f"plate_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        x_center = (plate_x + plate_w/2) / 640
        y_center = (plate_y + plate_h/2) / 640
        width = plate_w / 640
        height = plate_h / 640
        
        label_path = val_label_dir / f"plate_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created 50 training and 10 validation images.")
    return True


def train_model():
    """
    Train the YOLOv8 model for license plate detection.
    """
    print("\n" + "="*50)
    print("Starting YOLO License Plate Detection Training")
    print("="*50 + "\n")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create synthetic dataset if no real data exists
    train_images = list((DATA_DIR / "images" / "train").glob("*.jpg"))
    if len(train_images) < 10:
        create_synthetic_dataset()
    
    # Load YOLOv8 nano model (smallest, fastest)
    print("\nLoading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Load pretrained model
    
    # Train the model
    print("\nTraining model...")
    results = model.train(
        data=str(WORK_DIR / "data_config.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=str(MODEL_DIR),
        name='license_plate_detector',
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Model saved to: {MODEL_DIR / 'license_plate_detector'}")
    
    return model, results


if __name__ == "__main__":
    # Run training
    model, results = train_model()
    
    print("\nTraining completed successfully!")
    print(f"Best model path: {MODEL_DIR / 'license_plate_detector' / 'weights' / 'best.pt'}")

