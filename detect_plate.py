"""
License Plate Detection Inference Script
This script loads a trained YOLO model and detects license plates in images,
drawing bounding boxes around detected plates.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import argparse
import gradio as gr
from PIL import Image
import io

# Set working directory
WORK_DIR = Path("c:/Users/ADMIN/OneDrive/Desktop/insem lab obj")
MODEL_DIR = WORK_DIR / "models"


def load_model(model_path=None):
    """
    Load the trained YOLO model.
    """
    if model_path is None:
        model_path = MODEL_DIR / "license_plate_detector" / "weights" / "best.pt"
        
        if not model_path.exists():
            print("No trained model found. Using pretrained YOLOv8 model for demo.")
            model_path = "yolov8n.pt"
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    return model


def detect_and_draw_boxes(img_input, model, conf_threshold=0.5, save_path=None):
    """
    Detect license plates in an image and draw bounding boxes.
    Accepts either str path or np.array img.
    """
    if isinstance(img_input, str):
        img = cv2.imread(str(img_input))
    else:
        img = img_input.copy()
    
    if img is None:
        print(f"Error: Could not read image")
        return None
    
    original_height, original_width = img.shape[:2]
    
    # Run detection
    results = model(img, conf=conf_threshold)
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) > 0:
        print(f"Detected {len(boxes)} license plate(s)")
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            color = (0, 255, 0)
            thickness = 3
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"License Plate: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            print(f"  Plate {i+1}: Conf={conf:.3f}, Box=[{x1},{y1},{x2},{y2}]")
    else:
        print("No license plates detected in the image.")
    
    border_color = (100, 100, 100)
    border_thickness = 10
    img = cv2.copyMakeBorder(img, border_thickness, border_thickness, 
                             border_thickness, border_thickness, 
                             cv2.BORDER_CONSTANT, value=border_color)
    
    title = "License Plate Detection - YOLO"
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved result to: {save_path}")
    
    return img


def detect_np_array(np_img, model, conf_threshold=0.5):
    \"\"\"
    Detect on numpy array image for Gradio.
    \"\"\"
    result_img = detect_and_draw_boxes(np_img, model, conf_threshold)
    detections = []
    if result_img is not None:
        # Extract detections from model run (run again? use boxes if needed)
        results = model(np_img, conf=conf_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                detections.append(f"Plate {i+1}: {conf:.3f}")
    log = f"Detected {len(detections)} plates:\\n" + "\\n".join(detections) if detections else "No plates detected."
    return result_img, log


def create_sample_test_image():
    """
    Create a sample test image with a simulated license plate.
    """
    import random
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    cv2.rectangle(img, (100, 200), (540, 480), (100, 100, 100), -1)
    cv2.rectangle(img, (150, 220), (280, 300), (50, 150, 200), -1)
    cv2.rectangle(img, (320, 220), (490, 300), (50, 150, 200), -1)
    
    cv2.circle(img, (180, 480), 50, (30, 30, 30), -1)
    cv2.circle(img, (460, 480), 50, (30, 30, 30), -1)
    
    plate_x = random.randint(200, 350)
    plate_y = random.randint(380, 450)
    plate_w = random.randint(150, 200)
    plate_h = random.randint(40, 70)
    
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), 
                 (255, 255, 200), -1)
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), 
                 (0, 0, 0), 2)
    
    cv2.putText(img, "ABC-1234", (plate_x + 10, plate_y + plate_h//2 + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img, (plate_x, plate_y, plate_w, plate_h)


def main():
    """
    Main function for license plate detection.
    """
    print("="*60)
    print("  License Plate Detection with YOLO")
    print("="*60)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nProcessing image: {image_path}")
        
        model = load_model()
        
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        
        result_img = detect_and_draw_boxes(image_path, model, save_path=output_path)
        
        if result_img is not None:
            print(f"\nDetection complete! Output saved to: {output_path}")
            cv2.imshow("License Plate Detection", result_img)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to process image.")
            
    else:
        print("\nNo image path provided. Running demo with sample image...")
        
        sample_img, plate_box = create_sample_test_image()
        sample_path = WORK_DIR / "sample_test_image.jpg"
        cv2.imwrite(str(sample_path), sample_img)
        
        print(f"Created sample test image: {sample_path}")
        
        model = load_model()
        
        output_path = WORK_DIR / "sample_test_image_detected.jpg"
        result_img = detect_and_draw_boxes(sample_path, model, save_path=output_path)
        
        if result_img is not None:
            print(f"\nDemo complete! Output saved to: {output_path}")
            cv2.imshow("License Plate Detection - Demo", result_img)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

