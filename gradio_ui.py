import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

WORK_DIR = Path(".")
MODEL_DIR = WORK_DIR / "models"

def load_model():
    model_path = MODEL_DIR / "license_plate_detector" / "weights" / "best.pt"
    if model_path.exists():
        print(f"Loading trained model: {model_path}")
    else:
        print("Using YOLOv8n")
        model_path = "yolov8n.pt"
    return YOLO(str(model_path))

def detect_and_draw(img_bgr, model, conf=0.5):
    results = model(img_bgr, conf=conf)
    result = results[0]
    boxes = result.boxes
    n = 0
    if boxes is not None:
        n = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = box.conf[0].item()
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f'Plate {conf_score:.2f}'
            cv2.putText(img_bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    log = f'Detected {n} license plate(s)'
    border = 10
    img_bgr = cv2.copyMakeBorder(img_bgr, border, border, border, border, cv2.BORDER_CONSTANT, (100,100,100))
    cv2.putText(img_bgr, 'License Plate Detection - YOLO', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), log

def gradio_fn(img_pil, conf):
    if img_pil is None:
        return None, "Upload a vehicle image"
    model = load_model()
    np_img = np.array(img_pil)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    result_rgb, log = detect_and_draw(bgr.copy(), model, conf)
    return Image.fromarray(result_rgb), log

demo = gr.Interface(
    fn=gradio_fn,
    inputs=[gr.Image(type="pil"), gr.Slider(0.25, 0.9, 0.5, label="Conf Threshold")],
    outputs=[gr.Image(), gr.Textbox()],
    title="License Plate Detector",
    description="Upload vehicle photo to detect license plates"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
