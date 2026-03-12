# import gradio as gr
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# from pathlib import Path
# import os

# # Project paths
# WORK_DIR = Path(".")
# MODEL_DIR = WORK_DIR / "models"

# # -----------------------------
# # Load Model
# # -----------------------------
# def load_model():
#     model_path = MODEL_DIR / "license_plate_detector" / "weights" / "best.pt"

#     if model_path.exists():
#         print(f"✅ Loading trained model: {model_path}")
#         return YOLO(str(model_path))
#     else:
#         print("⚠️ Custom model not found. Using YOLOv8n default model.")
#         return YOLO("yolov8n.pt")

# model = load_model()


# # -----------------------------
# # Detection Function
# # -----------------------------
# def detect_and_draw(img_bgr, conf=0.5):

#     results = model(img_bgr, conf=conf)
#     result = results[0]

#     boxes = result.boxes
#     count = 0

#     if boxes is not None:
#         count = len(boxes)

#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             conf_score = float(box.conf[0])

#             cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(0,255,0),3)

#             label = f"Plate {conf_score:.2f}"
#             cv2.putText(
#                 img_bgr,
#                 label,
#                 (x1,y1-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0,255,0),
#                 2
#             )

#     log = f"Detected {count} license plate(s)"

#     img_bgr = cv2.copyMakeBorder(
#         img_bgr,10,10,10,10,
#         cv2.BORDER_CONSTANT,
#         value=(40,40,40)
#     )

#     cv2.putText(
#         img_bgr,
#         "YOLOv8 License Plate Detection",
#         (20,40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255,255,255),
#         2
#     )

#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), log


# # -----------------------------
# # Gradio Function
# # -----------------------------
# def gradio_fn(img_pil, conf):

#     if img_pil is None:
#         return None, "⚠️ Please upload an image"

#     np_img = np.array(img_pil)
#     bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

#     result_rgb, log = detect_and_draw(bgr.copy(), conf)

#     return Image.fromarray(result_rgb), log


# # -----------------------------
# # Custom CSS
# # -----------------------------
# custom_css = """
# body{
# background: linear-gradient(135deg,#020617,#0f172a,#1e293b);
# font-family: Arial;
# }

# .header{
# text-align:center;
# margin-bottom:20px;
# }

# .logo{
# width:70px;
# }

# .gr-button{
# background:linear-gradient(90deg,#22c55e,#16a34a);
# color:white;
# font-weight:bold;
# border:none;
# border-radius:10px;
# font-size:16px;
# }

# footer{
# text-align:center;
# margin-top:30px;
# color:#9ca3af;
# }
# """


# # -----------------------------
# # UI Layout
# # -----------------------------
# with gr.Blocks(css=custom_css) as demo:

#     gr.HTML("""
#     <div class="header">
#         <img class="logo" src="https://cdn-icons-png.flaticon.com/512/744/744465.png">
#         <h1 style="color:white;">AI License Plate Detection System</h1>
#         <p style="color:#9ca3af;">YOLOv8 Deep Learning Model</p>
#     </div>
#     """)

#     with gr.Group():
#         gr.Markdown("## 📤 Upload Vehicle Image")

#         input_img = gr.Image(type="pil", label="Upload Image")

#         conf_slider = gr.Slider(
#             0.25,
#             0.9,
#             value=0.5,
#             label="Detection Confidence"
#         )

#         detect_btn = gr.Button("🚀 Run Detection")


#     with gr.Group():
#         gr.Markdown("## 📊 Detection Result")

#         output_img = gr.Image(label="Detected Image")

#         output_log = gr.Textbox(label="System Log")


#     detect_btn.click(
#         fn=gradio_fn,
#         inputs=[input_img, conf_slider],
#         outputs=[output_img, output_log]
#     )

#     gr.HTML("""
#     <footer>
#     Developed for AI Vision Project • YOLOv8 • 2026
#     </footer>
#     """)


# # -----------------------------
# # Launch (Deployment Safe)
# # -----------------------------
# if __name__ == "__main__":

#     port = int(os.environ.get("PORT", 7860))

#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=port
#     )
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import os

# Project paths
WORK_DIR = Path(".")
MODEL_DIR = WORK_DIR / "models"

# -----------------------------
# Load Model
# -----------------------------
def load_model():
    model_path = MODEL_DIR / "license_plate_detector" / "weights" / "best.pt"

    if model_path.exists():
        print(f"Loading trained model: {model_path}")
        return YOLO(str(model_path))
    else:
        print("Custom model not found. Using YOLOv8n default model.")
        return YOLO("yolov8n.pt")

model = load_model()


# -----------------------------
# Detection Function
# -----------------------------
def detect_and_draw(img_bgr, conf=0.5):

    results = model(img_bgr, conf=conf)
    result = results[0]

    boxes = result.boxes
    count = 0

    if boxes is not None:
        count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])

            cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(0,255,0),3)

            label = f"Plate {conf_score:.2f}"
            cv2.putText(
                img_bgr,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    log = f"Detected {count} license plate(s)"

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), log


# -----------------------------
# Gradio Function
# -----------------------------
def gradio_fn(img_pil, conf):

    if img_pil is None:
        return None, "Please upload an image"

    np_img = np.array(img_pil)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    result_rgb, log = detect_and_draw(bgr.copy(), conf)

    return Image.fromarray(result_rgb), log


# -----------------------------
# Custom CSS (New UI Style)
# -----------------------------
custom_css = """
body{
background:#eef2ff;
font-family:'Segoe UI',sans-serif;
}

.header{
text-align:center;
padding:30px;
color:#1e293b;
}

.logo{
width:60px;
margin-bottom:10px;
}

.card{
background:white;
padding:25px;
border-radius:16px;
box-shadow:0 10px 25px rgba(0,0,0,0.08);
margin-bottom:25px;
}

.gr-button{
background:#2563eb;
color:white;
border:none;
border-radius:10px;
font-weight:600;
font-size:16px;
padding:10px;
}

.gr-button:hover{
background:#1d4ed8;
}

footer{
text-align:center;
margin-top:25px;
color:#64748b;
}
"""


# -----------------------------
# UI Layout
# -----------------------------
with gr.Blocks(css=custom_css) as demo:

    # Header
    gr.HTML("""
    <div class="header">
        <img class="logo" src="https://cdn-icons-png.flaticon.com/512/744/744465.png">
        <h1>License Plate Detection AI</h1>
        <p>Upload a vehicle image and detect license plates instantly</p>
    </div>
    """)

    # Upload Section
    with gr.Group(elem_classes="card"):

        gr.Markdown("### Upload Vehicle Image")

        input_img = gr.Image(type="pil", label="Vehicle Image")

        conf_slider = gr.Slider(
            0.25,
            0.9,
            value=0.5,
            label="Confidence Threshold"
        )

        detect_btn = gr.Button("Detect Plate")


    # Result Section
    with gr.Group(elem_classes="card"):

        gr.Markdown("### Detection Result")

        output_img = gr.Image(label="Detected Image")

        output_log = gr.Textbox(label="Detection Info")


    detect_btn.click(
        fn=gradio_fn,
        inputs=[input_img, conf_slider],
        outputs=[output_img, output_log]
    )

    # Footer
    gr.HTML("""
    <footer>
    License Plate Detection • YOLOv8 • AI Vision Project
    </footer>
    """)


# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )