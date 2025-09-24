import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# Load YOLO model
# -----------------------------

# -----------------------------
# Config โมเดลบน Hugging Face
# -----------------------------
HF_REPO_ID = "Phum52321/detection-app"
HF_FILENAME = "best.pt"

def load_model():
    weights = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return YOLO(weights)

_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

# -----------------------------
# Inference
# -----------------------------
def detect_image(image: np.ndarray, img_size=640, conf=0.25, iou=0.45):
    if image is None:
        return None
    model = get_model()
    results = model.predict(source=image, imgsz=img_size, conf=conf, iou=iou, verbose=False)
    return results[0].plot()

# -----------------------------
# Build Gradio UI
# -----------------------------
with gr.Blocks(title="Detection App") as demo:
    gr.Markdown(
        """
        # 🚀 Detection
        - **Upload Image**: เลือกอัปโหลดไฟล์รูป หรือใช้ **กล้องมือถือถ่ายตรงนี้ได้เลย**
        """
    )

    img_in = gr.Image(type="numpy", label="Upload or Take Photo")
    img_out = gr.Image(type="numpy", label="Detections")

    with gr.Accordion("⚙️ Settings", open=False):
        img_size = gr.Slider(320, 1280, value=640, step=32, label="Image size")
        conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence")
        iou = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU")

    run_btn = gr.Button("🔎 Detect")
    run_btn.click(fn=detect_image, inputs=[img_in, img_size, conf, iou], outputs=img_out)

    gr.Markdown(
        """
        **Tips**
        - บนมือถือ ปุ่ม Upload จะมีตัวเลือก **ถ่ายรูปจากกล้อง** ได้อัตโนมัติ
        - ถ้าไม่ขึ้น ให้เปิดเว็บใน Chrome (Android) หรือ Safari (iOS)
        - ต้องอนุญาตสิทธิ์การใช้กล้องใน browser
        """
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", 7860))
)

