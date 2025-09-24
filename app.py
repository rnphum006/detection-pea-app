import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from PIL import Image

# -----------------------------
# Load YOLO model
# -----------------------------
MODEL_PATH = os.getenv("YOLO_WEIGHTS", "best.pt")

class YoloSingleton:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model weights not found at {MODEL_PATH}. Place your trained file 'best.pt' here or set YOLO_WEIGHTS."
                )
            cls._model = YOLO(MODEL_PATH)
        return cls._model


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def run_inference(frame_rgb: np.ndarray, img_size=640, conf=0.25, iou=0.45):
    model = YoloSingleton.get_model()
    frame_bgr = rgb_to_bgr(frame_rgb)
    results = model.predict(
        source=frame_bgr, imgsz=img_size, conf=conf, iou=iou, verbose=False
    )
    annotated_bgr = results[0].plot()
    return bgr_to_rgb(annotated_bgr)


# -----------------------------
# Detect on uploaded image (or camera snapshot from mobile)
# -----------------------------
def detect_image(image: np.ndarray, img_size=640, conf=0.25, iou=0.45):
    if image is None:
        return None
    return run_inference(image, img_size=img_size, conf=conf, iou=iou)


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


if __name__ == "__main__":
    is_space = os.environ.get("SPACE_ID") is not None
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", 7860)), share=not is_space)
