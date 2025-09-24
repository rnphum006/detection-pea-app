from huggingface_hub import HfApi
from ultralytics import YOLO

# โหลดโมเดลตรงจาก Hugging Face
model = YOLO("Phum52321/detection-app/best.pt")

# ทดสอบ predict
results = model.predict(source="test.jpg")
results[0].show()

repo_id = "Phum52321/detection-app"   # <<== ใช้ username จริง

api = HfApi()

api.upload_file(
    path_or_fileobj="best.pt",
    path_in_repo="best.pt",
    repo_id=repo_id,
    repo_type="model"
)

print(f"✅ Upload เสร็จแล้ว: https://huggingface.co/{repo_id}")
