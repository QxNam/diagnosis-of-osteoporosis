from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/roi_yolov8n/weights/best.pt")

def detect_and_crop_roi(image_path, output_dir="crop", model=None, conf_threshold=0.3):
    # Load ảnh gốc
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)

    # Run inference
    results = model(image_path, conf=conf_threshold)[0]

    # Duyệt từng bounding box
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (224, 224))

        # Lưu ảnh crop
        crop_name = f"{os.path.splitext(image_name)[0]}_roi{i+1}.png"
        crop_path = os.path.join(output_dir, crop_name)
        cv2.imwrite(crop_path, cropped)
        print(f"Cropped ROI saved to: {crop_path}")

def apply_roi(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/normal", exist_ok=True)
    os.makedirs(f"{output_dir}/osteoporosis", exist_ok=True)
    out_map = {
        "normal": "normal",
        "osteoporosis": "osteoporosis",
        # "osteopenia": "osteoporosis"
    }
    cnt = 0
    for label in os.listdir(input_dir):
        if label in out_map.keys():
            files = os.listdir(f"{input_dir}/{label}")
            for file in files:
                cnt += 1
                detect_and_crop_roi(image_path=f"{input_dir}/{label}/{file}", output_dir=f"{output_dir}/{out_map[label]}", model=model)
