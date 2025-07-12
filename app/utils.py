from ultralytics import YOLO
import cv2
import numpy as np

from models.resnet50 import ResNet50
from models.vgg16 import VGG16
from models.mask_rcnn import get_model_instance_segmentation
import torch
import torch.nn.functional as F
from torchvision import transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'🚀 Use device: {device}')
class_names = ["normal", "osteoporosis"]
resnet50_model = ResNet50(num_classes=len(class_names))
resnet50_model_path = "weights/resnet50_best.pth"
resnet50_model.load_state_dict(torch.load(resnet50_model_path, map_location=device))
resnet50_model.to(device)
resnet50_model.eval()
vgg16_model = VGG16(num_classes=len(class_names))
vgg16_model_path = "weights/vgg16_best.pth"
vgg16_model.load_state_dict(torch.load(vgg16_model_path, map_location=device))
vgg16_model.to(device)
vgg16_model.eval()

model_cls = {
    "resnet50": resnet50_model,
    "vgg16": vgg16_model
}
model_roi = YOLO("weights/best.pt")
model_seg = get_model_instance_segmentation(num_classes=2)  # 1 class (object) + background
model_seg.load_state_dict(torch.load("weights/maskrcnn_best.pth", map_location=device))
model_seg.eval()

def extract_roi(image, conf_threshold=0.3, max_rois=2):
    '''Hàm trích xuất hình ảnh ROI'''
    # Run inference
    results = model_roi(image, conf=conf_threshold)[0]
    boxes = results.boxes
    height, width = image.shape[:2]
    
    # Clone ảnh gốc để vẽ
    image_with_boxes = image.copy()
    rois = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        rois.append(cropped)

        # Vẽ bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(rois) >= max_rois:
            break
    return image_with_boxes, rois

def adjust_gamma(image, gamma=0.5):
    '''Hàm điều chỉnh độ sáng theo gamma'''
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_xray(image):
    '''Hàm tăng cường ảnh X-ray'''
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = adjust_gamma(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def predict_cls(model_name, roi_images):
    model = model_cls[model_name]
    # Same transform as val/test
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    predictions = []
    transformed_images_cv2 = []
    for idx, roi in enumerate(roi_images):
        # Convert from numpy BGR to PIL Image
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = transforms.ToPILImage()(roi_rgb)

        # Transform and add batch dim
        input_tensor = transform(roi_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_label = class_names[pred_class]
            confidence = probs[0, pred_class].item()

        # Lưu kết quả
        predictions.append((pred_label, confidence))

        # Chuyển ảnh đã transform về dạng cv2 (grayscale 224x224, uint8)
        img_tensor = input_tensor.squeeze().cpu()  # (1, 224, 224)
        img_np = img_tensor * 0.229 + 0.485         # Unnormalize
        img_np = img_np.numpy() * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # (224, 224)
        transformed_images_cv2.append(img_np)

    return predictions, transformed_images_cv2

def segment_roi(img_rgb):
    img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = model_seg(img_tensor)

    # Visualize
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']

    img_show = img_rgb.copy()
    # Tạo một mask tổng hợp
    combined_mask = np.zeros(img_show.shape[:2], dtype=bool)

    for i in range(len(masks)):
        if scores[i] > 0.5:
            mask = masks[i, 0].cpu().numpy()
            combined_mask = np.logical_or(combined_mask, mask > 0.5)

    img_masked = img_rgb.copy()
    # Đặt ngoài vùng mask về đen
    img_masked[~combined_mask] = 0
    return img_masked