import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch
import numpy as np

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

num_classes = 2  # 1 class (object) + background
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("runs/segment/maskrcnn_best.pth", map_location="cpu"))
model.eval()

def seg_process(img_rgb):
    img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = model(img_tensor)

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