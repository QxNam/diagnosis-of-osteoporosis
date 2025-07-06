import cv2
import numpy as np

def adjust_gamma(image, gamma=0.5):
    '''Hàm điều chỉnh độ sáng theo gamma'''
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_xray(image):
    '''Hàm xử lý chính'''
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Sharpening
    # kernel_sharpen = np.array([[0, -1, 0],
    #                         [-1, 5, -1],
    #                         [0, -1, 0]])
    # image = cv2.filter2D(image, -1, kernel_sharpen)
    
    image = adjust_gamma(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def is_blurry(image, threshold=100):
    '''Hàm kiểm tra ảnh mờ bằng Laplacian variance'''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var

def unsharp_mask(image, strength=2.5):
    '''Hàm tăng cường khử mờ bằng unsharp masking'''
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)