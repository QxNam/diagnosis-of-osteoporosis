import cv2
import numpy as np

def adjust_exposure(img, factor=1.4):
    '''Điều chỉnh độ phơi sáng, hỗ trợ cả gray và RGB.'''
    img = img.astype(np.float32) * factor
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def adjust_gamma(image, gamma=0.5):
    '''Điều chỉnh gamma, hỗ trợ cả gray và RGB.'''
    tmp_image = image.copy()
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    if len(tmp_image.shape) == 2:  # gray
        return cv2.LUT(tmp_image, table)
    else:  # RGB
        return cv2.merge([cv2.LUT(tmp_image[:, :, c], table) for c in range(3)])

def clahe(image):
    '''Áp dụng CLAHE, hỗ trợ cả gray và RGB.'''
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        return image
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(ch) for ch in channels]
        return cv2.merge(clahe_channels)

def sharpen(image, strength=1.0):
    '''Sharpen bằng kernel hoặc unsharp mask, hỗ trợ gray/RGB.'''
    if len(image.shape) == 2:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]]) * strength
        return cv2.filter2D(image, -1, kernel)
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]]) * strength
        channels = cv2.split(image)
        sharp_channels = [cv2.filter2D(ch, -1, kernel) for ch in channels]
        return cv2.merge(sharp_channels)

def is_blurry(image, threshold=100):
    '''Hàm kiểm tra ảnh mờ bằng Laplacian variance'''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var

def unsharp_mask(image, strength=2.5):
    '''Hàm tăng cường khử mờ bằng unsharp masking'''
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def adjust_gamma_v1(image, gamma=0.5):
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
    
    image = adjust_gamma_v1(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image