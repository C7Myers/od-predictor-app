from PIL import Image
import numpy as np
import cv2

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    resized = cv2.resize(np.array(image), (100, 100))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    normalized = cv2.equalizeHist(blurred)
    return normalized.flatten() / 255.0
