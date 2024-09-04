import cv2
import numpy as np

def resize_image(image, max_width=800, max_height=600):
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    if w > max_width:
        w = max_width
        h = int(w / aspect_ratio)
    if h > max_height:
        h = max_height
        w = int(h * aspect_ratio)
    return cv2.resize(image, (w, h))