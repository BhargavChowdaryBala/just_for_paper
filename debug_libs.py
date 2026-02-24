import os
import sys

# Windows fixes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'

print("Starting Library Check...")

try:
    import numpy as np
    print(f"Numpy version: {np.__version__}")
except Exception as e:
    print(f"Numpy import failed: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV import failed: {e}")

print("Importing Ultralytics...")
try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
except Exception as e:
    print(f"Ultralytics import failed: {e}")

print("Importing Paddle...")
try:
    import paddle
    print(f"PaddlePaddle version: {paddle.__version__}")
except Exception as e:
    print(f"PaddlePaddle import failed: {e}")

print("Importing PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully")
except Exception as e:
    print(f"PaddleOCR import failed: {e}")

print("Check Complete.")
