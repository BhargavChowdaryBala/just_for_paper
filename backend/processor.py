import cv2
import os
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import threading

def extract_indian_number_plate(text_list):
    # Join OCR outputs
    text = " ".join(text_list).upper()

    # Remove IND / ND noise
    text = re.sub(r'\bIND\b|\bND\b', '', text)

    # Remove all non-alphanumeric chars
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Safe OCR corrections
    text = text.replace('O', '0').replace('I', '1')

    # ✅ Bharat Series: YYBHXXXX + optional letters
    match = re.search(r'\d{2}BH\d{4}[A-Z]{0,2}', text)
    if match:
        return match.group()

    # ✅ Normal Indian plates fallback
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{0,2}\d{1,4}', text)
    if match:
        return match.group()

    return "PLATE_NOT_DETECTED"

def apply_gamma_correction(img, gamma=1.2):
    """Brightens low-light images using non-linear gamma adjustment."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_dehaze(img):
    """Simplified dehazing using dark channel prior logic and adaptive contrast."""
    # Estimate dark channel
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    
    # Estimate atmospheric light (0.1% brightest pixels in dark channel)
    num_pixels = dark_channel.size
    num_brightest = max(1, int(num_pixels * 0.001))
    brightest_coords = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    atmospheric_light = np.max(img.reshape(-1, 3)[brightest_coords], axis=0)
    
    # Transmission map estimation
    normalized_img = img / atmospheric_light.astype(np.float32)
    min_normalized = np.min(normalized_img, axis=2)
    transmission = 1 - 0.95 * cv2.erode(min_normalized, kernel)
    
    # Guided filter approximation for smoothing transmission
    transmission = cv2.GaussianBlur(transmission, (41, 41), 0)
    
    # Recover scene radiance
    transmission = np.maximum(transmission, 0.1)
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = (img[:, :, i].astype(np.float32) - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """Applies CLAHE to enhance contrast."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

class ANPRProcessor:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.lock = threading.Lock()

    def process_image(self, img):
        """
        Processes an image with atmospheric refinement and AI detection.
        Returns: {plate_text, confidence, plate_crop, raw_texts, refined_img}
        """
        with self.lock:
            # 🌫️ Step 1: Environmental Refinement
            # De-noise 
            refined = cv2.bilateralFilter(img, 9, 75, 75)
            
            # De-haze (Fog reduction)
            refined = apply_dehaze(refined)
            
            # Low light boost
            gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 100:
                refined = apply_gamma_correction(refined, gamma=1.5)
            
            # Contrast stretch
            refined = apply_clahe(refined, clip_limit=1.5)

            # 🎯 Step 2: YOLO Detection on Refined Frame
            results = self.model.predict(source=refined, conf=0.4, verbose=False)
            
            plate_text = "NOT_DETECTED"
            confidence = 0
            plate_crop = None
            raw_texts = []

            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the box with highest confidence
                best_box = results[0].boxes[0]
                confidence = float(best_box.conf[0].cpu().item())
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                
                # Crop and enhance locally for OCR
                plate_crop = refined[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crop = apply_clahe(plate_crop, clip_limit=3.0)
                    
                    # 📝 Step 3: OCR
                    ocr_res = self.ocr.ocr(plate_crop, cls=True)
                    
                    if ocr_res and ocr_res[0]:
                        for line in ocr_res[0]:
                            raw_texts.append(line[1][0])
                        
                        plate_text = extract_indian_number_plate(raw_texts)

            return {
                "plate_text": plate_text,
                "confidence": round(confidence * 100, 1),
                "plate_crop": plate_crop,
                "raw_texts": raw_texts,
                "refined_img": refined
            }
