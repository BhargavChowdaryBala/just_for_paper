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

def apply_clahe(img):
    """Applies CLAHE to enhance contrast, helpful for OCR."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
        Processes an image to detect and OCR a license plate.
        Returns: {plate_text, confidence, plate_crop_b64, raw_texts, metrics}
        """
        with self.lock:
            # YOLO Detection
            results = self.model.predict(source=img, conf=0.4, verbose=False)
            
            plate_text = "NOT_DETECTED"
            confidence = 0
            plate_crop = None
            raw_texts = []

            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the box with highest confidence
                best_box = results[0].boxes[0]
                confidence = float(best_box.conf[0].cpu().item())
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                
                # Crop and enhance
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crop = apply_clahe(plate_crop)
                    
                    # OCR
                    # Save temp for PaddleOCR path compatibility if needed, 
                    # but it also accepts numpy arrays
                    ocr_res = self.ocr.ocr(plate_crop, cls=True)
                    
                    if ocr_res and ocr_res[0]:
                        for line in ocr_res[0]:
                            raw_texts.append(line[1][0])
                        
                        plate_text = extract_indian_number_plate(raw_texts)

            return {
                "plate_text": plate_text,
                "confidence": round(confidence * 100, 1),
                "plate_crop": plate_crop,
                "raw_texts": raw_texts
            }
