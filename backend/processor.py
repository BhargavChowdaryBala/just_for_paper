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

    def extract_indian_number_plate(self, text_list):
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

    def apply_gamma(self, img, gamma=1.2):
        """Brightens low-light images using non-linear gamma adjustment."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def dehaze_image(self, img):
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

    def apply_clahe(self, img, clip_limit=2.0, grid_size=(8, 8)):
        """Applies CLAHE to enhance contrast."""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return img

    def analyze_environment(self, img):
        """Analyzes image statistics to detect environmental conditions."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness
        mean_brightness = np.mean(gray)
        
        # 2. Contrast (Standard Deviation)
        contrast = np.std(gray)
        
        # 3. Blur (Laplacian Variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 4. Haze (Estimation based on Dark Channel)
        min_channel = np.min(img, axis=2)
        haze_score = np.mean(min_channel)
        
        conditions = []
        if mean_brightness < 60: conditions.append("Low Light")
        if haze_score > 100: conditions.append("Hazy/Dusty")
        if blur_score < 100: conditions.append("Motion Blur")
        if contrast < 40: conditions.append("Low Contrast")
        
        return {
            "brightness": mean_brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "haze_score": haze_score,
            "detected_conditions": conditions
        }

    def unsharp_mask(self, img, sigma=1.0, strength=1.5):
        """Applies unsharp masking to sharpen characters."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    def denoise_image(self, img):
        """Aggressive denoising for rainy/noisy conditions."""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def process_image(self, img):
        """
        Processes an image with atmospheric refinement and AI detection.
        Returns: {plate_text, confidence, plate_crop, raw_texts, refined_img}
        """
        with self.lock:
            if img is None:
                return None

            # 1. Analyze Environment
            env_stats = self.analyze_environment(img)
            print(f"[Adaptive] Detected Conditions: {env_stats['detected_conditions']}")
            
            # 2. Adaptive Preprocessing Chain
            refined_img = img.copy()
            
            # Apply Dehazing if hazy
            if "Hazy/Dusty" in env_stats['detected_conditions']:
                refined_img = self.dehaze_image(refined_img)
                
            # Apply Gamma Correction if dark
            if "Low Light" in env_stats['detected_conditions']:
                refined_img = self.apply_gamma(refined_img, gamma=1.8 if env_stats['brightness'] < 30 else 1.4)
                
            # Apply Sharpening if blurry or low contrast
            if "Motion Blur" in env_stats['detected_conditions'] or "Low Contrast" in env_stats['detected_conditions']:
                refined_img = self.unsharp_mask(refined_img)
                
            # Global Contrast Boost
            refined_img = self.apply_clahe(refined_img, clip_limit=3.0, grid_size=(8, 8))
            
            # Noise reduction
            refined_img = cv2.bilateralFilter(refined_img, 9, 75, 75)

            # 3. YOLO Detection on Refined Frame
            results = self.model(refined_img, conf=0.4)[0]
            
            final_plate_text = "Not Detected"
            final_confidence = 0
            plate_crop = None
            raw_texts = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop plate
                # Add small padding for context
                h, w = refined_img.shape[:2]
                pad = 5
                px1, py1 = max(0, x1-pad), max(0, y1-pad)
                px2, py2 = min(w, x2+pad), min(h, y2+pad)
                
                plate_crop = refined_img[py1:py2, px1:px2]
                
                # 4. Local Plate Normalization (High Intensity Polish)
                if plate_crop is not None and plate_crop.size > 0:
                    # Upscale for better OCR
                    plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    # Local sharpening
                    plate_crop = self.unsharp_mask(plate_crop, strength=2.0)
                    # Local contrast
                    plate_crop = self.apply_clahe(plate_crop, clip_limit=4.0)

                    # 5. OCR
                    ocr_results = self.ocr.ocr(plate_crop, cls=True)
                    
                    if ocr_results and ocr_results[0]:
                        for line in ocr_results[0]:
                            text = line[1][0]
                            ocr_conf = line[1][1]
                            raw_texts.append(text)
                            
                            plate_id = self.extract_indian_number_plate(text)
                            if plate_id:
                                final_plate_text = plate_id
                                final_confidence = int(ocr_conf * 100)
                                break
                
                if final_plate_text != "Not Detected":
                    break

            return {
                "plate_text": final_plate_text,
                "confidence": final_confidence,
                "plate_crop": plate_crop,
                "refined_img": refined_img,
                "raw_texts": raw_texts,
                "env_stats": env_stats
            }
