import os
import time
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from processor import ANPRProcessor

# Windows fixes for PaddleOCR/OneDNN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import traceback

app = Flask(__name__)
CORS(app)

# Initialize Processor
print("Initializing AI Models...")
try:
    processor = ANPRProcessor("best.pt")
    print("Models Ready.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize models: {e}")
    traceback.print_exc()
    processor = None

@app.route('/api/analyze', methods=['POST'])
def analyze():
    print(">>> Received /api/analyze request")
    if processor is None:
        print("!!! Error: Processor is None")
        return jsonify({"error": "AI Models failed to initialize. Check server logs."}), 500
    try:
        if 'image' not in request.files:
            print("!!! Error: No image in request.files")
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        print(f">>> Processing file: {file.filename}")
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("!!! Error: Failed to decode image")
            return jsonify({"error": "Invalid image format"}), 400

        print(">>> Starting AI Pipeline...")
        # Performance timing
        start_time = time.time()
        result = processor.process_image(img)
        end_time = time.time()
        print(f">>> AI Pipeline Finished in {round((end_time - start_time) * 1000, 2)}ms")
        
        # Encode plate crop to base64
        plate_b64 = None
        if result['plate_crop'] is not None:
            _, buffer = cv2.imencode('.jpg', result['plate_crop'])
            plate_b64 = base64.b64encode(buffer).decode('utf-8')
            print(">>> Encoded plate crop to base64")

        return jsonify({
            "plate_text": result['plate_text'],
            "confidence": result['confidence'],
            "plate_image": plate_b64,
            "raw_texts": result['raw_texts'],
            "execution_time": round((end_time - start_time) * 1000, 2)
        })

    except Exception as e:
        print(f"!!! Analysis Exception: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs')
def logs():
    return jsonify([])

@app.route('/api/status')
def status():
    return jsonify({"status": "Ready"})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
