import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Image as ImageIcon, Cpu, Activity, Zap, CheckCircle, AlertCircle, RefreshCcw } from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:5000';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const res = await axios.post(`${API_BASE}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (err) {
      console.error("Analysis failed:", err);
      alert("Failed to analyze image. Ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1>Intelligence ANPR</h1>
          <p>Advanced Plate Recognition System for Research Exploration</p>
        </motion.div>
      </header>

      <main className="checker-grid">
        <motion.section
          className="glass-card upload-section"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="card-label">Capture / Upload</div>

          <div className="dropzone" onClick={() => fileInputRef.current.click()}>
            {preview ? (
              <img src={preview} alt="Upload Preview" className="preview-img" />
            ) : (
              <div className="dropzone-hint">
                <ImageIcon size={48} opacity={0.3} style={{ marginBottom: '1rem' }} />
                <p>Click to browse or drop an image</p>
                <p style={{ fontSize: '0.7rem', marginTop: '0.5rem' }}>Supporting JPG, PNG</p>
              </div>
            )}
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: 'none' }}
            accept="image/*"
          />

          <button
            className={`neon-btn analyze-btn ${loading ? 'pulse' : ''}`}
            onClick={handleAnalyze}
            disabled={!image || loading}
          >
            {loading ? (
              <>
                <RefreshCcw className="pulse" size={20} />
                Analyzing...
              </>
            ) : (
              <>
                <Zap size={20} />
                Run Intelligence Analysis
              </>
            )}
          </button>
        </motion.section>

        <section className="glass-card results-section">
          <AnimatePresence mode="wait">
            {!result && !loading ? (
              <motion.div
                key="placeholder"
                className="result-placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Activity size={80} />
                <p style={{ marginTop: '1rem' }}>Upload an image to start AI detection</p>
              </motion.div>
            ) : loading ? (
              <motion.div
                key="loading"
                className="result-placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Cpu size={80} className="pulse" color="var(--neon-blue)" />
                <p style={{ marginTop: '1rem', color: 'var(--neon-blue)' }}>Processing Neural Networks...</p>
              </motion.div>
            ) : (
              <motion.div
                key="result"
                className="result-content"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: 'spring', damping: 20 }}
              >
                <div className="result-header">
                  <h2>Analysis Findings</h2>
                  {result.confidence > 70 && (
                    <div className="conf-badge">
                      <CheckCircle size={14} /> High Confidence
                    </div>
                  )}
                </div>

                <div className="display-grid">
                  <div className="display-card">
                    <div className="card-label">Detection Crop</div>
                    <div className="img-container">
                      {result.plate_image ? (
                        <img src={`data:image/jpeg;base64,${result.plate_image}`} alt="Plate Detection" />
                      ) : (
                        <div style={{ color: '#ff4f4f' }}>No Plate Found</div>
                      )}
                    </div>
                  </div>

                  <div className="display-card">
                    <div className="card-label">OCR Interpretation</div>
                    <div className="plate-id-hero">
                      <span className="card-label" style={{ fontSize: '0.6rem' }}>Extracted Plate</span>
                      <h3>{result.plate_text}</h3>
                    </div>

                    <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                      <div className="metric-item">
                        <div className="card-label">Confidence</div>
                        <div className="metric-val" style={{ color: result.confidence > 80 ? 'var(--neon-green)' : 'var(--neon-blue)' }}>
                          {result.confidence}%
                        </div>
                      </div>
                      <div className="metric-item">
                        <div className="card-label">Latency</div>
                        <div className="metric-val">{result.execution_time}ms</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="display-card">
                  <div className="card-label">Raw Neural Output</div>
                  <div className="raw-tags">
                    {result.raw_texts.length > 0 ? (
                      result.raw_texts.map((txt, i) => (
                        <span key={txt + i} className="tag">{txt}</span>
                      ))
                    ) : (
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No text fragments detected</span>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        © 2026 AI Transportation Lab - IEEE Research Documentation Support
      </footer>
    </div>
  );
}

export default App;
