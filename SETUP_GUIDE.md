# 🚀 Enhanced Virtual Jewelry Try-On - Setup Guide

## System Requirements

### Minimum Specifications
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: 8GB (16GB+ recommended for GPU)
- **Storage**: 5GB free space (for models)
- **Python**: 3.10 or 3.11

### Recommended for Best Performance
- **GPU**: NVIDIA CUDA 11.8+ (4GB VRAM minimum)
- **RAM**: 16GB+
- **SSD**: For faster model loading

### GPU Setup (Optional but Recommended)
Check CUDA availability:
```bash
nvidia-smi
```

If no GPU available, the system automatically falls back to CPU mode.

---

## Installation Steps

### 1. Clone/Setup Project
```bash
# Create project directory
mkdir jewelry-try-on
cd jewelry-try-on

# Initialize git (optional)
git init
```

### 2. Create Python Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt
```

**For GPU Support (NVIDIA CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU Support (NVIDIA CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-Only (default torch installation):**
```bash
pip install -r requirements.txt
# torch already includes CPU support
```

### 4. Download Model Files (First Run)

Models auto-download on first execution:
- `LiheYoung/depth-anything-small-hf` (~100MB)
- `mattmdjaga/segformer_b2_clothes` (~350MB)
- MediaPipe face/pose models (~10MB)

Set up model cache directory:
```bash
# Linux/macOS
export TRANSFORMERS_CACHE=~/.cache/huggingface/hub
export HF_HOME=~/.cache/huggingface

# Windows (PowerShell)
$env:TRANSFORMERS_CACHE="$env:USERPROFILE\.cache\huggingface\hub"
$env:HF_HOME="$env:USERPROFILE\.cache\huggingface"
```

### 5. Place Application Files

Copy these files to your project directory:
```
jewelry-try-on/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml          # Streamlit configuration (optional)
└── assets/                   # Sample images (optional)
    ├── sample_necklace.png
    ├── sample_earrings.png
    └── sample_model.jpg
```

---

## Running the Application

### Start Streamlit Server

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

The application will:
1. Launch at `http://localhost:8501`
2. Load AI models (first run takes 2-3 minutes)
3. Display UI in your browser

### Initial Load Timing

**First run (with GPU):** 60-120 seconds
- Models downloaded and cached
- CUDA initialization

**Subsequent runs:** 15-30 seconds
- Models loaded from cache

**CPU-only:** 2-3 minutes each run

---

## Configuration & Optimization

### Adjust Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
runOnSave = true
port = 8501

[logger]
level = "info"

[client]
toolbarMode = "minimal"

[theme]
primaryColor = "#9370DB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Performance Tuning

**For slower systems (CPU):**
```python
# In TryOnConfig in app.py:
use_gpu = False
edge_feather_radius = 2  # Reduce from 3
```

**For fast processing (GPU):**
```python
edge_feather_radius = 5  # Increase quality
color_match_strength = 0.3
shadow_intensity = 0.2
```

### Memory Management

**If running out of memory:**
1. Reduce input image size (max 1080p)
2. Process one jewelry item at a time
3. Clear Streamlit cache: `streamlit cache clear`

---

## Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit==1.39.0
```

### Issue: "CUDA out of memory"
**Solution:**
```python
# Edit config in app.py:
config.use_gpu = False  # Fall back to CPU
```

### Issue: Models not downloading
**Solution:**
```bash
# Manually download models:
pip install huggingface-hub
huggingface-cli download LiheYoung/depth-anything-small-hf
huggingface-cli download mattmdjaga/segformer_b2_clothes
```

### Issue: Slow on first run
**Expected behavior** - Models are downloading. Subsequent runs will be faster.

### Issue: "FaceMesh detection failed, trying Pose"
**This is normal** - Occurs with full-body photos. The system automatically falls back to Pose estimation.

### Issue: Black/corrupted output image
**Solution:**
1. Check input image format (JPG/PNG)
2. Ensure jewelry image has transparent background
3. Try reducing `color_match_strength` to 0.15

---

## Usage Guide

### 1. Prepare Input Assets

**Model Photo:**
- Format: JPG, PNG
- Resolution: 480-1080px width
- Content: Person wearing regular clothes (for necklaces)
- Quality: Good lighting, clear face

**Jewelry Images:**
- Format: PNG (with transparent background)
- Resolution: 200-600px
- Content: Isolated jewelry on transparent background
- Transparency: Clean alpha channel

### 2. Upload and Configure

1. Click "Model Photo" and upload person image
2. Select jewelry type (or "Multiple Items")
3. Upload jewelry asset(s)
4. Adjust configuration sliders (optional):
   - **Necklace Width**: 2.0-4.0 (relative to eye distance)
   - **Earring Size**: 0.5-1.2x
   - **Color Matching**: 0-1.0 strength
   - **Edge Softness**: 1-10 pixels
   - **Shadow Strength**: 0-0.5

### 3. Process

Click or drag to upload files. Processing begins automatically.
Status message shows:
- ✅ Detection method (FaceMesh/Pose)
- ✅ Processed items
- ❌ Errors (if any)

### 4. Download Result

Click "Download Result" to save PNG file.

---

## File Structure Reference

```
app.py - Main application containing:
├── TryOnConfig           - Configuration dataclass
├── ModelManager          - Model loading & caching
├── ImagePreprocessor     - Background removal, RGBA handling
├── DepthGeometryEngine   - 3D warping calculations
├── OcclusionAnalyzer     - Clothing/hair detection
├── AnatomicalDetector    - Face/body landmark detection
├── AdvancedCompositor    - Blending & compositing
├── EnhancedTryOnEngine   - Main pipeline
└── main()                - Streamlit UI
```

---

## Advanced: Running Headless

For batch processing without UI:

```bash
# Create batch_process.py
from app import EnhancedTryOnEngine, TryOnConfig, ModelManager
import cv2

config = TryOnConfig()
models = ModelManager.load_all_models(config)
engine = EnhancedTryOnEngine(models, config)

model_img = cv2.imread('model.jpg')
jewelry = cv2.imread('necklace.png', cv2.IMREAD_UNCHANGED)

result, status = engine.process(
    model_img,
    {'Necklace': jewelry}
)

cv2.imwrite('output.png', result)
print(status)
```

Run with:
```bash
python batch_process.py
```

---

## Performance Benchmarks

| Config | Device | Time | Quality |
|--------|--------|------|---------|
| Full quality | RTX 3090 | 3-5s | Excellent |
| Full quality | RTX 3060 | 8-12s | Excellent |
| Full quality | i7 CPU | 45-60s | Excellent |
| Low quality | CPU | 25-35s | Good |

---

## Deployment

### Docker Container

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t jewelry-tryon .
docker run -p 8501:8501 jewelry-tryon
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets.toml for API keys (if needed)
4. Deploy

---

## Support Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **MediaPipe**: https://mediapipe.dev
- **Transformers**: https://huggingface.co/docs/transformers
- **OpenCV**: https://docs.opencv.org

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run `streamlit run app.py`
3. ✅ Upload test images
4. ✅ Adjust configuration for your use case
5. ✅ Optimize for your hardware
6. ✅ Deploy or batch process

Good luck! 💎
