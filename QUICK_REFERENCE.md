# ⚡ Quick Reference Card - Jewelry Try-On System

## 🚀 Installation (Copy-Paste)

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open: `http://localhost:8501`

---

## 📁 File Structure
```
jewelry-try-on/
├── app.py                    ← Main application
├── requirements.txt          ← Dependencies
├── .streamlit/
│   └── config.toml          ← Settings (optional)
└── assets/                   ← Sample images (optional)
```

---

## ⚙️ Configuration Cheat Sheet

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `necklace_width_ratio` | 3.3 | 2.0-4.0 | Necklace size |
| `earring_height_ratio` | 0.75 | 0.5-1.2 | Earring size |
| `necklace_y_offset` | 2.2 | 1.0-3.5 | Position down from chin |
| `color_match_strength` | 0.25 | 0-1 | Color adaptation |
| `edge_feather_radius` | 3 | 1-10 | Edge softness |
| `shadow_intensity` | 0.15 | 0-0.5 | Shadow darkness |
| `necklace_curve_amplitude` | 0.25 | 0-1 | 3D warping |
| `use_gpu` | True | True/False | GPU acceleration |

---

## 🎨 Quick Configs

**Fast (CPU)**:
```python
config = TryOnConfig(
    color_match_strength=0.1,
    edge_feather_radius=1,
    shadow_intensity=0.05,
    necklace_curve_amplitude=0.1
)
```

**High Quality**:
```python
config = TryOnConfig(
    color_match_strength=0.35,
    edge_feather_radius=5,
    shadow_intensity=0.2,
    necklace_curve_amplitude=0.3
)
```

**Delicate Jewelry**:
```python
config = TryOnConfig(
    necklace_width_ratio=2.5,
    edge_feather_radius=5,
    shadow_intensity=0.1
)
```

**Heavy Jewelry**:
```python
config = TryOnConfig(
    necklace_width_ratio=3.8,
    edge_feather_radius=2,
    shadow_intensity=0.25
)
```

---

## 💻 API Quick Start

```python
from app import EnhancedTryOnEngine, TryOnConfig, ModelManager
import cv2
from PIL import Image
import numpy as np

# Initialize
config = TryOnConfig()
models = ModelManager.load_all_models(config)
engine = EnhancedTryOnEngine(models, config)

# Load images
model_img = cv2.imread('model.jpg')
jewelry = np.array(Image.open('necklace.png').convert('RGBA'))

# Process
result, status = engine.process(
    model_img,
    {'Necklace': jewelry, 'Earrings': None, 'Nose Pin': None}
)

# Save
cv2.imwrite('output.png', result)
```

---

## 🔧 Common Tweaks

**Necklace too small?**
```python
config.necklace_width_ratio = 3.5  # Increase
```

**Necklace too low?**
```python
config.necklace_y_offset = 1.8  # Decrease (move up)
```

**Jewelry looks harsh?**
```python
config.edge_feather_radius = 5  # Increase
config.color_match_strength = 0.3
```

**Processing too slow?**
```python
config.use_gpu = False  # Use CPU (slower)
# OR reduce input resolution
```

**Colors don't match?**
```python
config.color_match_strength = 0.4  # Increase
```

---

## 🐛 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| "No person detected" | Try different image with clear face |
| CUDA out of memory | Set `config.use_gpu = False` |
| Module not found | Run `pip install -r requirements.txt` |
| Slow on CPU | Use GPU or reduce resolution |
| Jewelry misaligned | Adjust `necklace_y_offset` |
| Harsh edges | Increase `edge_feather_radius` to 5 |
| Wrong colors | Increase `color_match_strength` to 0.3 |

---

## 📊 Performance Reference

| Hardware | Time per Image | Quality |
|----------|---|---|
| RTX 3060 | 4-5s | Excellent |
| i7-12700K CPU | 45-60s | Excellent |
| Low-end GPU | 8-12s | Good |
| High-end CPU | 25-35s | Good |

First run: Add 1-2 minutes for model loading

---

## 🎯 Component Reference

| Component | Purpose | Key Method |
|-----------|---------|---|
| `ModelManager` | Load AI models | `load_all_models()` |
| `ImagePreprocessor` | Prepare jewelry | `remove_background()` |
| `DepthGeometryEngine` | 3D warping | `cylindrical_warp()` |
| `OcclusionAnalyzer` | Hair/clothes detection | `get_segmentation_masks()` |
| `AnatomicalDetector` | Find landmarks | `detect_anatomy()` |
| `AdvancedCompositor` | Blend images | `safe_overlay()` |
| `EnhancedTryOnEngine` | Main pipeline | `process()` |

---

## 🔑 Key Landmarks

**Face (FaceMesh)**:
- 468 (left iris), 473 (right iris) → Eyes
- 152 (chin) → Necklace position
- 48 (left nostril) → Nose pin
- 177 (left ear lobe), 401 (right ear lobe) → Earrings

**Body (Pose)**:
- 2 (left eye), 5 (right eye) → Eyes
- 11 (left shoulder), 12 (right shoulder) → Necklace

---

## 📐 Sizing Reference

**Pupillary Distance (PD)**:
- Typical: 60-70mm
- In pixels: 50-150px depending on image
- Base unit for all jewelry sizing

**Necklace**:
- Width = PD × 3.3
- Height = width × (asset_height / asset_width)

**Earrings**:
- Height = PD × 0.75
- Width = height × (asset_width / asset_height)

**Nose Pin**:
- Width = PD × 0.15

---

## 🎛️ Streamlit Config Key Settings

```toml
[theme]
primaryColor = "#9370DB"

[server]
port = 8501
maxUploadSize = 500  # MB

[logger]
level = "info"  # or "debug"

[client]
toolbarMode = "minimal"
```

---

## 🚨 Error Messages & Meanings

| Message | Meaning | Fix |
|---------|---------|-----|
| "No person detected" | Face/body not found | Use clearer image |
| "CUDA out of memory" | GPU memory full | Use CPU mode |
| "Module not found" | Missing dependency | Install requirements |
| "Connection refused" | Can't download models | Check internet/HuggingFace |
| "FaceMesh failed, trying Pose" | Close-up didn't work | Normal - using fallback |

---

## 📱 For Batch Processing

```python
from pathlib import Path

for img_path in Path('images/').glob('*.jpg'):
    model_img = cv2.imread(str(img_path))
    result, status = engine.process(model_img, jewelry_dict)
    cv2.imwrite(f'output/{img_path.stem}_out.png', result)
    print(f"{img_path.name}: {status}")
```

---

## 🔍 Debugging Commands

```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Check detection
metrics = detector.detect_anatomy(img_rgb)
print(f"PD: {metrics.pd_px:.1f}px")
print(f"Method: {metrics.method}")

# Profile timing
import time
start = time.time()
result, _ = engine.process(model_img, jewelry_dict)
print(f"Time: {time.time() - start:.2f}s")

# Monitor memory
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1e9:.2f} GB")
```

---

## 🌐 Model Information

| Model | Size | Purpose |
|-------|------|---------|
| `depth-anything-small-hf` | 100MB | 3D depth |
| `segformer_b2_clothes` | 350MB | Clothing/hair |
| MediaPipe FaceMesh | 5MB | 468 face points |
| MediaPipe Pose | 5MB | 33 body points |

**Total**: ~460MB (auto-downloads first run)

---

## 🎓 Common Parameters

```python
# Conservative (accurate, slower)
config.necklace_width_ratio = 2.8
config.necklace_curve_amplitude = 0.15
config.color_match_strength = 0.2

# Aggressive (fast, artistic)
config.necklace_width_ratio = 3.8
config.necklace_curve_amplitude = 0.35
config.color_match_strength = 0.4
```

---

## 📚 Documentation Links

- **Full Setup**: SETUP_GUIDE.md
- **API Details**: API_REFERENCE.md
- **How It Works**: TECHNICAL_DOCUMENTATION.md
- **Problems**: TROUBLESHOOTING.md
- **Overview**: README.md
- **Index**: DOCUMENTATION_INDEX.md

---

## ⏱️ Timing Benchmarks

- First run: 60-120s (models load)
- GPU processing: 3-5s
- CPU processing: 45-60s
- Model download: 5-10 min
- Cached subsequent: 15-30s (GPU) or 2-3 min (CPU)

---

## 💡 Pro Tips

1. **Use GPU if available** - 10× faster
2. **Reduce image size** - Saves processing time
3. **Start with defaults** - Then tune parameters
4. **Batch process** - More efficient than single images
5. **Clear cache if issues** - `streamlit cache clear`
6. **Monitor GPU** - `nvidia-smi` in terminal
7. **Use edge_feather_radius=5** for best quality
8. **Lower `min_face_detection_confidence`** if detection fails

---

**Version**: 1.0  
**Last Updated**: November 2024  
**Print-friendly**: Yes (save as PDF)

Keep this handy! 💎
