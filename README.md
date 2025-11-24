# 💎 Enhanced Virtual Jewelry Try-On System

A production-ready Streamlit application combining photorealistic compositing, 3D depth awareness, strand-level hair matting, and advanced occlusion handling for virtual jewelry try-on experiences.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **🎯 Intelligent Jewelry Placement**: Automatic positioning using facial/body landmarks
- **🌊 3D-Aware Warping**: Necklaces follow natural neck contours using depth estimation
- **💇 Strand-Level Hair Matting**: MODNet-powered precision occlusion for earrings
- **🎭 Advanced Occlusion Handling**: Jewelry properly layers with hair and clothing
- **🎨 Photorealistic Compositing**: Color matching, soft shadows, skin pressure, and environmental reflections
- **🔄 Smart Fallback**: Handles close-ups (FaceMesh) and full-body shots (Pose estimation)
- **📊 Side-by-Side Comparison**: View original and result images simultaneously
- **⚡ GPU-Accelerated**: Optional CUDA support for 10× speedup
- **📥 Auto-Download Models**: Automatic model fetching from multiple mirrors
- **🎛️ Configurable Pipeline**: Tunable parameters for different jewelry types and scenarios

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- 8GB RAM (16GB+ recommended)
- Optional: NVIDIA GPU with CUDA 11.8+

### 2. Installation (2 minutes)

```bash
# Clone or download project
cd jewelry-try-on

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run Application (30 seconds)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. 💫

**Note**: On first run, the application will automatically download:
- AI models from Hugging Face (~450MB)
- MODNet matting model from mirror sources (~25MB)

---

## 🆕 What's New in v2.0

### Hair Matting Engine
- **Automatic MODNet download** from multiple mirror sources
- **Strand-level hair detection** for natural earring occlusion
- **Graceful fallback** to semantic segmentation if unavailable

### Enhanced Realism
- **Skin pressure simulation**: Jewelry creates subtle deformation on skin
- **Environmental reflections**: Metallic surfaces reflect surroundings
- **Advanced shadow casting**: Soft, depth-aware shadows

### Improved UI/UX
- **Side-by-side comparison**: Original vs. Result view
- **Better visual feedback**: Clear status messages
- **Download button**: Save results directly from the interface

---

## 📖 Documentation

### For Getting Started
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation, configuration, troubleshooting
  - System requirements
  - Step-by-step installation
  - Model download handling
  - Deployment options

### For Understanding the System
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Architecture and design
  - Component overview
  - Matting engine architecture
  - Detailed algorithm explanations
  - Performance characteristics

### For Using the API
- **[API_REFERENCE.md](API_REFERENCE.md)** - Class and method documentation
  - Complete API reference
  - MattingEngine usage
  - Usage examples
  - Common patterns

### For Solving Problems
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Issues and optimization
  - Model download failures
  - Installation problems
  - Runtime issues
  - Performance tuning

---

## 🎯 Use Cases

### Virtual Try-On Platforms
```python
result, status = engine.process(
    model_img,
    {'Necklace': jewelry_rgba, 'Earrings': earring_rgba}
)
```

### Jewelry Design Preview
```python
# Process same model with different designs
results = []
for jewelry_design in design_collection:
    result, _ = engine.process(model_img, {'Necklace': jewelry_design})
    results.append(result)
```

### Batch Processing
```python
for model_path in image_paths:
    model_img = cv2.imread(model_path)
    result, status = engine.process(model_img, jewelry_dict)
    cv2.imwrite(f'output/{Path(model_path).stem}.png', result)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     Streamlit UI Layer              │
│  Upload • Configure • Compare       │
│  • Side-by-side View                │
│  • Download Results                 │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  EnhancedTryOnEngine (Pipeline)     │
│  • Anatomy detection                │
│  • Depth mapping                    │
│  • Hair matting (MODNet)            │
│  • Jewelry processing               │
│  • Occlusion handling               │
└────────────┬────────────────────────┘
             │
     ┌───────┼───────┬────────┐
     │       │       │        │
┌────▼──┐ ┌─▼──┐ ┌──▼────┐ ┌─▼─────┐
│Depth  │ │Seg │ │Anatomy│ │Matting│
│Engine │ │ m  │ │Detector│ │Engine │
│       │ │    │ │        │ │(MODNet)│
└───────┘ └────┘ └────────┘ └───────┘
     │       │       │         │
     └───────┼───────┴─────────┘
             │
        ┌────▼──────┐
        │Compositor │
        │ • Blend   │
        │ • Color   │
        │ • Shadow  │
        │ • Reflect │
        │ • Pressure│
        └───────────┘
             │
        ┌────▼──────┐
        │Output Img │
        └───────────┘
```

### Key Components

1. **ModelManager** - Centralized model loading with caching
2. **MattingEngine** - MODNet-based hair matting with auto-download
3. **ImagePreprocessor** - Background removal, RGBA handling
4. **DepthGeometryEngine** - 3D depth mapping and cylindrical warping
5. **OcclusionAnalyzer** - Hair/clothing detection via semantic segmentation
6. **AnatomicalDetector** - Facial/body landmark detection with fallback
7. **AdvancedCompositor** - Photorealistic blending with advanced effects
8. **EnhancedTryOnEngine** - Main orchestration pipeline

---

## 🔧 Configuration

### Via UI Sidebar

```
Configuration
├── Sizing
│   ├── Necklace Width: 2.0-4.0
│   └── Earring Size: 0.5-1.2
├── Blending
│   ├── Color Matching: 0.0-1.0
│   ├── Reflection: 0.0-1.0
│   └── Shadows: 0.0-0.5
└── Detection
    └── Confidence: 0.3-1.0
```

### Programmatically

```python
config = TryOnConfig(
    necklace_width_ratio=3.5,
    earring_height_ratio=0.75,
    color_match_strength=0.35,
    reflection_intensity=0.3,
    shadow_intensity=0.2,
    edge_feather_radius=3,
    use_gpu=True
)
```

---

## 📊 Performance

| Scenario | GPU (RTX 3060) | CPU (i7-12700K) |
|----------|---|---|
| 1080×1080, Necklace | 4-5s | 45-60s |
| 1080×1080, Multiple | 7-9s | 65-80s |
| 1080×1080, w/ Matting | 6-8s | 55-70s |
| 640×640, Single | 2-3s | 20-25s |
| Model Load (First Run) | 5-8s | 20-30s |

**GPU significantly recommended for interactive use.**

---

## 🎨 Model Assets

### Models Used

| Component | Model | Size | Source | Auto-Download |
|-----------|-------|------|--------|---------------|
| Depth | depth-anything-small | 100MB | Hugging Face | ✅ |
| Segmentation | segformer_b2_clothes | 350MB | Hugging Face | ✅ |
| Hair Matting | MODNet ONNX | 25MB | Multiple Mirrors | ✅ |
| Face | MediaPipe FaceMesh | 5MB | Google | ✅ |
| Pose | MediaPipe Pose | 5MB | Google | ✅ |

### Auto-Download Behavior

**First Run**: All models download automatically (~485MB total)

**MODNet Matting Model**:
- Tries multiple public mirrors automatically
- Falls back to semantic segmentation if download fails
- Stores at: `modnet.onnx` (or custom path)

```bash
# Manual download if needed:
huggingface-cli download LiheYoung/depth-anything-small-hf
huggingface-cli download mattmdjaga/segformer_b2_clothes
```

---

## 📋 Requirements

```
Core:
- streamlit>=1.39.0
- opencv-python>=4.8.1
- numpy>=1.24.3
- pillow>=10.1.0

Deep Learning:
- torch>=2.0.1
- transformers>=4.35.2
- mediapipe>=0.10.9
- onnxruntime>=1.16.0
- onnxruntime-gpu (optional)

Processing:
- rembg>=0.0.55
- requests>=2.31.0
```

See `requirements.txt` for full list with pinned versions.

---

## 🛠️ Troubleshooting

### Model Download Issues

**"All download mirrors failed"**
```bash
# Check internet connection
# Try manual download from:
# https://huggingface.co/kirp/modnet/resolve/main/modnet_photographic_portrait_matting.onnx
# Save as: modnet.onnx in project directory
```

**"Could not find model at: ..."**
```python
# Update model path in app.py:
local_model_path = "path/to/your/modnet.onnx"
```

### Hair Matting Disabled

**"⚠️ Hair matting will be DISABLED"**
- System will use semantic segmentation instead
- Slightly less accurate for complex hairstyles
- No impact on functionality

### CUDA out of memory
```python
config.use_gpu = False  # Fall back to CPU
```

See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for comprehensive solutions.

---

## 📚 API Usage Examples

### Basic Usage with Matting
```python
from app import EnhancedTryOnEngine, TryOnConfig, ModelManager
import cv2
from PIL import Image
import numpy as np

# Initialize
config = TryOnConfig()
models = ModelManager.load_all_models(config)
engine = EnhancedTryOnEngine(models, config)

# Check matting status
if engine.matting.enabled:
    print("✅ Hair matting enabled")
else:
    print("⚠️ Using fallback segmentation")

# Load images
model_img = cv2.imread('model.jpg')
earrings = np.array(Image.open('earrings.png').convert('RGBA'))

# Process
result, status = engine.process(
    model_img,
    {'Earrings': earrings}
)

# Save
cv2.imwrite('output.png', result)
print(status)  # ✅ Success: Earrings
```

### Custom Matting Model Path
```python
# In ModelManager.load_all_models():
models['matting_engine'] = MattingEngine(
    model_path="custom/path/modnet.onnx"
)
```

### Disable Auto-Download (Use Local Only)
```python
class MattingEngine:
    def __init__(self, model_path="modnet.onnx", auto_download=False):
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            if auto_download:
                self._download_model_robust()
            else:
                logging.warning("Model not found. Matting disabled.")
                return
```

---

## 🌐 Deployment

### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t jewelry-try-on .
docker run -p 8501:8501 jewelry-try-on
```

### Streamlit Cloud
1. Push to GitHub
2. Connect at https://streamlit.io/cloud
3. Select repository and deploy
4. Models download automatically on first launch

### AWS/GCP/Azure
Deploy Docker image on compute instances with GPU support for best performance.

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional jewelry types (rings, bracelets)
- Real-time webcam processing
- Multi-person support
- Advanced texture mapping
- Performance optimizations
- More matting model options

---

## 📄 License

MIT License - see LICENSE file

---

## 🔗 Quick Links

- **Setup**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Architecture**: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- **API**: [API_REFERENCE.md](API_REFERENCE.md)
- **Help**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📞 Support

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
- Review [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for design details
- See [API_REFERENCE.md](API_REFERENCE.md) for usage examples

---

## 🎓 References

### Depth Estimation
- "Depth Anything" (2024) - [Paper](https://arxiv.org/abs/2401.10891)
- Model: `LiheYoung/depth-anything-small-hf`

### Semantic Segmentation
- SegFormer B2 - [Paper](https://arxiv.org/abs/2105.15203)
- Model: `mattmdjaga/segformer_b2_clothes`

### Hair Matting
- MODNet - [Paper](https://arxiv.org/abs/2011.11961)
- "Is a Green Screen Really Necessary for Real-Time Portrait Matting?"

### Face Landmarks
- MediaPipe FaceMesh - [Documentation](https://mediapipe.dev)

### Pose Estimation
- MediaPipe Pose - [Documentation](https://mediapipe.dev)

---

## 📈 Roadmap

- [x] Strand-level hair matting
- [x] Side-by-side comparison UI
- [x] Auto-download models
- [ ] Real-time camera mode
- [ ] Multi-person support
- [ ] Ring virtual try-on
- [ ] Bracelet support
- [ ] Advanced lighting estimation
- [ ] Mobile app
- [ ] REST API endpoint

---

**Version**: 2.0  
**Status**: ✅ Production Ready  
**Last Updated**: November 2024

---

## Quick Decision Tree

**Just want to use it?**
→ Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)

**Want to understand how it works?**
→ Read [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

**Want to use the API?**
→ Check [API_REFERENCE.md](API_REFERENCE.md)

**Something broken?**
→ See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Models not downloading?**
→ Check internet connection and firewall settings

---

Happy try-on! 💎✨
