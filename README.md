# 🤖 Deepfake Detection System

<div align="center">

![Status](https://img.shields.io/badge/Status-Experimental-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

**Deep learning system for detecting manipulated facial video content**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-current-results) • [Limitations](#-known-limitations) • [Future Work](#-future-improvements)

</div>

---

## 📌 Overview

An experimental deepfake detection system built with PyTorch and EfficientNet-B1, trained on multiple academic datasets (FaceForensics++, Celeb-DF). This project demonstrates deep learning techniques for video manipulation detection with GPU acceleration.

**Current Status:** 🟡 **Experimental** - Works well on training datasets, needs improvement for real-world generalization.

---

## ✨ Features

✅ **Multi-Dataset Training** - Trained on 25,000+ videos from FaceForensics++ and Celeb-DF  
✅ **GPU Accelerated** - CUDA-optimized inference with mixed precision training  
✅ **Smart Class Balancing** - 70/30 sampling ratio to prevent prediction bias  
✅ **Video-Level Splits** - Prevents data leakage during training  
✅ **Grad-CAM Visualization** - Shows which facial regions the model focuses on  
✅ **Frame-by-Frame Analysis** - Detailed confidence scores per frame  
✅ **Temporal Consistency Check** - Analyzes prediction stability across video  

---

## 📊 Current Results

### Training Performance

```
╔════════════════════════════════════════╗
║   TRAINING RESULTS (Jan 2026)          ║
╠════════════════════════════════════════╣
║ Test Accuracy:    83.73%               ║
║ Precision:        91.24%               ║
║ Recall:           84.91%               ║
║ F1-Score:         87.96%               ║
║                                        ║
║ Confusion Matrix:                      ║
║   True Negatives:  2,755               ║
║   False Positives: 647                 ║
║   False Negatives: 1,198               ║
║   True Positives:  6,740               ║
╚════════════════════════════════════════╝
```

**Trained On:**
- FaceForensics++ (original, deepfakes, face2face, faceswap, neuraltextures, DeepFakeDetection)
- Celeb-DF (Celeb-real, YouTube-real, Celeb-synthesis)
- Total: 12,593 videos, 150,288 frames

**Performance Metrics:**
- Inference Speed: ~230ms per frame (RTX 4060 Laptop GPU)
- Memory Usage: ~2-3GB VRAM
- Batch Size: 24 frames
- Model Size: 7.17M parameters

---

## ⚠️ Known Limitations

**Current Issues:**

🔴 **Overfitting to Training Data**
- Model performs well (83%+) on FaceForensics++/Celeb-DF test sets
- **Poor generalization to real-world videos** - tends to classify authentic videos as fake
- Likely memorized dataset-specific artifacts rather than learning general deepfake features

🟡 **Dataset Bias**
- Training data heavily skewed toward specific deepfake generation methods
- May not detect newer deepfake techniques (Stable Diffusion, advanced GANs)
- Limited diversity in facial features and lighting conditions

🟡 **Other Limitations**
- Requires clear, frontal face visibility
- Lower accuracy on compressed/low-quality videos
- No multi-face support (single face per frame)
- Sensitive to video quality and compression

**Why This Happens:**
- Academic datasets (FaceForensics++, Celeb-DF) have specific visual signatures
- Model learned these signatures instead of general manipulation features
- Need more diverse, realistic training data for real-world deployment

---

## 🛠️ Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.0+ | Deep learning framework |
| CUDA | 12.4 | GPU acceleration |
| EfficientNet-B1 | Pretrained | CNN backbone |
| OpenCV | 4.5+ | Video processing |
| NumPy | 1.20+ | Numerical computing |
| Matplotlib/Seaborn | 3.x/0.13 | Visualization |
| scikit-learn | 1.7+ | Metrics & evaluation |

---

## 🚀 Installation

### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- Python 3.10 or higher
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/YashovardhanB28/deepfake-detector-production.git
cd deepfake-detector-production
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv_gpu
venv_gpu\Scripts\activate

# macOS/Linux
python3 -m venv venv_gpu
source venv_gpu/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install opencv-python pillow numpy matplotlib seaborn scikit-learn tqdm
```

### Step 4: Verify Installation

```bash
python src/setup_checker.py
```

Should show ✅ for Python, packages, GPU, and disk space.

---

## 📖 Usage

### Quick Start

```bash
# Test a video
python src/3_test_video.py
```

When prompted, enter your video path:
```
📹 Enter video path: C:\path\to\your\video.mp4
```

### Training From Scratch

**1. Extract Frames**

```bash
python src/1_extract_frames.py
```

This processes videos from:
- `data/faceforensics/archive/FaceForensics++_C23/`
- `data/archive (1)/` (Celeb-DF)

**2. Train Model**

```bash
python src/2_train_model.py
```

Training configuration:
- 10 epochs
- Batch size: 24
- Learning rate: 1e-4
- Smart sampling (70/30 fake/real ratio)
- Weighted loss function
- Mixed precision training

**3. Test Model**

```bash
python src/3_test_video.py
```

### Output Explanation

```
📊 FRAME-BY-FRAME ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 1 @ 0.00s:
   🔴 Prediction: FAKE
   Confidence: 89.45%
   Probabilities: Real=10.6%, Fake=89.4%
   • 🔴 VERY HIGH confidence - clear fake characteristics
   • ✅ Highly consistent across frames (85.2%)

🎯 TEMPORAL CONSISTENCY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━
   Fake detections: 9 (90.0%)
   Real detections: 1 (10.0%)
   Prediction consistency: 90.0%

⚖️ FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━
   🔴 DEEPFAKE DETECTED
   Confidence: 87.34%
```

**Visualizations saved to:** `analysis_results/<video_name>/`
- `frame_analysis.png` - Grad-CAM heatmaps showing attention regions
- `confidence_timeline.png` - Confidence scores over time
- `statistics.png` - Overall statistics

---

## 🏗️ Architecture

### Model Pipeline

```
Input Video (MP4/AVI)
    ↓
┌─────────────────────────────┐
│ Frame Extraction            │
│ - Extract 10 evenly-spaced  │
│ - Resize to 224x224         │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Preprocessing               │
│ - Normalize to ImageNet     │
│ - Data augmentation (train) │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ EfficientNet-B1 Backbone    │
│ - Pretrained on ImageNet    │
│ - Feature extraction        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Classification Head         │
│ - FC: 1280 → 512 (ReLU)     │
│ - Dropout: 0.3              │
│ - FC: 512 → 2 (Softmax)     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Temporal Aggregation        │
│ - Multi-frame voting        │
│ - Confidence averaging      │
└─────────────────────────────┘
    ↓
Final Verdict: REAL / FAKE
```

### Key Techniques

**Smart Class Balancing**
- Original: 85% fake / 15% real (severe imbalance)
- Applied: 70% fake / 30% real (balanced sampling)
- Weighted loss to handle remaining imbalance

**In-Place Operation Fix**
- Disabled `inplace=True` in all ReLU/SiLU activations
- Prevents gradient computation errors with mixed precision training

**Grad-CAM Visualization**
- Highlights which facial regions influenced the decision
- Helps understand what the model "sees"

---

## 📁 Project Structure

```
deepfake-detector-production/
├── src/
│   ├── 1_extract_frames.py      # Frame extraction from datasets
│   ├── 2_train_model.py         # Training script
│   ├── 3_test_video.py          # Interactive testing
│   └── setup_checker.py         # Environment verification
├── data/
│   ├── faceforensics/           # FaceForensics++ dataset
│   ├── archive (1)/             # Celeb-DF dataset
│   └── all_datasets_frames/     # Extracted frames
├── models/
│   └── best_model.pth           # Trained weights (83.73% acc)
├── analysis_results/            # Test outputs with visualizations
├── documentation/               # Guides and documentation
├── old_versions/                # Previous experimental scripts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔬 Training Details

### Dataset Information

**FaceForensics++ (C23 compression)**
- Original: 2,000 videos (real)
- Deepfakes: 2,000 videos
- Face2Face: 2,000 videos
- FaceSwap: 2,000 videos
- NeuralTextures: 2,000 videos
- DeepFakeDetection: 2,000 videos

**Celeb-DF**
- Celeb-real: 1,180 videos
- YouTube-real: 600 videos
- Celeb-synthesis: 11,278 videos (fake)

**After Smart Sampling:**
- Real videos: 3,778 (100% kept)
- Fake videos: 8,815 (sampled from 21,270)
- Total: 12,593 videos
- Frame count: 75,558 frames (6 per video)

### Training Hyperparameters

```python
BATCH_SIZE = 24
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
IMG_SIZE = 224
DROPOUT = 0.3
OPTIMIZER = AdamW
SCHEDULER = ReduceLROnPlateau
MIXED_PRECISION = True
```

### Data Augmentation (Training Only)

- Random horizontal flip (50%)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation: ±20%)

---

## 🔮 Future Improvements

### Planned Enhancements

**High Priority:**
- [ ] **Improve Real-World Generalization**
  - Add diverse real-world videos to training
  - Implement domain adaptation techniques
  - Use data augmentation that simulates real-world conditions
  
- [ ] **Add More Datasets**
  - DFDC (Facebook Deepfake Detection Challenge)
  - WildDeepfake
  - Custom collected real-world videos
  
- [ ] **Model Architecture Improvements**
  - Try Vision Transformers (ViT)
  - Ensemble multiple models
  - Add temporal modeling (LSTM/GRU)

**Medium Priority:**
- [ ] Multi-face detection support
- [ ] Audio-visual analysis (detect audio deepfakes)
- [ ] Real-time video stream processing
- [ ] Web UI for easy testing
- [ ] REST API for integration

**Low Priority:**
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] Explainable AI features
- [ ] Adversarial robustness testing
- [ ] Model compression/quantization

### Known Issues to Fix

1. **Overfitting** - Model needs diverse training data
2. **Compression sensitivity** - Improve robustness to video compression
3. **Lighting sensitivity** - Handle varied lighting conditions better
4. **Partial face handling** - Detect when face is partially visible

---

## 🤝 Contributing

Contributions welcome! This is an experimental project with room for improvement.

**Ways to help:**
- Test on your own videos and report results
- Suggest dataset additions
- Propose architecture improvements
- Fix bugs or improve code quality
- Improve documentation

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 👤 Author

**Yashovardhan Bangur**

🎮 XR/Unreal Engine Developer  
🤖 Applied AI Researcher (ML & NLP)  
📍 Based in Ahmedabad, Gujarat 🇮🇳  
🌍 Currently in New Jersey, USA

### Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yashovardhan-bangur-83677a31a)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF6B6B?style=for-the-badge&logo=firefoxbrowser&logoColor=white)](https://yashovardhanbangurportfolio.netlify.app/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YashovardhanB28)

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **NVIDIA** - CUDA GPU acceleration
- **FaceForensics++ Team** - Dataset and research
- **Celeb-DF Team** - Dataset contribution
- **Research Community** - Ongoing deepfake research

---

## 📞 Contact

📧 Email: yashovardhanbangur2801@gmail.com  
🐛 Issues: [GitHub Issues](https://github.com/YashovardhanB28/deepfake-detector-production/issues)

---

<div align="center">

⭐ **If this project helped you, please star it!**

**Last Updated:** January 2026

*"Detecting deception through deep learning - an experimental journey."*

</div>
