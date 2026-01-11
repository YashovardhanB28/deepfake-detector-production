# 🤖 Deepfake Detection System

<div align="center">

![Status](https://img.shields.io/badge/Status-Experimental-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Deep learning system for detecting manipulated video content**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-current-results) • [Limitations](#-known-limitations) • [Contributing](#-contributing)

</div>

---

## 📌 Overview

An experimental deepfake detection system built with PyTorch and EfficientNet-B1, trained on multiple academic datasets including FaceForensics++ and Celeb-DF. This project demonstrates advanced deep learning techniques for video manipulation detection with GPU acceleration and comprehensive visualization tools.

**Current Status:** 🟡 **Experimental Research Project**
- ✅ Strong performance (83.73%) on academic test datasets
- ⚠️ Overfits to training data - needs improvement for real-world deployment
- 🔬 Actively being improved with more diverse data and techniques

---

## ✨ Features

### Core Capabilities
✅ **Multi-Dataset Training** - Trained on 25,000+ videos from FaceForensics++ and Celeb-DF  
✅ **GPU Accelerated** - CUDA-optimized inference with mixed precision training  
✅ **Smart Class Balancing** - 70/30 fake/real sampling ratio to prevent bias  
✅ **Video-Level Splits** - Prevents data leakage during training  
✅ **Grad-CAM Visualization** - Shows which facial regions influenced decisions  
✅ **Frame-by-Frame Analysis** - Detailed confidence scores and explanations  
✅ **Temporal Consistency** - Multi-frame voting and confidence smoothing  

### Technical Features
✅ **Production-Ready Code** - Clean, documented, and modular architecture  
✅ **Easy to Use** - Simple command-line interface with interactive prompts  
✅ **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
✅ **Research Backed** - Based on published deepfake detection methodologies  

---

## 📊 Current Results

### Training Performance (January 2026)

```
╔═══════════════════════════════════════╗
║   ACADEMIC DATASET PERFORMANCE        ║
╠═══════════════════════════════════════╣
║ Test Accuracy:    83.73%              ║
║ Precision:        91.24%              ║
║ Recall:           84.91%              ║
║ F1-Score:         87.96%              ║
║                                       ║
║ Confusion Matrix:                     ║
║   True Negatives:  2,755              ║
║   False Positives: 647                ║
║   False Negatives: 1,198              ║
║   True Positives:  6,740              ║
╚═══════════════════════════════════════╝
```

### Training Data
- **Total Videos:** 12,593 (after smart sampling)
- **Total Frames:** 150,288 (extracted at 6 frames/video)
- **Datasets:** FaceForensics++ (all subsets) + Celeb-DF
- **Split:** 70% train, 15% validation, 15% test (video-level split)

### Performance Metrics
- **Inference Speed:** ~230ms per frame (RTX 4060 Laptop GPU)
- **Memory Usage:** ~2-3GB VRAM per video
- **Batch Processing:** 24 frames/batch
- **Model Parameters:** 7.17M (EfficientNet-B1)
- **Real-time Capable:** ✅ Yes on modern GPUs

---

## ⚠️ Known Limitations

### Current Challenges

**🔴 Primary Issue: Overfitting to Training Data**

The model achieves 83%+ accuracy on academic test sets (FaceForensics++, Celeb-DF) but shows **poor generalization to real-world videos**. Authentic videos from the wild are often misclassified as fake.

**Root Cause:**
- Model learned dataset-specific artifacts instead of general deepfake features
- Training data has specific visual signatures (compression, lighting, generation methods)
- Limited diversity in facial features, backgrounds, and video quality

**Evidence:**
- ✅ Works well: Videos from FaceForensics++, Celeb-DF test sets
- ❌ Struggles with: Personal recordings, YouTube videos, different lighting conditions

### Other Limitations

🟡 **Dataset & Method Constraints**
- Trained primarily on older deepfake methods (2018-2020)
- May not detect newer techniques (Stable Diffusion face swaps, advanced GANs)
- Limited exposure to diverse skin tones, ages, and facial features

🟡 **Technical Constraints**
- Requires clear frontal face visibility
- Single face per frame (no multi-face support)
- Lower accuracy on heavily compressed videos
- Sensitive to extreme lighting or occlusions

### Why This Happens

Academic datasets like FaceForensics++ have consistent characteristics:
- Similar video quality and compression
- Controlled lighting environments  
- Specific deepfake generation pipelines
- Limited background variety

The model memorized these patterns rather than learning universal manipulation indicators. **This is a common challenge in deepfake detection research.**

---

## 🛠️ Tech Stack

### Core Framework

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.0+ | Deep learning framework |
| CUDA | 12.4 | GPU acceleration |
| **EfficientNet-B1** | Pretrained | CNN backbone architecture |
| OpenCV | 4.5+ | Video & image processing |
| NumPy | 1.20+ | Numerical computing |
| Matplotlib/Seaborn | Latest | Visualization & Grad-CAM |
| scikit-learn | 1.7+ | Metrics & evaluation |

### Model Architecture

**Backbone:** EfficientNet-B1 (pretrained on ImageNet)
- Input: 224×224×3 RGB image
- Parameters: 7.17M total
- Output features: 1280-dimensional vector

**Detection Head:**
- FC Layer 1: 1280 → 512 (ReLU + Dropout 0.3)
- FC Layer 2: 512 → 2 (Softmax)
- Output: [Real probability, Fake probability]

---

## 🚀 Installation

### Prerequisites
- **NVIDIA GPU** with CUDA support (highly recommended)
- **Python 3.10+**
- **~5GB disk space** for model and dependencies
- **Git**

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YashovardhanB28/deepfake-detector-production.git
cd deepfake-detector-production

# 2. Create virtual environment
python -m venv venv_gpu
# Activate: Windows
venv_gpu\Scripts\activate
# Activate: macOS/Linux
source venv_gpu/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install opencv-python pillow numpy matplotlib seaborn scikit-learn tqdm

# 5. Verify installation
python src/setup_checker.py
```

Expected output: ✅ for Python, packages, GPU, and disk space.

### Download Pre-trained Model

The trained model (`best_model.pth`, 83.73% accuracy) is included in the repository under `models/`.

**Alternative:** Train from scratch using the training scripts (see [Training](#training-from-scratch)).

---

## 📖 Usage

### Quick Test on Any Video

```bash
python src/3_test_video.py
```

When prompted, enter your video path:
```
📹 Enter video path: C:\Users\YourName\Videos\test.mp4
```

**What you'll get:**
- Frame-by-frame predictions with confidence scores
- Grad-CAM visualizations showing attention regions
- Temporal consistency analysis
- Final verdict with overall confidence
- Beautiful visualizations saved to `analysis_results/`

### Example Output

```
📊 FRAME-BY-FRAME ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 1 @ 0.00s:
   🔴 Prediction: FAKE
   Confidence: 89.45%
   Probabilities: Real=10.55%, Fake=89.45%
   • 🔴 VERY HIGH confidence - clear manipulation indicators
   • ✅ Highly consistent with other frames (85.2%)

Frame 2 @ 1.20s:
   🔴 Prediction: FAKE
   Confidence: 87.23%
   ...

🎯 TEMPORAL CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Fake predictions: 9/10 (90.0%)
   Real predictions: 1/10 (10.0%)
   Consistency score: 90.0%

⚖️ FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🔴 DEEPFAKE DETECTED
   Overall confidence: 87.34%
   
📊 Visualizations saved to:
   analysis_results/test_20260112_143022/
```

### Advanced Options

```bash
# Specify custom output directory
python src/3_test_video.py --output my_results/

# Process multiple videos in batch
python src/3_test_video.py --batch videos/*.mp4

# Use CPU instead of GPU (slower)
python src/3_test_video.py --device cpu

# Adjust frame sampling (default: 10 frames)
python src/3_test_video.py --num-frames 20
```

### Python API Usage

```python
from src.test_video import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(
    model_path='models/best_model.pth',
    device='cuda'
)

# Analyze a video
results = detector.analyze_video(
    video_path='video.mp4',
    num_frames=10,
    save_visualizations=True
)

# Access results
print(f"Verdict: {results['verdict']}")
print(f"Confidence: {results['confidence']:.2%}")
print(f"Fake frames: {results['fake_count']}/{results['total_frames']}")

# Individual frame predictions
for frame_result in results['frames']:
    print(f"Frame {frame_result['index']}: {frame_result['prediction']} ({frame_result['confidence']:.2%})")
```

---

## 🗂️ Architecture & Pipeline

### Complete System Pipeline

```
Input Video (MP4/AVI/MOV)
    ↓
┌─────────────────────────────┐
│ STAGE 1: Frame Extraction   │
│ • Extract 10 evenly spaced   │
│ • Resize to 224×224          │
│ • RGB color space            │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ STAGE 2: Preprocessing      │
│ • Normalize to ImageNet     │
│ • Convert to tensor          │
│ • Batch preparation          │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ STAGE 3: EfficientNet-B1    │
│ • Feature extraction         │
│ • 1280-dim feature vector    │
│ • Mixed precision (FP16)     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ STAGE 4: Classification     │
│ • FC layers with dropout     │
│ • Softmax activation         │
│ • Real/Fake probabilities    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ STAGE 5: Temporal Analysis  │
│ • Multi-frame voting         │
│ • Consistency checking       │
│ • Confidence aggregation     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ STAGE 6: Visualization      │
│ • Grad-CAM heatmaps          │
│ • Confidence timeline        │
│ • Statistics & metrics       │
└─────────────────────────────┘
    ↓
Final Report + Visualizations
```

### Key Techniques Implemented

**1. Smart Class Balancing**
- Original dataset: 85% fake / 15% real (severe imbalance)
- Applied sampling: 70% fake / 30% real
- Weighted loss function to handle remaining imbalance
- **Result:** Balanced precision/recall, avoiding prediction bias

**2. Grad-CAM Visualization**
- Shows which facial regions the model focuses on
- Helps understand decision-making process
- Useful for debugging and research
- Generates interpretable heatmap overlays

**3. Temporal Consistency Analysis**
- Analyzes predictions across multiple frames
- Detects inconsistent predictions (possible errors)
- Smooths confidence scores over time
- More robust than single-frame classification

**4. In-Place Operation Fix**
- Disabled `inplace=True` in all activation functions
- Prevents gradient computation errors
- Required for mixed precision training compatibility

---

## 📚 Training From Scratch

### Dataset Preparation

**Step 1: Download Datasets**

- **FaceForensics++** (C23 compression): [Download here](https://github.com/ondyari/FaceForensics)
- **Celeb-DF**: [Download here](https://github.com/yuezunli/celeb-deepfakeforensics)

Place in:
```
data/
├── faceforensics/archive/FaceForensics++_C23/
└── archive (1)/  # Celeb-DF
```

**Step 2: Extract Frames**

```bash
python src/1_extract_frames.py
```

This processes:
- FaceForensics++: original, deepfakes, face2face, faceswap, neuraltextures, DeepFakeDetection
- Celeb-DF: Celeb-real, YouTube-real, Celeb-synthesis

Output: `data/all_datasets_frames/` (150K+ frames)

**Step 3: Train Model**

```bash
python src/2_train_model.py
```

### Training Configuration

```python
# Hyperparameters
BATCH_SIZE = 24
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
IMG_SIZE = 224
DROPOUT = 0.3

# Optimizer & Scheduler
OPTIMIZER = AdamW
SCHEDULER = ReduceLROnPlateau(patience=2)

# Data Augmentation (training only)
- Random horizontal flip (50%)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation: ±20%)

# Special Techniques
MIXED_PRECISION = True  # AMP for faster training
CLASS_BALANCING = "smart_sampling"  # 70/30 ratio
SPLIT_STRATEGY = "video_level"  # Prevents data leakage
```

### Training Timeline

On RTX 4060 Laptop GPU:
- **Epoch 1-3:** Learning basic features (~2 hours/epoch)
- **Epoch 4-7:** Refining decision boundaries (~1.5 hours/epoch)
- **Epoch 8-10:** Fine-tuning and convergence (~1 hour/epoch)
- **Total:** ~15-20 hours for full training

### Expected Results

After 10 epochs:
- Train accuracy: ~95-98%
- Validation accuracy: ~85-90%
- Test accuracy: ~83-85%

**Signs of good training:**
- Validation loss follows training loss (no huge gap)
- Confusion matrix shows balanced FP/FN
- Grad-CAM focuses on facial features, not background

---

## 📁 Project Structure

```
deepfake-detector-production/
├── src/
│   ├── 1_extract_frames.py      # Frame extraction pipeline
│   ├── 2_train_model.py         # Training script with smart balancing
│   ├── 3_test_video.py          # Interactive testing with Grad-CAM
│   └── setup_checker.py         # Environment verification
├── data/
│   ├── faceforensics/           # FaceForensics++ dataset
│   ├── archive (1)/             # Celeb-DF dataset
│   └── all_datasets_frames/     # Extracted frames (train/val/test)
├── models/
│   └── best_model.pth           # Trained weights (83.73% accuracy)
├── analysis_results/            # Test outputs with visualizations
│   └── <video_name>_<timestamp>/
│       ├── frame_analysis.png   # Grad-CAM heatmaps
│       ├── confidence_timeline.png
│       └── statistics.png
├── documentation/
│   ├── bugs_fixed_doc.md        # Bug fixes and solutions
│   ├── quickstart_guide.md      # Quick setup guide
│   └── usage_guide.md           # Detailed usage instructions
├── old_versions/                # Previous experimental scripts
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
├── cleanup_repo.py              # Repository organization script
└── README.md                    # This file
```

---

## 🔮 Future Improvements

### High Priority (Addressing Overfitting)

- [ ] **Diverse Real-World Training Data**
  - Collect personal videos, YouTube clips, TikTok content
  - Include varied lighting, compression, and quality levels
  - Add diverse facial features, ages, and skin tones
  
- [ ] **Domain Adaptation Techniques**
  - Train on academic + real-world data mix
  - Use adversarial domain adaptation
  - Fine-tune on specific real-world scenarios

- [ ] **More Benchmark Datasets**
  - DFDC (Facebook Deepfake Detection Challenge)
  - WildDeepfake
  - UADFV
  - Custom collected datasets

- [ ] **Advanced Architectures**
  - Vision Transformers (ViT, Swin Transformer)
  - Ensemble of multiple models
  - Temporal models (LSTM/GRU for video sequences)

### Medium Priority (Feature Enhancements)

- [ ] Multi-face detection and tracking
- [ ] Audio-visual analysis (detect voice deepfakes)
- [ ] Real-time video stream processing
- [ ] Web UI for easy testing
- [ ] REST API for integration
- [ ] Confidence calibration
- [ ] Adversarial robustness testing

### Low Priority (Deployment & Optimization)

- [ ] Mobile deployment (ONNX/TensorFlow Lite)
- [ ] Model compression and quantization
- [ ] Edge device optimization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Docker containerization

---

## 🔬 Research & Publications

**Published Research (2024)**
- **Title:** AI/ML Techniques for Deepfake Detection
- **Status:** Peer-reviewed academic publication
- **Contribution:** Analysis of detection methodologies and performance

**Key References:**
- Li et al. (2018) - FaceForensics Dataset
- Rossler et al. (2019) - FaceForensics++ Benchmark
- Dolhansky et al. (2020) - DFDC Challenge
- He et al. (2015) - ResNet Architecture
- Tan & Le (2019) - EfficientNet Architecture

---

## 🤝 Contributing

Contributions are welcome! This is an experimental research project with significant room for improvement.

### Ways to Contribute

1. **Testing & Feedback**
   - Test on diverse videos and report results
   - Document where it works/fails
   - Suggest improvements

2. **Code Improvements**
   - Fix bugs or improve performance
   - Add new features
   - Improve documentation

3. **Research Contributions**
   - Suggest better architectures
   - Propose training techniques
   - Share relevant papers

4. **Dataset Contributions**
   - Share diverse real-world videos
   - Help collect training data
   - Suggest new benchmark datasets

### How to Contribute

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-improvement

# 3. Make your changes
# 4. Test thoroughly
python src/setup_checker.py
python src/3_test_video.py

# 5. Commit with clear message
git commit -m "feat: Add your improvement description"

# 6. Push and create pull request
git push origin feature/your-improvement
```

### Code Style Guidelines

- Follow PEP 8 (Python style guide)
- Add type hints where possible
- Include docstrings for functions
- Write tests for new features
- Update documentation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

Free for academic research, personal projects, and commercial use with attribution.

---

## 👤 About the Developer

**Yashovardhan Bangur**

🎮 **XR/Unreal Engine Developer**  
🤖 **Applied AI Researcher (ML & NLP)**  
📍 **Location:** Ahmedabad, Gujarat, India 🇮🇳  
🌎 **Currently:** New Jersey, USA

### Skills & Interests
- Deep Learning & Computer Vision
- Unreal Engine & XR Development
- Natural Language Processing
- GPU-Accelerated Computing
- Research & Academic Writing

### Connect With Me

<p align="left">
<a href="https://linkedin.com/in/yashovardhan-bangur-83677a31a">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
<a href="https://yashovardhanbangurportfolio.netlify.app/">
  <img src="https://img.shields.io/badge/Portfolio-FF6B6B?style=for-the-badge&logo=firefoxbrowser&logoColor=white" alt="Portfolio"/>
</a>
<a href="https://github.com/YashovardhanB28">
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>
</p>

---

## 🙏 Acknowledgments

- **PyTorch Team** - Exceptional deep learning framework
- **NVIDIA** - CUDA GPU acceleration support
- **FaceForensics++ Team** - High-quality dataset and benchmark
- **Celeb-DF Team** - Diverse deepfake dataset
- **Research Community** - Ongoing deepfake detection research
- **Open Source Contributors** - Libraries and tools used in this project

---

## 📞 Support & Contact

Have questions, found bugs, or want to collaborate?

- 📧 **Email:** yashovardhanbangur2801@gmail.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/YashovardhanB28/deepfake-detector-production/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/YashovardhanB28/deepfake-detector-production/discussions)

**Response Time:** Usually within 24-48 hours

---

<div align="center">

### ⭐ If this project helped you, please give it a star!

**Last Updated:** January 2026

---

*"Detecting deception through deep learning - an experimental journey toward more trustworthy media."*

**Made with ❤️ and honest research by Yashovardhan Bangur**

---

**Disclaimer:** This is an experimental research project. While it performs well on academic benchmarks, it has known limitations with real-world videos. Use responsibly and verify results with multiple sources.

</div>