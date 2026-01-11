# 🎯 Deepfake Detector - Complete Training Guide

## 📋 Quick Overview

You have **3 datasets** ready to train the ultimate deepfake detector:

1. **FaceForensics++** - Multiple fake generation methods
2. **Celeb-DF** - Celebrity deepfakes
3. **WildDeepfake** - Real-world deepfakes

## 🚀 Step-by-Step Process

### Step 1: Extract Frames (6-8 frames per video)

```bash
python 1_extract_frames_all_datasets.py
```

**What it does:**
- Scans all 3 datasets
- Extracts 6 frames per video (optimized for 18GB storage)
- Organizes frames by dataset and label
- Creates `video_registry.json` to track videos
- Prevents data leakage with video-level tracking

**Expected output:**
```
D:\deepfake_detector_production\data\all_datasets_frames\
├── faceforensics\
│   ├── real\frames\
│   └── fake\frames\
├── celeb_df\
│   ├── real\frames\
│   └── fake\frames\
├── wild_deepfake\
│   ├── real\frames\
│   └── fake\frames\
└── video_registry.json
```

**Time estimate:** 1-2 hours depending on dataset size

---

### Step 2: Train the Model

```bash
python 2_train_all_datasets.py
```

**What it does:**
- Loads ALL extracted frames
- Creates video-level splits (70% train, 15% val, 15% test)
- **NO DATA LEAKAGE** - frames from same video stay together
- Trains EfficientNet-B1 model
- Uses mixed precision (faster training)
- Saves best model based on validation accuracy

**Training features:**
- ✅ Mixup augmentation (prevents overfitting)
- ✅ Label smoothing
- ✅ Heavy data augmentation
- ✅ Early stopping (stops if not improving)
- ✅ Gradient clipping (stable training)

**Saved files:**
- `models/best_model_multi_dataset.pth` - Best model
- `logs/training_history_multi_dataset.json` - Training metrics
- `logs/test_results_multi_dataset.json` - Final test results

**Time estimate:** 4-12 hours depending on GPU

---

### Step 3: Test the Model

```bash
python 3_test_detector.py
```

**Testing options:**

#### Option 1: Test Single Video
```python
# Interactive mode
python 3_test_detector.py
# Choose option 1, enter video path
```

#### Option 2: Test Directory
```python
# Tests all videos in a folder
python 3_test_detector.py
# Choose option 2, enter directory path
```

#### Option 3: Test from Dataset
```python
# Tests videos from the test set
python 3_test_detector.py
# Choose option 3, specify number of videos
```

#### Option 4: Quick Test (Recommended First!)
```python
# Tests 10 random videos from test set
python 3_test_detector.py
# Choose option 4
```

**What you'll see:**
```
🎬 Testing: example_video.mp4
   Extracted 20 frames
   Frame  1: FAKE (confidence: 87.3%)
   Frame  2: FAKE (confidence: 92.1%)
   ...

📊 RESULTS:
   Prediction: FAKE
   Confidence: 89.45%
   Frame breakdown:
      REAL frames: 3/20 (15.0%)
      FAKE frames: 17/20 (85.0%)
```

---

## 🎓 Understanding the System

### Why Video-Level Splits?

**CRITICAL:** We split by VIDEO, not by frames!

❌ **WRONG:** Random frame split
- Train: [video1_frame1, video2_frame3, video1_frame5]
- Test: [video1_frame2, video2_frame1]
- **Problem:** Model sees same faces in train and test = cheating!

✅ **CORRECT:** Video-level split
- Train: [all frames from video1, all frames from video2]
- Test: [all frames from video3, all frames from video4]
- **Benefit:** Model must generalize to new faces!

### Why Multiple Datasets?

| Dataset | Strength | Weaknesses |
|---------|----------|------------|
| FaceForensics++ | High quality, multiple methods | Only 160 unique people |
| Celeb-DF | Celebrity faces, YouTube real | Limited scenarios |
| WildDeepfake | Real-world diversity | Variable quality |

**Combined = Best of all worlds!**

### Model Architecture

```
EfficientNet-B1 (Pretrained on ImageNet)
    ↓
Dropout (50%)
    ↓
Dense Layer (512 units)
    ↓
Dropout (30%)
    ↓
Output (2 classes: Real/Fake)
```

**Why EfficientNet-B1?**
- ✅ Good balance of speed and accuracy
- ✅ Pre-trained on ImageNet (transfer learning)
- ✅ ~7M parameters (not too big, not too small)

---

## 🔍 Expected Results

### After Training:

**Good signs:**
- ✅ Validation accuracy: 85-95%
- ✅ Test accuracy: 80-92%
- ✅ Small gap between train and val (< 10%)
- ✅ High precision AND recall (> 0.80)

**Bad signs:**
- ❌ Train accuracy >> Val accuracy (overfitting)
- ❌ Both train and val accuracy low (underfitting)
- ❌ High accuracy but low F1 (imbalanced predictions)

### During Testing:

**Real-world videos might be harder!**
- Dataset videos: 80-95% accuracy (expected)
- Real-world videos: 60-85% accuracy (normal)
- Compressed/low-quality: May perform worse

---

## 🐛 Troubleshooting

### Problem: Out of Memory
**Solution:**
- Reduce `BATCH_SIZE` in config (line 36)
- Use fewer frames per video (line 15)

### Problem: Training too slow
**Solution:**
- Reduce `NUM_WORKERS` if CPU bottleneck
- Check GPU usage with `nvidia-smi`

### Problem: Model not improving
**Solution:**
- Check if data is balanced (real vs fake)
- Ensure video registry was created correctly
- Try lower learning rate

### Problem: Test accuracy much lower than validation
**Solution:**
- This is normal! Test set is truly unseen
- Check if specific datasets perform worse
- May need more diverse training data

---

## 📊 Monitoring Training

Watch for these metrics:

1. **Loss decreasing** - Model is learning
2. **Accuracy increasing** - Model getting better
3. **Val accuracy tracking train** - No overfitting
4. **F1 score high** - Balanced predictions

**Good training curve:**
```
Epoch 1:  Train 65% | Val 62% ✓
Epoch 5:  Train 78% | Val 75% ✓
Epoch 10: Train 85% | Val 82% ✓
Epoch 15: Train 88% | Val 86% ✓ BEST
Epoch 20: Train 91% | Val 85% ⚠️ (slight overfit, still okay)
```

---

## 🎯 Advanced Tips

### For Better Results:

1. **Balance your data** - Equal real and fake videos
2. **Use all datasets** - Don't skip any!
3. **Let it train longer** - Patience pays off
4. **Test on diverse videos** - Check generalization
5. **Check failure cases** - Learn what model struggles with

### For Production Use:

1. **Ensemble models** - Train 3-5 models, average predictions
2. **Threshold tuning** - Adjust confidence threshold for your use case
3. **Frame sampling** - More frames = better accuracy (but slower)
4. **Post-processing** - Use temporal consistency (consecutive frames should agree)

---

## 📝 Files Reference

| File | Purpose | When to Run |
|------|---------|-------------|
| `1_extract_frames_all_datasets.py` | Extract frames from videos | Once, before training |
| `2_train_all_datasets.py` | Train the model | Once, or retrain with changes |
| `3_test_detector.py` | Test trained model | Anytime after training |
| `video_registry.json` | Track videos for splitting | Auto-generated by script 1 |
| `best_model_multi_dataset.pth` | Trained model weights | Auto-saved during training |

---

## ✅ Success Checklist

Before you start:
- [ ] All 3 datasets downloaded
- [ ] Paths correct in scripts
- [ ] At least 18GB free space
- [ ] GPU available (CUDA)
- [ ] Python packages installed

After extraction:
- [ ] video_registry.json created
- [ ] Frames extracted in organized folders
- [ ] Both real and fake folders have frames
- [ ] No error messages

After training:
- [ ] Model file exists in models/
- [ ] Training completed without crashes
- [ ] Validation accuracy > 75%
- [ ] History saved in logs/

Before testing:
- [ ] Model file exists
- [ ] Test set has videos
- [ ] Can successfully test one video

---

## 🎉 You're Ready!

Run the scripts in order:
1. Extract → 2. Train → 3. Test

**Good luck! You're building something awesome! 🚀**

Questions? Check the error messages - they're designed to be helpful!