# 🚀 Quick Start Guide

## Get Your Deepfake Detector Running in 3 Commands!

---

## Prerequisites ✅

Make sure you have:
- ✅ Python 3.8+
- ✅ NVIDIA GPU with CUDA (recommended)
- ✅ All 3 datasets downloaded
- ✅ At least 18GB free space

---

## Step 0: Verify Setup (2 minutes)

```bash
python 0_check_setup.py
```

**This will check:**
- Python version
- Required packages
- GPU/CUDA availability
- Dataset locations
- Disk space

**If it fails:** Install missing packages
```bash
pip install torch torchvision opencv-python Pillow numpy scikit-learn tqdm
```

---

## Step 1: Extract Frames (1-2 hours)

```bash
python 1_extract_frames_all_datasets.py
```

**What happens:**
- Scans all 3 datasets
- Extracts 6 frames per video
- Creates organized folder structure
- Saves video registry for training

**Expected output:**
```
✅ TOTAL:
   Videos processed: 5000+
   Frames extracted: 30000+
   Estimated size: ~15 GB
```

**Coffee break time! ☕**

---

## Step 2: Train Model (4-12 hours)

```bash
python 2_train_all_datasets.py
```

**What happens:**
- Creates video-level splits (70/15/15)
- Trains EfficientNet-B1 model
- Saves best model automatically
- Stops early if not improving

**Watch for:**
```
Epoch 1: Train 65% | Val 62%
Epoch 5: Train 78% | Val 75%
Epoch 10: Train 85% | Val 82%
✅ Best model saved
```

**Good time for a meal! 🍕**

---

## Step 3: Test Model (Instant!)

```bash
python 3_test_detector.py
```

**Choose option 4 for quick test:**
```
Choose option: 4

🚀 Running quick test...
✅ video_001 | True: real | Pred: real
✅ video_002 | True: fake | Pred: fake
...
📊 ACCURACY: 85.5% (9/10)
```

**Done! 🎉**

---

## Testing Your Own Videos

### Single Video:
```bash
python 3_test_detector.py
# Choose option 1
# Enter: C:\path\to\your\video.mp4
```

### Entire Folder:
```bash
python 3_test_detector.py
# Choose option 2
# Enter: C:\path\to\folder
```

---

## Expected Results

### After Training:
- ✅ Val Accuracy: 85-95%
- ✅ Test Accuracy: 80-92%
- ✅ F1 Score: 0.80-0.93

### Testing Real Videos:
- Dataset videos: 80-95% accuracy ✨
- Real-world videos: 65-85% accuracy (normal!)
- Compressed/low-quality: May be lower

---

## Files Created

After running all steps:

```
D:\deepfake_detector_production\
├── data\
│   └── all_datasets_frames\
│       ├── faceforensics\
│       ├── celeb_df\
│       ├── wild_deepfake\
│       └── video_registry.json ← Important!
│
├── models\
│   └── best_model_multi_dataset.pth ← Your trained model
│
└── logs\
    ├── training_history.json
    └── test_results.json
```

---

## Troubleshooting

### "Model not found"
→ Run step 2 (training) first!

### "video_registry.json not found"
→ Run step 1 (extraction) first!

### "Out of memory"
→ Edit config in training script:
```python
BATCH_SIZE = 16  # Reduce from 24
```

### "Videos not found"
→ Check paths in extraction script match your folders

---

## How Long Does It Take?

| Step | Time | Can Skip? |
|------|------|-----------|
| 0. Setup Check | 2 min | No |
| 1. Extraction | 1-2 hours | No |
| 2. Training | 4-12 hours | No |
| 3. Testing | Instant | Yes (but why?) |

**Total: ~8 hours** (mostly unattended)

---

## Pro Tips

1. **Run overnight:** Training takes hours, set it up before bed
2. **Monitor GPU:** Use `nvidia-smi` to check GPU usage
3. **Save checkpoints:** Model auto-saves, don't interrupt training
4. **Test progressively:** Quick test → dataset test → own videos
5. **Check logs:** training_history.json shows learning curves

---

## Success Indicators

### ✅ Extraction Successful:
```
✅ TOTAL:
   Videos processed: 5000+
   Frames extracted: 30000+
```

### ✅ Training Going Well:
```
Epoch 5: Train 78% | Val 75%  ← Val should be close to Train
✅ Best model saved  ← Getting better!
```

### ✅ Good Test Results:
```
Test Accuracy: 87.5%  ← Above 80% is good!
Precision: 0.89
Recall: 0.86
F1: 0.87
```

---

## What Each Metric Means

- **Accuracy:** Overall correctness (aim for >80%)
- **Precision:** Of predicted fakes, how many are actually fake?
- **Recall:** Of actual fakes, how many did we catch?
- **F1 Score:** Balance of precision and recall (aim for >0.80)

---

## Next Steps

Once trained and tested:

1. **Test real-world videos** from YouTube, social media
2. **Check failure cases** - what does it struggle with?
3. **Fine-tune** if needed (lower learning rate, more epochs)
4. **Deploy** - use in your applications!

---

## Need Help?

### Check these in order:

1. **Error messages** - they're designed to be helpful!
2. **BUGS_FIXED_AND_IMPROVEMENTS.md** - known issues and fixes
3. **README_USAGE.md** - detailed guide
4. **Training logs** - in `logs/training_history.json`

---

## Final Checklist

Before you start:
- [ ] Python 3.8+ installed
- [ ] GPU with CUDA (or patient for CPU training!)
- [ ] Datasets downloaded and extracted
- [ ] 18GB+ free space
- [ ] Packages installed (`pip install ...`)

Then just run:
```bash
python 0_check_setup.py      # Verify
python 1_extract_frames_all_datasets.py  # Extract (1-2hr)
python 2_train_all_datasets.py           # Train (4-12hr)
python 3_test_detector.py                # Test (instant!)
```

**That's it! You're building a production-grade deepfake detector! 🎉**