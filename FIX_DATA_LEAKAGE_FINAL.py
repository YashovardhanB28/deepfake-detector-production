"""
FIX DATA LEAKAGE - COMPLETE SCRIPT
Purpose: Split FaceForensics++ videos by VIDEO ID and re-extract frames cleanly
Date: December 28, 2025
Author: AI Assistant

WHAT THIS DOES:
1. Scans all videos in FaceForensics++_C23
2. Splits videos (not frames) into train/val/test by VIDEO ID
3. Re-extracts frames from each set separately
4. Creates data_clean/ folder with zero data leakage
5. Logs everything for transparency

OUTPUT:
data_clean/
├── train/real/frames/ (60K images from different videos)
├── train/fake/frames/ (60K images from different videos)
├── val/real/frames/   (20K images from different videos)
├── val/fake/frames/   (20K images from different videos)
├── test/real/frames/  (20K images from different videos)
└── test/fake/frames/  (20K images from different videos)

NO DATA LEAKAGE: Each video appears in ONLY ONE set (train/val/test)
"""

import os
import cv2
import random
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path(r"D:\deepfake_detector_production")
FACEFORENSICS_PATH = BASE_PATH / "data" / "faceforensics" / "FaceForensics++_C23"
DATA_CLEAN_PATH = BASE_PATH / "data" / "data_clean"
LOG_FILE = BASE_PATH / "FIX_DATA_LEAKAGE_LOG.txt"

# Video counts per category (adjust if different)
CONFIG = {
    "originals": {"path": "originals", "is_real": True, "count": 1000},
    "Deepfakes": {"path": "Deepfakes", "is_real": False, "count": 1000},
    "Face2Face": {"path": "Face2Face", "is_real": False, "count": 1000},
    "FaceSwap": {"path": "FaceSwap", "is_real": False, "count": 1000},
    "FaceShifter": {"path": "FaceShifter", "is_real": False, "count": 1000},
    "NeuralTextures": {"path": "NeuralTextures", "is_real": False, "count": 1000},
    "DeepFakeDetection": {"path": "DeepFakeDetection", "is_real": False, "count": 1000},
}

SPLIT_RATIO = {
    "train": 0.6,    # 60% videos to train
    "val": 0.2,      # 20% videos to val
    "test": 0.2      # 20% videos to test
}

FRAMES_PER_VIDEO = {
    "real": 150,     # Extract 150 frames per real video
    "fake": 30       # Extract 30 frames per fake video
}

# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

logger = Logger(LOG_FILE)

# ============================================================================
# STEP 1: SCAN ALL VIDEOS
# ============================================================================

def scan_videos():
    """Scan FaceForensics++_C23 and list all videos"""
    logger.log("\n" + "="*80)
    logger.log("STEP 1: SCANNING ALL VIDEOS IN FACEFORENSICS++_C23")
    logger.log("="*80)
    
    video_dict = {}  # {category: [video_names]}
    
    for category, info in CONFIG.items():
        category_path = FACEFORENSICS_PATH / info["path"]
        
        if not category_path.exists():
            logger.log(f"⚠️  Not found: {category_path}")
            continue
        
        # Find all .mp4 files
        videos = list(category_path.glob("*.mp4"))
        video_dict[category] = sorted([v.stem for v in videos])
        
        logger.log(f"✅ {category}: Found {len(videos)} videos")
    
    # Log summary
    total_videos = sum(len(v) for v in video_dict.values())
    logger.log(f"\n📊 TOTAL VIDEOS: {total_videos}")
    
    return video_dict

# ============================================================================
# STEP 2: SPLIT VIDEOS BY VIDEO ID
# ============================================================================

def split_videos_by_id(video_dict):
    """Split videos into train/val/test by VIDEO ID (no leakage)"""
    logger.log("\n" + "="*80)
    logger.log("STEP 2: SPLITTING VIDEOS BY VIDEO ID (NO LEAKAGE)")
    logger.log("="*80)
    
    random.seed(42)  # For reproducibility
    
    splits = {"train": [], "val": [], "test": []}
    
    for category, videos in video_dict.items():
        # Shuffle videos
        shuffled = videos.copy()
        random.shuffle(shuffled)
        
        # Calculate split points
        total = len(shuffled)
        train_size = int(total * SPLIT_RATIO["train"])
        val_size = int(total * SPLIT_RATIO["val"])
        
        # Assign to splits
        train_videos = shuffled[:train_size]
        val_videos = shuffled[train_size:train_size+val_size]
        test_videos = shuffled[train_size+val_size:]
        
        is_real = CONFIG[category]["is_real"]
        label = "real" if is_real else "fake"
        
        for v in train_videos:
            splits["train"].append((category, v, label))
        for v in val_videos:
            splits["val"].append((category, v, label))
        for v in test_videos:
            splits["test"].append((category, v, label))
        
        logger.log(f"\n{category} ({label}):")
        logger.log(f"  Train: {len(train_videos)} videos")
        logger.log(f"  Val:   {len(val_videos)} videos")
        logger.log(f"  Test:  {len(test_videos)} videos")
    
    # Log summary
    logger.log(f"\n📊 SPLIT SUMMARY:")
    logger.log(f"  Train: {len(splits['train'])} videos")
    logger.log(f"  Val:   {len(splits['val'])} videos")
    logger.log(f"  Test:  {len(splits['test'])} videos")
    
    return splits

# ============================================================================
# STEP 3: EXTRACT FRAMES FROM VIDEOS
# ============================================================================

def extract_frames_from_video(video_path, num_frames, output_dir, video_name, category):
    """Extract frames from a single video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return 0
        
        # Evenly sample frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        extracted = 0
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                
                # Save with pattern: category_videoname_frameindex
                output_path = output_dir / f"{category}_{video_name}_{i:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                extracted += 1
        
        cap.release()
        return extracted
    
    except Exception as e:
        logger.log(f"⚠️  Error extracting from {video_path}: {e}")
        return 0

def extract_all_frames(splits):
    """Extract frames from all videos in splits"""
    logger.log("\n" + "="*80)
    logger.log("STEP 3: EXTRACTING FRAMES FROM VIDEOS")
    logger.log("="*80)
    
    # Create output directories
    DATA_CLEAN_PATH.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (DATA_CLEAN_PATH / split_name / label / "frames").mkdir(
                parents=True, exist_ok=True
            )
    
    # Extract frames for each split
    stats = {"train": {"real": 0, "fake": 0},
             "val": {"real": 0, "fake": 0},
             "test": {"real": 0, "fake": 0}}
    
    for split_name, videos in splits.items():
        logger.log(f"\n📁 Processing {split_name} split ({len(videos)} videos)...")
        
        for category, video_name, label in tqdm(videos, desc=f"{split_name} {label}"):
            # Get video path
            video_path = FACEFORENSICS_PATH / CONFIG[category]["path"] / f"{video_name}.mp4"
            
            if not video_path.exists():
                continue
            
            # Determine number of frames
            num_frames = FRAMES_PER_VIDEO[label]
            
            # Output directory
            output_dir = DATA_CLEAN_PATH / split_name / label / "frames"
            
            # Extract frames
            extracted = extract_frames_from_video(
                video_path, num_frames, output_dir, category, video_name
            )
            
            if extracted > 0:
                stats[split_name][label] += extracted
    
    # Log final stats
    logger.log(f"\n📊 EXTRACTION SUMMARY:")
    for split_name in ["train", "val", "test"]:
        real_count = stats[split_name]["real"]
        fake_count = stats[split_name]["fake"]
        total = real_count + fake_count
        logger.log(f"  {split_name}: {real_count} real + {fake_count} fake = {total} frames")
    
    return stats

# ============================================================================
# STEP 4: VERIFICATION
# ============================================================================

def verify_clean_split():
    """Verify no data leakage occurred"""
    logger.log("\n" + "="*80)
    logger.log("STEP 4: VERIFYING NO DATA LEAKAGE")
    logger.log("="*80)
    
    # Check that no video appears in multiple splits
    train_videos = set()
    val_videos = set()
    test_videos = set()
    
    for split_path, video_set in [
        (DATA_CLEAN_PATH / "train", train_videos),
        (DATA_CLEAN_PATH / "val", val_videos),
        (DATA_CLEAN_PATH / "test", test_videos)
    ]:
        for label_dir in split_path.glob("*/frames"):
            for frame_file in label_dir.glob("*.jpg"):
                # Extract video name from filename
                parts = frame_file.stem.split("_")
                if len(parts) >= 3:
                    video_id = "_".join(parts[:-1])
                    video_set.add(video_id)
    
    # Check for overlap
    overlap_train_val = train_videos & val_videos
    overlap_train_test = train_videos & test_videos
    overlap_val_test = val_videos & test_videos
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.log("❌ DATA LEAKAGE DETECTED!")
        if overlap_train_val:
            logger.log(f"   Train-Val overlap: {overlap_train_val}")
        if overlap_train_test:
            logger.log(f"   Train-Test overlap: {overlap_train_test}")
        if overlap_val_test:
            logger.log(f"   Val-Test overlap: {overlap_val_test}")
        return False
    else:
        logger.log("✅ NO DATA LEAKAGE - All videos are in separate splits!")
        logger.log(f"   Train videos: {len(train_videos)}")
        logger.log(f"   Val videos: {len(val_videos)}")
        logger.log(f"   Test videos: {len(test_videos)}")
        return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.log("\n" + "="*80)
    logger.log("FIX DATA LEAKAGE - COMPLETE WORKFLOW")
    logger.log("="*80)
    logger.log(f"Start time: {datetime.now()}")
    
    # Step 1: Scan videos
    video_dict = scan_videos()
    
    # Step 2: Split videos by ID
    splits = split_videos_by_id(video_dict)
    
    # Step 3: Extract frames
    stats = extract_all_frames(splits)
    
    # Step 4: Verify
    is_clean = verify_clean_split()
    
    # Final summary
    logger.log("\n" + "="*80)
    logger.log("PHASE 1 COMPLETE!")
    logger.log("="*80)
    
    if is_clean:
        logger.log("✅ data_clean/ folder is ready with ZERO data leakage")
        logger.log("\nNEXT STEPS:")
        logger.log("1. Update MY_TRAINING_SCRIPT.py to use: data/data_clean/")
        logger.log("   Change: DATA_PATH = BASE_PATH / 'data' / 'data_clean'")
        logger.log("2. Run training for 2-3 epochs")
        logger.log("3. Verify accuracy is ~85% (honest, not 97%)")
        logger.log("4. Save model as 'baseline_85_percent_model.pth'")
        logger.log("5. Move to Phase 2: Collect modern deepfakes")
    else:
        logger.log("❌ Data leakage still detected - check the log")
    
    logger.log(f"\nEnd time: {datetime.now()}")
    logger.log("\n✅ Log saved to: FIX_DATA_LEAKAGE_LOG.txt")
