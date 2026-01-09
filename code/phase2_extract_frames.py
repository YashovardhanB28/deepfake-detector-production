import os
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
# 1. DATA SECURITY - Add to code/phase2_extract_frames.py
import hashlib
import logging
from pathlib import Path

# Implement: Encryption + Access Control
class DataSecurityManager:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Track ALL access/changes"""
        logging.basicConfig(
            filename='logs/security_audit.log',
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
    
    def verify_frame_integrity(self, frame_path):
        """Prevent corrupted data"""
        file_hash = hashlib.sha256()
        with open(frame_path, 'rb') as f:
            file_hash.update(f.read())
        return file_hash.hexdigest()
    
    def backup_frames(self, source_dir, backup_dir):
        """Automated backup"""
        # Copy to backup location (cloud or external drive)
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        logging.info(f"✅ Backup completed: {backup_dir}")
    
    def encrypt_sensitive_data(self, file_path):
        """Encrypt video/frame storage"""
        # Use: from cryptography.fernet import Fernet
        # Encrypt dataset with key stored separately
        pass
    
    def audit_access(self, user, action, resource):
        """Log who did what"""
        logging.info(f"USER: {user} | ACTION: {action} | RESOURCE: {resource}")

# Usage in Phase 2:
security = DataSecurityManager('D:\\deepfake_detector_production')
security.setup_logging()
security.audit_access('system', 'extract_start', 'real_videos')

# After each batch of frames:
security.backup_frames('frames/real', 'D:\\backup\\frames_real')


base_dir = Path("D:/deepfake_detector_production")
data_dir = base_dir / "data"
processed_dir = data_dir / "processed"

# CORRECTED PATHS - Based on actual FaceForensics++ structure
video_sources = {
    'real': data_dir / "faceforensics" / "archive" / "FaceForensics++_C23" / "original",
    'fake': data_dir / "faceforensics" / "archive" / "FaceForensics++_C23" / "Deepfakes",  # Start with Deepfakes
}



def extract_frames(video_path, output_dir, max_frames=100):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: 
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


print("🚀 Phase 2: Frame Extraction Starting...")
print(f"Base directory: {base_dir}\n")

# ============================================================================
# PROCESS REAL VIDEOS (original/)
# ============================================================================
print("\n📹 Processing REAL videos...")
real_videos = sorted(list(video_sources['real'].glob("*.mp4")))
print(f"Found {len(real_videos)} real videos\n")

# Create directories
for split in ['train', 'val', 'test']:
    out_dir = processed_dir / split / "real" / "frames"
    os.makedirs(out_dir, exist_ok=True)

real_count = 0
real_frames_total = 0

for video in tqdm(real_videos, desc="real", unit="video"):
    try:
        # Distribute across splits: 80% train, 10% val, 10% test
        if real_count % 10 == 0:
            split = 'val'
        elif real_count % 10 == 1:
            split = 'test'
        else:
            split = 'train'
        
        out_dir = processed_dir / split / "real" / "frames"
        os.makedirs(out_dir, exist_ok=True)
        
        # Extract frames
        frames = extract_frames(video, out_dir, max_frames=100)
        
        # Save frames
        for i, frame in enumerate(frames):
            output_path = out_dir / f"{video.stem}_{i:04d}.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            real_frames_total += 1
        
        real_count += 1
    
    except Exception as e:
        print(f"\n⚠️ Error processing {video.name}: {str(e)}")
        continue

print(f"✅ Real videos processed: {real_count}")
print(f"   Frames extracted: {real_frames_total}\n")

# ============================================================================
# PROCESS FAKE VIDEOS (Deepfakes/)
# ============================================================================
print("\n📹 Processing FAKE videos...")
fake_videos = sorted(list(video_sources['fake'].glob("*.mp4")))
print(f"Found {len(fake_videos)} fake videos\n")

# Create directories
for split in ['train', 'val', 'test']:
    out_dir = processed_dir / split / "fake" / "frames"
    os.makedirs(out_dir, exist_ok=True)

fake_count = 0
fake_frames_total = 0

for video in tqdm(fake_videos, desc="fake", unit="video"):
    try:
        # Distribute across splits: 80% train, 10% val, 10% test
        if fake_count % 10 == 0:
            split = 'val'
        elif fake_count % 10 == 1:
            split = 'test'
        else:
            split = 'train'
        
        out_dir = processed_dir / split / "fake" / "frames"
        os.makedirs(out_dir, exist_ok=True)
        
        # Extract frames
        frames = extract_frames(video, out_dir, max_frames=100)
        
        # Save frames
        for i, frame in enumerate(frames):
            output_path = out_dir / f"{video.stem}_{i:04d}.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            fake_frames_total += 1
        
        fake_count += 1
    
    except Exception as e:
        print(f"\n⚠️ Error processing {video.name}: {str(e)}")
        continue

print(f"✅ Fake videos processed: {fake_count}")
print(f"   Frames extracted: {fake_frames_total}\n")

# ============================================================================
# VERIFICATION & SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ Phase 2 COMPLETE - Frame Extraction Finished!")
print("="*70)

print("\n📋 DATA SUMMARY:\n")

total_frames = 0
for split in ['train', 'val', 'test']:
    print(f"📂 {split.upper()} Set:")
    split_total = 0
    
    for label in ['real', 'fake']:
        frame_dir = processed_dir / split / label / "frames"
        if frame_dir.exists():
            count = len(list(frame_dir.glob("*.jpg")))
            print(f"   ├─ {label}: {count:,} frames")
            split_total += count
        else:
            print(f"   ├─ {label}: 0 frames")
    
    print(f"   └─ Subtotal: {split_total:,} frames\n")
    total_frames += split_total

print("="*70)
print(f"🎯 TOTAL FRAMES EXTRACTED: {total_frames:,}")
print("="*70)

print(f"\n📊 Statistics:")
print(f"   • Real videos: {real_count} → {real_frames_total:,} frames")
print(f"   • Fake videos: {fake_count} → {fake_frames_total:,} frames")
print(f"   • Total videos: {real_count + fake_count}")
print(f"   • Output location: {processed_dir}")

print(f"\n✨ Phase 2 Ready for Phase 3: Model Training! 🚀\n")
