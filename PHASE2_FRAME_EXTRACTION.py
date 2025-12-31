"""
PHASE 2: FRAME EXTRACTION FROM YOUTUBE VIDEOS
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
REAL_VIDEO_FOLDER = "youtube_videos/real/"
FAKE_VIDEO_FOLDER = "youtube_videos/fake/"
OUTPUT_BASE = "datasets/youtube_realworld/frames/"

FRAMES_PER_REAL_VIDEO = 30
FRAMES_PER_FAKE_VIDEO = 10
FRAME_SIZE = (224, 224)

def create_output_folders():
    os.makedirs(f"{OUTPUT_BASE}real", exist_ok=True)
    os.makedirs(f"{OUTPUT_BASE}fake", exist_ok=True)
    print(f"✓ Output folders created at {OUTPUT_BASE}")

def extract_frames_from_video(video_path, output_folder, num_frames, label_name):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ Could not open {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"  ✗ No frames in {video_path}")
            cap.release()
            return 0
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        extracted_count = 0
        
        for frame_idx, target_frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_resized = cv2.resize(frame, FRAME_SIZE)
            video_name = Path(video_path).stem
            output_path = f"{output_folder}/{label_name}_{frame_idx:04d}.jpg"
            cv2.imwrite(output_path, frame_resized)
            extracted_count += 1
        
        cap.release()
        return extracted_count
        
    except Exception as e:
        print(f"  ✗ Error processing {video_path}: {e}")
        return 0

def extract_real_videos():
    print("\n" + "="*60)
    print("EXTRACTING FRAMES FROM REAL VIDEOS")
    print("="*60)
    
    if not os.path.exists(REAL_VIDEO_FOLDER):
        print(f"✗ Real videos folder not found: {REAL_VIDEO_FOLDER}")
        return 0
    
    real_videos = sorted([f for f in os.listdir(REAL_VIDEO_FOLDER) 
                         if f.lower().endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov'))])
    
    if not real_videos:
        print(f"✗ No videos found in {REAL_VIDEO_FOLDER}")
        return 0
    
    print(f"Found {len(real_videos)} real videos")
    print(f"Extracting {FRAMES_PER_REAL_VIDEO} frames per video...")
    print()
    
    total_frames = 0
    
    for video_file in tqdm(real_videos, desc="Real videos", unit="video"):
        video_path = os.path.join(REAL_VIDEO_FOLDER, video_file)
        video_name = Path(video_file).stem
        
        frames_extracted = extract_frames_from_video(
            video_path,
            f"{OUTPUT_BASE}real",
            FRAMES_PER_REAL_VIDEO,
            f"real_{video_name}"
        )
        total_frames += frames_extracted
    
    print(f"\n✓ Real videos complete: {total_frames} frames extracted")
    return total_frames

def extract_fake_videos():
    print("\n" + "="*60)
    print("EXTRACTING FRAMES FROM DEEPFAKE VIDEOS")
    print("="*60)
    
    if not os.path.exists(FAKE_VIDEO_FOLDER):
        print(f"✗ Deepfake videos folder not found: {FAKE_VIDEO_FOLDER}")
        return 0
    
    fake_videos = sorted([f for f in os.listdir(FAKE_VIDEO_FOLDER) 
                         if f.lower().endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov'))])
    
    if not fake_videos:
        print(f"✗ No videos found in {FAKE_VIDEO_FOLDER}")
        return 0
    
    print(f"Found {len(fake_videos)} deepfake videos")
    print(f"Extracting {FRAMES_PER_FAKE_VIDEO} frames per video...")
    print()
    
    total_frames = 0
    
    for video_file in tqdm(fake_videos, desc="Deepfakes", unit="video"):
        video_path = os.path.join(FAKE_VIDEO_FOLDER, video_file)
        video_name = Path(video_file).stem
        
        frames_extracted = extract_frames_from_video(
            video_path,
            f"{OUTPUT_BASE}fake",
            FRAMES_PER_FAKE_VIDEO,
            f"fake_{video_name}"
        )
        total_frames += frames_extracted
    
    print(f"\n✓ Deepfakes complete: {total_frames} frames extracted")
    return total_frames

def verify_extraction():
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    real_frames = len([f for f in os.listdir(f"{OUTPUT_BASE}real") 
                      if f.endswith('.jpg')])
    fake_frames = len([f for f in os.listdir(f"{OUTPUT_BASE}fake") 
                      if f.endswith('.jpg')])
    
    print(f"Real frames extracted: {real_frames}")
    print(f"Fake frames extracted: {fake_frames}")
    print(f"Total frames: {real_frames + fake_frames}")
    
    if real_frames > 0 and fake_frames > 0:
        ratio = real_frames / fake_frames
        if 2.5 < ratio < 3.5:
            print(f"✓ Good balance for fine-tuning")
        else:
            print(f"⚠ Imbalance detected")
    
    return real_frames, fake_frames

def main():
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + "  PHASE 2: YOUTUBE FRAME EXTRACTION FOR FINE-TUNING  ".center(58) + "║")
    print("╚" + "="*58 + "╝\n")
    
    start_time = time.time()
    
    create_output_folders()
    real_count = extract_real_videos()
    fake_count = extract_fake_videos()
    real_verified, fake_verified = verify_extraction()
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Frames extracted:")
    print(f"  Real:  {real_verified} frames")
    print(f"  Fake:  {fake_verified} frames")
    print(f"  Total: {real_verified + fake_verified} frames")
    print(f"\nTime elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Output location: {OUTPUT_BASE}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
