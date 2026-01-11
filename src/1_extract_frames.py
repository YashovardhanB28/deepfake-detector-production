"""
INTELLIGENT FRAME EXTRACTION FOR ALL DATASETS
DEBUGGED & PRODUCTION READY VERSION

Extracts frames smartly with 18GB storage limit

Datasets:
1. FaceForensics++ (handles multiple folder structures)
2. Celeb-DF (Celeb-real, YouTube-real, Celeb-synthesis)
3. WildDeepfake (train/val/test splits with real/fake)

Author: Yashovardhan Bangur
Version: PRODUCTION_V1_FIXED
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
from collections import defaultdict

class MultiDatasetFrameExtractor:
    def __init__(self):
        self.base_dir = Path("D:/deepfake_detector_production")
        
        # Source paths
        self.faceforensics_path = Path("D:/deepfake_detector_production/data/faceforensics/archive/FaceForensics++_C23")
        self.celeb_df_path = Path("D:/deepfake_detector_production/data/archive (1)")
        self.wild_deepfake_path = Path("D:/deepfake_detector_production/data/archive")
        
        # Output path
        self.output_dir = self.base_dir / "data" / "all_datasets_frames"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Frame extraction settings (optimized for 18GB)
        self.frames_per_video = 6  # Conservative for storage
        self.img_size = (224, 224)
        
        # Tracking
        self.stats = defaultdict(lambda: {'videos': 0, 'frames': 0, 'errors': 0})
        self.video_registry = {}
        self.used_ids = set()  # Prevent ID collisions
        
    def generate_safe_video_id(self, dataset, category, filename):
        """Generate unique, filesystem-safe video ID"""
        # Clean filename
        clean_name = filename.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Create base ID
        base_id = f"{dataset}_{category}_{clean_name}"
        
        # If collision, add hash
        if base_id in self.used_ids:
            hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
            base_id = f"{dataset}_{category}_{clean_name}_{hash_suffix}"
        
        self.used_ids.add(base_id)
        return base_id
    
    def extract_frames_from_video(self, video_path, output_dir, video_id, label):
        """Extract frames evenly from video with error handling"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return 0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Skip videos with issues
            if total_frames == 0 or fps == 0:
                cap.release()
                return 0
            
            # Skip very short videos (less than 1 second)
            if total_frames < fps:
                cap.release()
                return 0
            
            # Calculate frame indices (evenly spaced)
            if total_frames < self.frames_per_video:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
            
            extracted = 0
            output_dir.mkdir(exist_ok=True, parents=True)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    try:
                        # Resize
                        frame_resized = cv2.resize(frame, self.img_size)
                        
                        # Save with SAFE filename
                        frame_name = f"{video_id}_frame_{idx:06d}.jpg"
                        frame_path = output_dir / frame_name
                        
                        success = cv2.imwrite(str(frame_path), frame_resized, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                        
                        if success:
                            extracted += 1
                    except Exception as e:
                        continue
            
            cap.release()
            return extracted
            
        except Exception as e:
            print(f"  ⚠️ Error with {video_path.name}: {e}")
            return 0
    
    def process_faceforensics(self):
        """Process FaceForensics++ dataset (handles multiple folder structures)"""
        print("\n" + "="*80)
        print("📁 PROCESSING FACEFORENSICS++")
        print("="*80)
        
        if not self.faceforensics_path.exists():
            print(f"⚠️ FaceForensics path not found: {self.faceforensics_path}")
            return
        
        # Try multiple possible folder names (Kaggle vs official)
        folder_mappings = [
            # Official structure
            {
                'original_sequences': 'real',
                'Deepfakes': 'fake',
                'Face2Face': 'fake',
                'FaceSwap': 'fake',
                'NeuralTextures': 'fake',
                'DeepFakeDetection': 'fake'
            },
            # Kaggle structure (lowercase)
            {
                'original': 'real',
                'deepfakes': 'fake',
                'face2face': 'fake',
                'faceswap': 'fake',
                'neuraltextures': 'fake',
                'DeepFakeDetection': 'fake'
            },
            # Mixed structure
            {
                'original': 'real',
                'Deepfakes': 'fake',
                'Face2Face': 'fake',
                'FaceSwap': 'fake',
                'NeuralTextures': 'fake',
                'DeepFakeDetection': 'fake'
            }
        ]
        
        # Find which structure exists
        found_structure = None
        for structure in folder_mappings:
            test_folder = list(structure.keys())[0]
            if (self.faceforensics_path / test_folder).exists():
                found_structure = structure
                break
        
        if not found_structure:
            # Try to auto-detect
            print("  🔍 Auto-detecting folder structure...")
            subdirs = [d for d in self.faceforensics_path.iterdir() if d.is_dir()]
            print(f"  Found folders: {[d.name for d in subdirs]}")
            
            # Auto-create mapping
            found_structure = {}
            for subdir in subdirs:
                name_lower = subdir.name.lower()
                if 'original' in name_lower or 'real' in name_lower:
                    found_structure[subdir.name] = 'real'
                else:
                    found_structure[subdir.name] = 'fake'
            
            if not found_structure:
                print("  ❌ Could not detect structure. Please check folder contents.")
                return
        
        print(f"  ✓ Using structure: {list(found_structure.keys())}")
        
        for folder_name, label in found_structure.items():
            folder_path = self.faceforensics_path / folder_name
            
            if not folder_path.exists():
                continue
            
            # Find videos (might be in subfolders)
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
                video_files.extend(folder_path.rglob(ext))
            
            if not video_files:
                print(f"  ⚠️ No videos in {folder_name}")
                continue
            
            print(f"\n🎬 Processing {folder_name} ({label}) - {len(video_files)} videos")
            
            output_base = self.output_dir / "faceforensics" / label / "frames"
            
            for video_path in tqdm(video_files, desc=f"  {folder_name}"):
                # Generate safe video ID
                video_id = self.generate_safe_video_id(
                    'ff', folder_name, video_path.stem
                )
                
                frames_extracted = self.extract_frames_from_video(
                    video_path, output_base, video_id, label
                )
                
                if frames_extracted > 0:
                    self.stats[f'faceforensics_{label}']['videos'] += 1
                    self.stats[f'faceforensics_{label}']['frames'] += frames_extracted
                    
                    # Register video
                    self.video_registry[video_id] = {
                        'dataset': 'faceforensics',
                        'label': label,
                        'source': folder_name,
                        'frames': frames_extracted,
                        'original_path': str(video_path)
                    }
                else:
                    self.stats[f'faceforensics_{label}']['errors'] += 1
    
    def process_celeb_df(self):
        """Process Celeb-DF dataset"""
        print("\n" + "="*80)
        print("📁 PROCESSING CELEB-DF")
        print("="*80)
        
        if not self.celeb_df_path.exists():
            print(f"⚠️ Celeb-DF path not found: {self.celeb_df_path}")
            return
        
        # Map folders to labels
        folder_mapping = {
            'Celeb-real': 'real',
            'YouTube-real': 'real',
            'Celeb-synthesis': 'fake'
        }
        
        for folder_name, label in folder_mapping.items():
            folder_path = self.celeb_df_path / folder_name
            
            if not folder_path.exists():
                print(f"  ⚠️ Folder not found: {folder_name}")
                continue
            
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
                video_files.extend(list(folder_path.glob(ext)))
            
            if not video_files:
                print(f"  ⚠️ No videos in {folder_name}")
                continue
            
            print(f"\n🎬 Processing {folder_name} ({label}) - {len(video_files)} videos")
            
            output_base = self.output_dir / "celeb_df" / label / "frames"
            
            for video_path in tqdm(video_files, desc=f"  {folder_name}"):
                video_id = self.generate_safe_video_id(
                    'celeb', folder_name, video_path.stem
                )
                
                frames_extracted = self.extract_frames_from_video(
                    video_path, output_base, video_id, label
                )
                
                if frames_extracted > 0:
                    self.stats[f'celeb_df_{label}']['videos'] += 1
                    self.stats[f'celeb_df_{label}']['frames'] += frames_extracted
                    
                    self.video_registry[video_id] = {
                        'dataset': 'celeb_df',
                        'label': label,
                        'source': folder_name,
                        'frames': frames_extracted,
                        'original_path': str(video_path)
                    }
                else:
                    self.stats[f'celeb_df_{label}']['errors'] += 1
    
    def process_wild_deepfake(self):
        """Process WildDeepfake dataset"""
        print("\n" + "="*80)
        print("📁 PROCESSING WILD DEEPFAKE")
        print("="*80)
        
        if not self.wild_deepfake_path.exists():
            print(f"⚠️ WildDeepfake path not found: {self.wild_deepfake_path}")
            return
        
        for split in ['train', 'valid', 'test']:
            for label in ['real', 'fake']:
                folder_path = self.wild_deepfake_path / split / label
                
                if not folder_path.exists():
                    continue
                
                video_files = []
                for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
                    video_files.extend(list(folder_path.glob(ext)))
                
                if not video_files:
                    continue
                
                print(f"\n🎬 Processing {split}/{label} - {len(video_files)} videos")
                
                output_base = self.output_dir / "wild_deepfake" / label / "frames"
                
                for video_path in tqdm(video_files, desc=f"  {split}/{label}"):
                    video_id = self.generate_safe_video_id(
                        'wild', f"{split}_{label}", video_path.stem
                    )
                    
                    frames_extracted = self.extract_frames_from_video(
                        video_path, output_base, video_id, label
                    )
                    
                    if frames_extracted > 0:
                        self.stats[f'wild_deepfake_{label}']['videos'] += 1
                        self.stats[f'wild_deepfake_{label}']['frames'] += frames_extracted
                        
                        self.video_registry[video_id] = {
                            'dataset': 'wild_deepfake',
                            'label': label,
                            'source': split,
                            'frames': frames_extracted,
                            'original_path': str(video_path)
                        }
                    else:
                        self.stats[f'wild_deepfake_{label}']['errors'] += 1
    
    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*80)
        print("📊 EXTRACTION SUMMARY")
        print("="*80)
        
        total_videos = 0
        total_frames = 0
        
        for key, data in sorted(self.stats.items()):
            print(f"\n{key}:")
            print(f"  Videos: {data['videos']}")
            print(f"  Frames: {data['frames']}")
            if data['errors'] > 0:
                print(f"  Errors: {data['errors']} (skipped)")
            
            total_videos += data['videos']
            total_frames += data['frames']
        
        print("\n" + "="*80)
        print(f"✅ TOTAL:")
        print(f"   Videos processed: {total_videos}")
        print(f"   Frames extracted: {total_frames}")
        print(f"   Unique video IDs: {len(self.video_registry)}")
        
        # More accurate size estimate (80KB per frame average)
        avg_size_gb = (total_frames * 0.08) / 1024
        print(f"   Estimated size: ~{avg_size_gb:.2f} GB")
        
        if avg_size_gb > 18:
            print(f"   ⚠️ WARNING: May exceed 18GB! Consider reducing frames_per_video")
        
        print("="*80)
    
    def save_metadata(self):
        """Save video registry for training"""
        metadata_path = self.output_dir / "video_registry.json"
        
        metadata = {
            'video_registry': self.video_registry,
            'stats': dict(self.stats),
            'config': {
                'frames_per_video': self.frames_per_video,
                'img_size': self.img_size,
                'total_videos': len(self.video_registry),
                'total_frames': sum(v['frames'] for v in self.video_registry.values())
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Metadata saved to: {metadata_path}")
    
    def run(self):
        """Run complete extraction"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║       MULTI-DATASET FRAME EXTRACTION (FIXED)                 ║
║                                                              ║
║  📁 FaceForensics++ (auto-detects folder structure)         ║
║  📁 Celeb-DF (3 source folders)                             ║
║  📁 WildDeepfake (train/val/test splits)                    ║
║                                                              ║
║  ⚡ Optimized for 18GB storage                               ║
║  🎯 6 frames per video                                       ║
║  🛡️ Robust error handling                                   ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Extract from all datasets
        self.process_faceforensics()
        self.process_celeb_df()
        self.process_wild_deepfake()
        
        # Summary
        self.print_summary()
        
        # Save metadata
        self.save_metadata()
        
        print("\n✅ Extraction complete! Ready for training.\n")

if __name__ == "__main__":
    try:
        extractor = MultiDatasetFrameExtractor()
        extractor.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Extraction interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
