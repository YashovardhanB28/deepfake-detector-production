"""
SETUP CHECKER
Verifies everything is ready before training

Run this FIRST to avoid issues!

Author: Yashovardhan Bangur
"""

import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (Need Python 3.8+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (not installed)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🎮 Checking GPU/CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ⚠️  CUDA not available (will use CPU - much slower)")
            print("   Consider installing CUDA from: https://pytorch.org/get-started/locally/")
            return False
    except:
        print("   ❌ Cannot check CUDA (PyTorch not installed)")
        return False

def check_datasets():
    """Check if datasets exist"""
    print("\n📁 Checking datasets...")
    
    datasets = {
        'FaceForensics++': Path("D:/deepfake_detector_production/data/faceforensics/archive/FaceForensics++_C23"),
        'Celeb-DF': Path("D:/deepfake_detector_production/data/archive (1)"),
        'WildDeepfake': Path("D:/deepfake_detector_production/data/archive")
    }
    
    all_found = True
    
    for name, path in datasets.items():
        if path.exists():
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mov']:
                video_count += len(list(path.rglob(ext)))
            
            print(f"   ✅ {name}: {path}")
            print(f"      Found ~{video_count} video files")
        else:
            print(f"   ❌ {name}: {path} (not found)")
            all_found = False
    
    return all_found

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        base_path = Path("D:/deepfake_detector_production")
        
        if base_path.exists():
            total, used, free = shutil.disk_usage(base_path.drive)
            free_gb = free / (1024**3)
            
            print(f"   Free space: {free_gb:.1f} GB")
            
            if free_gb >= 18:
                print(f"   ✅ Sufficient space (need ~18GB)")
                return True
            else:
                print(f"   ⚠️  Low space (need ~18GB, have {free_gb:.1f}GB)")
                print(f"      Consider:")
                print(f"      - Reducing frames_per_video in extraction script")
                print(f"      - Using only 2 datasets instead of 3")
                return False
        else:
            print("   ⚠️  Cannot check (base directory doesn't exist)")
            return False
    except Exception as e:
        print(f"   ⚠️  Cannot check disk space: {e}")
        return False

def check_directories():
    """Create necessary directories"""
    print("\n📂 Checking/creating directories...")
    
    base = Path("D:/deepfake_detector_production")
    dirs = [
        base / "data" / "all_datasets_frames",
        base / "models",
        base / "logs"
    ]
    
    for dir_path in dirs:
        try:
            dir_path.mkdir(exist_ok=True, parents=True)
            print(f"   ✅ {dir_path}")
        except Exception as e:
            print(f"   ❌ {dir_path}: {e}")
            return False
    
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       DEEPFAKE DETECTOR SETUP CHECKER (FIXED VERSION)        ║
║                                                              ║
║  Run this BEFORE starting extraction/training                ║
║  Checks: Python, packages, GPU, datasets, disk space         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    checks = {
        'Python version': check_python_version(),
        'Required packages': check_packages(),
        'GPU/CUDA': check_cuda(),
        'Datasets': check_datasets(),
        'Disk space': check_disk_space(),
        'Directories': check_directories()
    }
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed and name in ['Python version', 'Required packages', 'Datasets']:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All critical checks passed!")
        print("   You're ready to run:")
        print("   1. python 1_extract_frames_all_datasets.py")
        print("   2. python 2_train_all_datasets.py")
        print("   3. python 3_test_detector.py")
    else:
        print("\n⚠️  Some checks failed!")
        print("   Fix the issues above before proceeding.")
        
        if not checks['Required packages']:
            print("\n   Quick fix:")
            print("   pip install torch torchvision opencv-python Pillow numpy scikit-learn tqdm")
        
        if not checks['Datasets']:
            print("\n   Make sure you've downloaded all datasets to:")
            print("   - D:/deepfake_detector_production/data/faceforensics/archive/FaceForensics++_C23")
            print("   - D:/deepfake_detector_production/data/archive (1)")
            print("   - D:/deepfake_detector_production/data/archive")
    
    print("\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during checks: {e}")
        import traceback
        traceback.print_exc()