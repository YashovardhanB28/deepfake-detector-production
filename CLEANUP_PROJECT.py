"""
CLEANUP SCRIPT - REMOVE ALL MISLEADING/OLD FILES
Purpose: Clean up your project folder before Phase 1 data fix
Date: December 28, 2025
Author: AI Assistant

WHAT THIS REMOVES:
1. Old test scripts (FINAL_test.py, test_real_deepfakes.py, etc.) - these confused you
2. Old/corrupted models (model_v1-v5) - keep only best_model
3. Old logs and metrics - start fresh
4. Temporary files and debug outputs
5. OLD data folders - we're creating data_clean/

WHAT THIS KEEPS:
✅ MY_TRAINING_SCRIPT.py - needed to retrain
✅ best_model.pth checkpoint - your baseline
✅ FaceForensics++_C23 videos - raw source data
✅ diagnose_now_FIXED.py - for verification
✅ All .txt summary files
"""

import os
import shutil
from pathlib import Path
import json

BASE_PATH = Path(r"D:\deepfake_detector_production")

# Files/folders to DELETE (they mislead you)
TO_DELETE = [
    # Old test scripts - CONFUSING
    BASE_PATH / "FINAL_test.py",
    BASE_PATH / "test_model.py",
    BASE_PATH / "test_real_deepfakes.py",
    BASE_PATH / "test_real_deepfakes_FIXED.py",
    BASE_PATH / "run_test.py",
    
    # Old model files - KEEP ONLY BEST
    BASE_PATH / "models" / "model_v1.pth",
    BASE_PATH / "models" / "model_v2.pth",
    BASE_PATH / "models" / "model_v3.pth",
    BASE_PATH / "models" / "model_v4.pth",
    BASE_PATH / "models" / "model_v5.pth",
    
    # Old logs - START FRESH
    BASE_PATH / "training_log.txt",
    BASE_PATH / "MY_TRAINING_LOGS.txt",
    
    # Debug outputs - CLUTTER
    BASE_PATH / "debug_output.txt",
    BASE_PATH / "extraction_log.txt",
    
    # Old metrics
    BASE_PATH / "metrics" / "metrics.json",
    
    # Other confusing test scripts
    BASE_PATH / "compare_all.py",
    BASE_PATH / "debug_models.py",
    BASE_PATH / "diagnostic_results.py",
    BASE_PATH / "find_accuracy.py",
    BASE_PATH / "perfect_compare.py",
    BASE_PATH / "phase2_verify.py",
    
    # The corrupted data folder (we're replacing with data_clean)
    # DON'T DELETE YET - we need to scan it for videos first
    # BASE_PATH / "data" / "processed",
]

# Folders that are OK to clear
CLEAR_FOLDERS = [
    BASE_PATH / "metrics",  # Clear metrics, not delete
]

print("="*80)
print("CLEANUP SCRIPT - REMOVE MISLEADING FILES")
print("="*80)

# Show what will be deleted
print("\n🗑️  FILES TO DELETE:\n")
files_to_delete = []
for path in TO_DELETE:
    if path.exists():
        files_to_delete.append(path)
        size = path.stat().st_size if path.is_file() else "folder"
        print(f"  ❌ {path.name} ({size})")

print(f"\nTotal files to delete: {len(files_to_delete)}")

# Show what will be cleared
print("\n📁 FOLDERS TO CLEAR:\n")
for folder in CLEAR_FOLDERS:
    if folder.exists():
        print(f"  🔄 {folder}")

# Show what will be KEPT
print("\n✅ FILES TO KEEP:\n")
keep_files = [
    "MY_TRAINING_SCRIPT.py",
    "diagnose_now_FIXED.py",
    "MY_PROJECT_SUMMARY.txt",
    "MY_TEST_RESULTS.txt",
    "MY_TRAINING_LOGS.txt",
    "checkpoints/best_model.pth",
    "models/model_best.pth",
    "data/faceforensics/FaceForensics++_C23 (ALL VIDEOS)"
]
for f in keep_files:
    print(f"  ✅ {f}")

# Ask for confirmation
print("\n" + "="*80)
response = input("🔴 ARE YOU SURE? This will DELETE files permanently. (type 'YES' to confirm): ")

if response.upper() != "YES":
    print("❌ Cleanup cancelled. No files deleted.")
    exit()

print("\n⏳ Starting cleanup...\n")

# Delete files
deleted_count = 0
for path in files_to_delete:
    try:
        if path.is_file():
            os.remove(path)
            print(f"✅ Deleted: {path.name}")
            deleted_count += 1
        elif path.is_dir():
            shutil.rmtree(path)
            print(f"✅ Deleted folder: {path.name}/")
            deleted_count += 1
    except Exception as e:
        print(f"⚠️  Could not delete {path.name}: {e}")

# Clear folders (but keep them)
cleared_count = 0
for folder in CLEAR_FOLDERS:
    try:
        if folder.exists():
            for file in folder.glob("*"):
                if file.is_file():
                    os.remove(file)
                    cleared_count += 1
                elif file.is_dir():
                    shutil.rmtree(file)
                    cleared_count += 1
            print(f"✅ Cleared folder: {folder.name}/")
    except Exception as e:
        print(f"⚠️  Could not clear {folder.name}: {e}")

print("\n" + "="*80)
print(f"✅ CLEANUP COMPLETE!")
print(f"   Deleted: {deleted_count} items")
print(f"   Cleared: {cleared_count} items")
print("="*80)

print("\n📋 YOUR PROJECT IS NOW CLEAN:")
print("   - ❌ No confusing test scripts")
print("   - ❌ No old model versions")
print("   - ❌ No misleading logs")
print("   - ✅ Ready for Phase 1 (Data Fix)")
print("\n🚀 Next step: Run FIX_DATA_LEAKAGE_FINAL.py")
