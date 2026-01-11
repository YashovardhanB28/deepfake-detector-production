"""
Repository Cleanup Script
Organizes messy files into proper structure
"""

from pathlib import Path
import shutil

def cleanup_repo():
    base = Path("D:/deepfake_detector_production")
    
    # Create proper structure
    dirs_to_create = [
        "src",
        "old_versions",
        "documentation"
    ]
    
    for d in dirs_to_create:
        (base / d).mkdir(exist_ok=True)
    
    print("🧹 Cleaning up repository...\n")
    
    # Files to keep in root
    keep_in_root = {
        'frame_extractor_fixed.py': 'src/1_extract_frames.py',
        'train_all_datasets_fixed1.py': 'src/2_train_model.py',
        'test_any_video1.py': 'src/3_test_video.py',
        'setup_checker.py': 'src/setup_checker.py',
        'requirements.txt': 'requirements.txt',
        '.gitignore': '.gitignore'
    }
    
    # Move and rename important files
    print("📦 Organizing main files...")
    for old_name, new_path in keep_in_root.items():
        old_file = base / old_name
        new_file = base / new_path
        
        if old_file.exists():
            shutil.copy2(old_file, new_file)
            print(f"  ✓ {old_name} → {new_path}")
    
    # Move documentation
    print("\n📚 Organizing documentation...")
    docs_to_move = [
        'quickstart_guide.md',
        'usage_guide.md',
        'bugs_fixed_doc.md'
    ]
    
    for doc in docs_to_move:
        old_file = base / doc
        new_file = base / 'documentation' / doc
        if old_file.exists():
            shutil.copy2(old_file, new_file)
            print(f"  ✓ {doc} → documentation/{doc}")
    
    # Move old versions
    print("\n🗄️  Archiving old versions...")
    old_versions = [
        'deepfake_fixed_training.py',
        'deepfake_fixed_training (1).py',
        'train_all_datasets.py',
        'train_all_datasets_fixed.py',
        'train_FIXED_OVERFITTING.py',
        'train_ensemble.py',
        'train_production.py',
        'train_simple.py',
        'train_ultimate.py',
        'train.py',
        'evaluate.py',
        'evaluate_fast.py',
        'extract_frames.py',
        'extract_frames_BETTER.py',
        'extract_frames_smart.py',
        'test_detector.py',
        'test_any_video.py',
        'app.py',
        'diagnose.py',
        'check.py',
        'check_gpu.py',
        'fix_dataset.py',
        'reduce_frames.py'
    ]
    
    for old_file in old_versions:
        src = base / old_file
        dst = base / 'old_versions' / old_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {old_file}")
    
    # Create README for old_versions
    readme = base / 'old_versions' / 'README.md'
    readme.write_text("""# Old Versions

These are previous iterations and experimental scripts.
**Use the files in /src/ for current implementation.**

Archive date: January 2026
""")
    
    print("\n✅ Cleanup complete!")
    print("\nNew structure:")
    print("  /src/ - Main implementation files")
    print("  /documentation/ - Guides and docs")
    print("  /old_versions/ - Archived experiments")
    print("  /data/ - Training data")
    print("  /models/ - Trained models")
    print("  /analysis_results/ - Test outputs")

if __name__ == "__main__":
    cleanup_repo()
