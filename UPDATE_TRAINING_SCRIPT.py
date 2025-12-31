"""
UPDATE TRAINING SCRIPT - CHANGE DATA PATH
Purpose: Automatically update MY_TRAINING_SCRIPT.py to use data_clean/
Date: December 28, 2025
Author: AI Assistant
"""

from pathlib import Path

BASE_PATH = Path(r"D:\deepfake_detector_production")
SCRIPT_PATH = BASE_PATH / "MY_TRAINING_SCRIPT.py"

# What to replace
OLD_PATH = '''DATA_PATH = BASE_PATH / "data" / "processed"'''
NEW_PATH = '''DATA_PATH = BASE_PATH / "data" / "data_clean"  # Updated to use clean data (Phase 1)'''

print("="*80)
print("UPDATING TRAINING SCRIPT")
print("="*80)

# Read the script
with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Check if already updated
if "data_clean" in content:
    print("✅ Script already updated to use data_clean/")
    exit()

# Replace the path
if OLD_PATH in content:
    new_content = content.replace(OLD_PATH, NEW_PATH)
    
    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Updated MY_TRAINING_SCRIPT.py")
    print(f"   Changed: {OLD_PATH}")
    print(f"   To:      {NEW_PATH}")
else:
    print("⚠️  Could not find the exact line to update.")
    print(f"   Please manually change in MY_TRAINING_SCRIPT.py:")
    print(f"   From: DATA_PATH = BASE_PATH / 'data' / 'processed'")
    print(f"   To:   DATA_PATH = BASE_PATH / 'data' / 'data_clean'")

print("\n✅ Done! Next: Run FIX_DATA_LEAKAGE_FINAL.py")
