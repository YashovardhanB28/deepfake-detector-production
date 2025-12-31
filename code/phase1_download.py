import os
import requests
import subprocess
import zipfile
from pathlib import Path

base_dir = Path("D:/deepfake_detector_production")
data_dir = base_dir / "data"
ff_dir = data_dir / "faceforensics"
df_dir = data_dir / "deepfake_eval_2024"

os.makedirs(ff_dir, exist_ok=True)
os.makedirs(df_dir, exist_ok=True)

print("🚀 Phase 1: Dataset Download Started")
print(f"Base: {base_dir}")

# 1. FaceForensics++ (Kaggle C23 subset - immediate access, 10GB)
print("\n📥 Step 1: FaceForensics++ (Kaggle C23)")
kaggle_cmd = [
    "kaggle", "datasets", "download", "-d", "xdxd003/ff-c23",
    "-p", str(ff_dir), "--unzip"
]
result = subprocess.run(kaggle_cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("❌ Kaggle error. Install: pip install kaggle")
    print("Upload API key: ~/.kaggle/kaggle.json")
else:
    print("✅ FaceForensics++ downloaded")

# 2. Deepfake-Eval-2024 (HuggingFace - validation set)
print("\n📥 Step 2: Deepfake-Eval-2024 (HuggingFace)")
subprocess.run(["pip", "install", "datasets"], cwd=base_dir)
import datasets
ds = datasets.load_dataset("nuriachandra/Deepfake-Eval-2024", split="train[:1000]")
ds.to_parquet(str(df_dir / "deepfake_eval.parquet"))
print("✅ Deepfake-Eval-2024 sampled (1k items for validation)")

# 3. Verify structure
print("\n🔍 Folder check:")
for d in [ff_dir, df_dir]:
    print(f"{d}: {len(list(d.rglob('*')))} files")

print("✅ Phase 1 COMPLETE - Datasets ready!")
