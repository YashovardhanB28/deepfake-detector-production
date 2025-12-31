# 🎯 YOUR D: DRIVE SETUP - Specific Configuration

## Your Storage Confirmed
C: Drive: 30.5 GB free (keep for OS)
D: Drive: 116.7 GB free ✓ (PERFECT for project)

## Directory Structure (On D: Drive)

D:\deepfake_detector_production\
├─ 📁 docs\              (Reference files)
├─ 📁 code\              (Python scripts)
├─ 📁 data\              (Datasets)
├─ 📁 models\            (Trained models)
├─ 📁 results\           (Test results)
├─ 📁 deployment\        (API + Frontend)
├─ 📁 logs\              (Training logs)
├─ 📁 venv\              (Virtual environment)
└─ README.md             (Project overview)

## Python Path Configuration

Base: D:\deepfake_detector_production
CONFIG_ROOT = r"D:\deepfake_detector_production"
DATA_ROOT = os.path.join(CONFIG_ROOT, "data")
MODELS_ROOT = os.path.join(CONFIG_ROOT, "models")

## Advantages of This Setup
✅ C: drive stays clean (OS safe)
✅ D: drive dedicated for project
✅ 116.7 GB perfect for full project
✅ Separate system and data drives
✅ Professional configuration

## Data Organization

data/
├─ faceforensics/         ← 10GB (training)
├─ deepfake_eval_2024/    ← 75GB (validation)
├─ custom_videos/         ← 10GB (domain-specific)
└─ processed/             ← 20GB (extracted + augmented)

## READY TO START
This configuration is optimal for your system!
