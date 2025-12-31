"""
DIAGNOSTIC TEST: TinyNet vs ResNet50
Purpose: Determine ROOT CAUSE (DATA vs OVERFITTING)
Runtime: ~4-6 hours on RTX 4060
EDIT ONLY LINES 25-28 WITH YOUR PATHS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms

print("\n" + "="*80)
print("DIAGNOSTIC TEST: TinyNet vs ResNet50")
print("="*80)

# ==============================================================================
# EDIT THESE 4 PATHS TO YOUR ACTUAL FOLDERS (LINES 25-28)
# ==============================================================================
TRAIN_REAL_DIR = r"D:\deepfake_detector_production\data\processed\train\real\frames"
TRAIN_FAKE_DIR = r"D:\deepfake_detector_production\data\processed\train\fake\frames"
VAL_REAL_DIR   = r"D:\deepfake_detector_production\data\processed\val\real\frames"
VAL_FAKE_DIR   = r"D:\deepfake_detector_production\data\processed\val\fake\frames"

SUBSET_SIZE = 10000  # 10K real + 10K fake (for speed)
BATCH_SIZE  = 32
EPOCHS      = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📋 CONFIG:")
print(f"  Train Real: {TRAIN_REAL_DIR}")
print(f"  Train Fake: {TRAIN_FAKE_DIR}")
print(f"  Val Real:   {VAL_REAL_DIR}")
print(f"  Val Fake:   {VAL_FAKE_DIR}")
print(f"  Device:     {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")

# ==============================================================================
# TINYNET MODEL (3-layer CNN, ~50K params)
# ==============================================================================
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

# ==============================================================================
# DATASET CLASS
# ==============================================================================
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, max_samples=None, augment=False):
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.augment = augment
        
        self.real_images = sorted([p for p in self.real_dir.glob("*") 
                                 if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        self.fake_images = sorted([p for p in self.fake_dir.glob("*") 
                                 if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        
        if max_samples:
            self.real_images = self.real_images[:max_samples]
            self.fake_images = self.fake_images[:max_samples]
            
        self.samples = [(p, 0) for p in self.real_images] + [(p, 1) for p in self.fake_images]
        
        print(f"  Loaded: {len(self.real_images)} real, {len(self.fake_images)} fake")
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="[TRAIN]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.1%}")
    
    return total_loss/len(loader), correct/total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="[VAL]")
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.1%}")
    
    return total_loss/len(loader), correct/total

# ==============================================================================
# DIAGNOSIS LOGIC
# ==============================================================================
def diagnose(resnet_train=0.8326, resnet_val=0.5541, tinynet_results=None):
    r_gap = resnet_train - resnet_val
    t_train = tinynet_results["train_accs"][-1]
    t_val = tinynet_results["val_accs"][-1]
    t_gap = t_train - t_val
    
    print("\n"*2 + "="*80)
    print("🎯 DIAGNOSTIC RESULTS")
    print("="*80)
    print(f"ResNet50: Train {resnet_train:.1%} | Val {resnet_val:.1%} | Gap {r_gap:.1%}")
    print(f"TinyNet:  Train {t_train:.1%}   | Val {t_val:.1%}   | Gap {t_gap:.1%}")
    print("-"*80)
    
    if t_val < 0.62:
        print("🔴 DATA PROBLEM")
        print("Both models fail → Your dataset is the issue")
        return "DATA_PROBLEM"
    elif t_val > 0.72 and resnet_val < 0.65:
        print("🟡 OVERFITTING PROBLEM") 
        print("TinyNet good, ResNet50 bad → ResNet50 memorizing data")
        return "OVERFITTING_PROBLEM"
    elif t_val > 0.78:
        print("✅ GOOD DATA")
        print("Both models good → Continue training + ensemble")
        return "GOOD_DATA"
    else:
        print("🔵 MIXED PROBLEM")
        print("Data + model both need work")
        return "MIXED_PROBLEM"

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("\n📂 STEP 1: Loading training data (10K subset)...")
    train_ds = DeepfakeDataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, SUBSET_SIZE, augment=True)
    train_loader = DataLoader(train_ds, BATCH_SIZE, True, num_workers=4)
    
    print("📂 STEP 2: Loading validation data...")
    val_ds = DeepfakeDataset(VAL_REAL_DIR, VAL_FAKE_DIR, augment=False)
    val_loader = DataLoader(val_ds, BATCH_SIZE, False, num_workers=4)
    
    print("🔧 STEP 3: Creating TinyNet...")
    model = TinyNet().to(DEVICE)
    print(f"   {sum(p.numel() for p in model.parameters()):,} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("🚀 STEP 4: Training (5 epochs)...")
    results = {"train_accs":[], "val_accs":[], "train_losses":[], "val_losses":[]}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_l, va_a = val_epoch(model, val_loader, criterion, DEVICE)
        
        results["train_accs"].append(tr_a)
        results["val_accs"].append(va_a)
        results["train_losses"].append(tr_l)
        results["val_losses"].append(va_l)
        
        print(f"📊 Train: {tr_a:.1%} | Val: {va_a:.1%} | Gap: {tr_a-va_a:.1%}")
    
    diagnosis = diagnose(tinynet_results=results)
    
    # Save results
    output = {
        "diagnosis": diagnosis,
        "resnet50": {"train_acc": 0.8326, "val_acc": 0.5541},
        "tinynet": results,
        "paths": { "train_real": TRAIN_REAL_DIR, "train_fake": TRAIN_FAKE_DIR,
                  "val_real": VAL_REAL_DIR, "val_fake": VAL_FAKE_DIR }
    }
    with open("diagnostic_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ COMPLETE! Diagnosis: {diagnosis}")
    print(f"✅ Results saved: diagnostic_results.json")
    print("\nNext: Check MANDATORY_ACTION_PLAN.md for your diagnosis path!")
