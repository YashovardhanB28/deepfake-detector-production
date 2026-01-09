import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 3: PRODUCTION TRAINING (12-18 HOURS)
# ============================================================================

class DeepfakeDataset(Dataset):
    """Production-grade dataset with augmentation"""
    def __init__(self, root, split='train', augment=True):
        self.samples = []
        self.augment = augment
        
        real_dir = Path(root) / split / 'real' / 'frames'
        fake_dir = Path(root) / split / 'fake' / 'frames'
        
        # Load ALL available images (not just 100)
        if real_dir.exists():
            real_files = sorted(list(real_dir.glob('*.jpg')))
            print(f"Found {len(real_files)} real images")
            for f in real_files:
                self.samples.append((str(f), 0, 'real'))
        
        if fake_dir.exists():
            fake_files = sorted(list(fake_dir.glob('*.jpg')))
            print(f"Found {len(fake_files)} fake images")
            for f in fake_files:
                self.samples.append((str(f), 1, 'fake'))
        
        real_count = len([s for s in self.samples if s[1]==0])
        fake_count = len([s for s in self.samples if s[1]==1])
        print(f"[{split.upper()}] Real: {real_count} | Fake: {fake_count} | Total: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, label_type = self.samples[idx]
        
        # Read image
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Data augmentation for training
            if self.augment:
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)  # Horizontal flip
                
                # Random brightness
                brightness = np.random.uniform(0.8, 1.2)
                img = np.uint8(np.clip(img * brightness, 0, 255))
                
                # Random contrast
                contrast = np.random.uniform(0.9, 1.1)
                img = np.uint8(np.clip((img - 128) * contrast + 128, 0, 255))
                
                # Random rotation (small angles)
                if np.random.rand() > 0.7:
                    angle = np.random.uniform(-15, 15)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h))
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img = normalize(img)
        
        return img, label


# ============================================================================
# SETUP
# ============================================================================

BASE = Path("D:/deepfake_detector_production")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "="*80)
print("🚀 PHASE 3: PRODUCTION TRAINING (12-18 HOURS)")
print("="*80 + "\n")

print(f"Device: {device}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")


# ============================================================================
# LOAD DATA
# ============================================================================

print("📂 Loading dataset...\n")
train_ds = DeepfakeDataset(str(BASE / "data/processed"), 'train', augment=True)
print()
val_ds = DeepfakeDataset(str(BASE / "data/processed"), 'val', augment=False)

BATCH_SIZE = 32  # Larger batch for stability
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=True
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE, 
    num_workers=0, 
    pin_memory=True
)

print(f"\nTotal training batches per epoch: {len(train_loader)}")
print(f"Total validation batches per epoch: {len(val_loader)}\n")


# ============================================================================
# MODEL SETUP
# ============================================================================

print("🔧 Loading ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# FIXED CODE:
num_classes = 2  # Binary classification: Real (0) or Deepfake (1)

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),  # Add activation
    nn.Dropout(0.5),  # Prevent overfitting
    nn.Linear(512, num_classes)
)



model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")


# ============================================================================
# LOSS & OPTIMIZER
# ============================================================================

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

print(f"Loss Function: CrossEntropyLoss (label_smoothing=0.1)")
print(f"Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)")
print(f"Scheduler: CosineAnnealingLR (T_max=100)\n")


# ============================================================================
# TRAINING LOOP (12-18 HOURS)
# ============================================================================

(BASE / "models").mkdir(exist_ok=True)
(BASE / "logs").mkdir(exist_ok=True)

best_acc = 0
best_loss = float('inf')
patience = 10
patience_counter = 0

NUM_EPOCHS = 100

training_history = {
    'epochs': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rate': []
}

start_time = time.time()

print("="*80)
print("TRAINING STARTED")
print("="*80 + "\n")

try:
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # ===== TRAINING =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [TRAIN]", 
            unit='batch',
            ncols=100
        )
        
        for imgs, lbls in progress_bar:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, lbls)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)
            
            current_acc = train_correct / train_total if train_total > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [VAL]", 
                unit='batch',
                ncols=100
            )
            
            for imgs, lbls in progress_bar:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, lbls)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
                
                current_acc = val_correct / val_total if val_total > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # ===== SCHEDULER STEP =====
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ===== LOGGING =====
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        training_history['epochs'].append(epoch)
        training_history['train_loss'].append(train_loss_avg)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss_avg)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(current_lr)
        
        # Print stats
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | {epoch_time:6.1f}s | Total: {hours:5.1f}h")
        print(f"  Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss_avg:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {current_lr:.2e} | Best: {best_acc:.4f}")
        
        # ===== MODEL SAVING =====
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), BASE / "models" / f"model_best.pth")
            print(f"  ✅ NEW BEST MODEL SAVED! (Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), BASE / "models" / f"model_epoch_{epoch}.pth")
            print(f"  💾 Checkpoint saved: model_epoch_{epoch}.pth")
        
        # Save history every epoch
        with open(BASE / "logs" / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break

except KeyboardInterrupt:
    print(f"\n\n⚠️  Training interrupted by user at epoch {epoch}")

# ============================================================================
# TRAINING COMPLETE
# ============================================================================

total_time = time.time() - start_time
hours = int(total_time / 3600)
minutes = int((total_time % 3600) / 60)
seconds = int(total_time % 60)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n⏱️  Total Training Time: {hours}h {minutes}m {seconds}s")
print(f"📊 Best Validation Accuracy: {best_acc:.4f} (Loss: {best_loss:.4f})")
print(f"📁 Best Model: models/model_best.pth")
print(f"📈 Training History: logs/training_history.json\n")

print("📌 SUMMARY:")
print(f"  • Total Epochs Trained: {epoch}")
print(f"  • Training Samples: {len(train_ds):,}")
print(f"  • Validation Samples: {len(val_ds):,}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Model: ResNet50 (ImageNet pretrained)")
print(f"  • Device: {device}")
print(f"  • Optimizer: AdamW with CosineAnnealing")
print(f"  • Data Augmentation: Enabled (flip, brightness, contrast, rotation)\n")
