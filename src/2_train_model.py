"""
MULTI-DATASET DEEPFAKE DETECTOR TRAINING SCRIPT
FIXED: In-place operations + Smart class balancing

Trains on FaceForensics++, Celeb-DF, WildDeepfake combined
Video-level splits to prevent data leakage
Smart sampling for optimal learning (70/30 fake/real ratio)

Author: Yashovardhan Bangur
Version: PRODUCTION_V3_BALANCED
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import random
import time
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Training configuration"""
    # Paths
    BASE_DIR = Path("D:/deepfake_detector_production")
    FRAMES_DIR = BASE_DIR / "data" / "all_datasets_frames"
    METADATA_PATH = FRAMES_DIR / "video_registry.json"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Training
    BATCH_SIZE = 24
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Class balancing (target ratio for fake videos)
    TARGET_FAKE_RATIO = 0.70  # 70% fake, 30% real
    
    # Image
    IMG_SIZE = 224
    
    # Mixed precision
    USE_AMP = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DeepfakeDataset(Dataset):
    """Dataset for deepfake frames"""
    def __init__(self, video_ids, frames_dir, metadata, transform=None):
        self.frames_dir = Path(frames_dir)
        self.metadata = metadata
        self.transform = transform
        
        # Build frame list
        self.samples = []
        
        for video_id in video_ids:
            if video_id not in metadata:
                continue
            
            video_info = metadata[video_id]
            label = 0 if video_info['label'] == 'real' else 1
            dataset = video_info['dataset']
            
            # Find frames for this video
            frame_dir = self.frames_dir / dataset / video_info['label'] / "frames"
            
            if not frame_dir.exists():
                continue
            
            # Get all frames for this video
            frames = sorted(frame_dir.glob(f"{video_id}_frame_*.jpg"))
            
            for frame_path in frames:
                self.samples.append({
                    'path': frame_path,
                    'label': label,
                    'video_id': video_id
                })
        
        print(f"   ✓ Loaded {len(self.samples)} frames")
        
        # Count by label
        real_count = sum(1 for s in self.samples if s['label'] == 0)
        fake_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"   Real: {real_count} ({100*real_count/len(self.samples):.1f}%)")
        print(f"   Fake: {fake_count} ({100*fake_count/len(self.samples):.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return img, label

def get_transforms(split='train'):
    """Get data augmentation transforms"""
    if split == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

class DeepfakeDetector(nn.Module):
    """EfficientNet-B1 based deepfake detector - FIXED for in-place operations"""
    def __init__(self, num_classes=2, dropout=0.3):
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained EfficientNet-B1
        self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1')
        
        # CRITICAL FIX: Disable in-place operations
        self._disable_inplace_ops(self.backbone)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=False),  # CRITICAL: inplace=False
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(512, num_classes)
        )
    
    def _disable_inplace_ops(self, module):
        """Recursively disable in-place operations in the model"""
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.ReLU6, nn.SiLU)):
                setattr(module, child_name, type(child)(inplace=False))
            else:
                self._disable_inplace_ops(child)
    
    def forward(self, x):
        return self.backbone(x)

def balance_dataset(metadata, target_fake_ratio=0.70, seed=42):
    """
    Smart sampling to balance dataset
    Keeps all real videos, samples fake videos to achieve target ratio
    """
    random.seed(seed)
    
    # Separate by label
    real_videos = [vid for vid, info in metadata.items() if info['label'] == 'real']
    fake_videos = [vid for vid, info in metadata.items() if info['label'] == 'fake']
    
    num_real = len(real_videos)
    
    # Calculate how many fake videos we need for target ratio
    # target_fake_ratio = num_fake / (num_fake + num_real)
    # Solving for num_fake: num_fake = num_real * target_fake_ratio / (1 - target_fake_ratio)
    num_fake_needed = int(num_real * target_fake_ratio / (1 - target_fake_ratio))
    
    # Sample fake videos
    if num_fake_needed < len(fake_videos):
        random.shuffle(fake_videos)
        fake_videos_sampled = fake_videos[:num_fake_needed]
        print(f"\n⚖️  SMART SAMPLING:")
        print(f"   Real videos: {num_real} (keeping all)")
        print(f"   Fake videos: {num_fake_needed} (sampled from {len(fake_videos)})")
        print(f"   Target ratio: {target_fake_ratio*100:.0f}% fake / {(1-target_fake_ratio)*100:.0f}% real")
    else:
        fake_videos_sampled = fake_videos
        print(f"\n⚖️  Using all available videos (not enough fake videos to sample)")
    
    return real_videos, fake_videos_sampled

def create_video_level_splits(real_videos, fake_videos, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data by video IDs to prevent leakage"""
    random.seed(seed)
    
    # Shuffle
    real_videos_copy = real_videos.copy()
    fake_videos_copy = fake_videos.copy()
    random.shuffle(real_videos_copy)
    random.shuffle(fake_videos_copy)
    
    # Calculate split points
    real_train_end = int(len(real_videos_copy) * train_ratio)
    real_val_end = int(len(real_videos_copy) * (train_ratio + val_ratio))
    
    fake_train_end = int(len(fake_videos_copy) * train_ratio)
    fake_val_end = int(len(fake_videos_copy) * (train_ratio + val_ratio))
    
    # Split
    train_videos = real_videos_copy[:real_train_end] + fake_videos_copy[:fake_train_end]
    val_videos = real_videos_copy[real_train_end:real_val_end] + fake_videos_copy[fake_train_end:fake_val_end]
    test_videos = real_videos_copy[real_val_end:] + fake_videos_copy[fake_val_end:]
    
    # Shuffle again
    random.shuffle(train_videos)
    random.shuffle(val_videos)
    random.shuffle(test_videos)
    
    return {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }

def calculate_class_weights(dataset_labels):
    """Calculate class weights for loss function"""
    unique, counts = np.unique(dataset_labels, return_counts=True)
    total = len(dataset_labels)
    weights = torch.FloatTensor([total / (len(unique) * count) for count in counts])
    return weights

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} [TRAIN]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress
        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100*accuracy_score(all_labels, all_preds):.2f}%"
        })
    
    # Calculate metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics

@torch.no_grad()
def validate(model, dataloader, criterion, device, split_name='VAL'):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"         [{split_name}]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return metrics

def save_checkpoint(model, optimizer, epoch, metrics, config, filename):
    """Save model checkpoint"""
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'img_size': config.IMG_SIZE
        }
    }
    
    save_path = config.MODELS_DIR / filename
    torch.save(checkpoint, save_path)
    print(f"   💾 Saved: {save_path}")

def train():
    """Main training function"""
    config = Config()
    set_seed(config.SEED)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  MULTI-DATASET DEEPFAKE DETECTOR (BALANCED + FIXED)          ║
║                                                              ║
║  📦 3 Datasets Combined                                      ║
║  🎯 Video-level splits (NO LEAKAGE)                          ║
║  ⚖️  Smart sampling (70/30 fake/real)                        ║
║  🧠 EfficientNet-B1 + Class weights                          ║
║  ⚡ Mixed precision (in-place ops FIXED)                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load metadata
    with open(config.METADATA_PATH, 'r') as f:
        data = json.load(f)
        metadata = data['video_registry']
    
    print("\n" + "="*80)
    print("📊 ORIGINAL DATASET DISTRIBUTION")
    print("="*80)
    
    real_count = sum(1 for v in metadata.values() if v['label'] == 'real')
    fake_count = sum(1 for v in metadata.values() if v['label'] == 'fake')
    print(f"   Real: {real_count} ({100*real_count/(real_count+fake_count):.1f}%)")
    print(f"   Fake: {fake_count} ({100*fake_count/(real_count+fake_count):.1f}%)")
    print(f"   ⚠️  Highly imbalanced - will cause model to predict only 'fake'")
    
    # Balance dataset
    real_videos, fake_videos = balance_dataset(metadata, config.TARGET_FAKE_RATIO, config.SEED)
    
    print("\n" + "="*80)
    print("🎬 CREATING VIDEO-LEVEL SPLITS")
    print("="*80)
    
    # Create splits
    splits = create_video_level_splits(
        real_videos,
        fake_videos,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.SEED
    )
    
    # Print split info
    for split_name in ['train', 'val', 'test']:
        videos = splits[split_name]
        real = sum(1 for v in videos if metadata[v]['label'] == 'real')
        fake = sum(1 for v in videos if metadata[v]['label'] == 'fake')
        print(f"\n{split_name.upper()}:")
        print(f"   Real videos: {real}")
        print(f"   Fake videos: {fake}")
        print(f"   Total videos: {len(videos)}")
        print(f"   Balance: {100*fake/len(videos):.1f}% fake / {100*real/len(videos):.1f}% real")
    
    # Create datasets
    print("\n" + "="*80)
    print("📚 CREATING DATALOADERS")
    print("="*80)
    
    print(f"\n🔍 Loading TRAIN dataset...")
    print(f"   Looking for {len(splits['train'])} videos...")
    train_dataset = DeepfakeDataset(
        splits['train'],
        config.FRAMES_DIR,
        metadata,
        transform=get_transforms('train')
    )
    
    print(f"\n🔍 Loading VAL dataset...")
    print(f"   Looking for {len(splits['val'])} videos...")
    val_dataset = DeepfakeDataset(
        splits['val'],
        config.FRAMES_DIR,
        metadata,
        transform=get_transforms('val')
    )
    
    print(f"\n🔍 Loading TEST dataset...")
    print(f"   Looking for {len(splits['test'])} videos...")
    test_dataset = DeepfakeDataset(
        splits['test'],
        config.FRAMES_DIR,
        metadata,
        transform=get_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    print("\n" + "="*80)
    print("🏗️ BUILDING MODEL")
    print("="*80)
    
    model = DeepfakeDetector(num_classes=2, dropout=0.3)
    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    print(f"   Device: {config.DEVICE}")
    
    # Calculate class weights
    train_labels = [s['label'] for s in train_dataset.samples]
    class_weights = calculate_class_weights(train_labels).to(config.DEVICE)
    print(f"\n   Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE, 'VAL')
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print epoch summary
        print(f"\n   📊 Epoch {epoch} Summary:")
        print(f"      TRAIN - Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.2f}%, F1: {train_metrics['f1']:.4f}")
        print(f"      VAL   - Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, val_metrics, config, 'best_model.pth')
            print(f"      ⭐ New best validation accuracy: {100*best_val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, config, f'checkpoint_epoch_{epoch}.pth')
    
    # Final test
    print("\n" + "="*80)
    print("🧪 FINAL TEST EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(config.MODELS_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, config.DEVICE, 'TEST')
    
    print(f"\n   📊 Test Results:")
    print(f"      Accuracy:  {100*test_metrics['accuracy']:.2f}%")
    print(f"      Precision: {test_metrics['precision']:.4f}")
    print(f"      Recall:    {test_metrics['recall']:.4f}")
    print(f"      F1-Score:  {test_metrics['f1']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"      [[TN FP]")
    print(f"       [FN TP]]")
    print(f"      {test_metrics['confusion_matrix']}")
    
    print("\n✅ Training complete!\n")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
