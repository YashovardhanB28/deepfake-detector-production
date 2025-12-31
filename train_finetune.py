import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BATCH_SIZE = 32
LEARNING_RATE = 0.00001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
NEW_CHECKPOINT_PATH = 'checkpoints/finetuned_model.pth'

def train_finetune():
    print(f'Starting Fine-Tuning on {DEVICE}')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('data/processed/train', transform=transform)
    val_dataset = datasets.ImageFolder('data/processed/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f'Training Data: {len(train_dataset)} images')
    print(f'Validation Data: {len(val_dataset)} images')

    print('Loading ResNet50 architecture...')
    model = models.resnet50(weights=None)
    
    # MATCHING THE SAVED MODEL STRUCTURE
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    print(f'Loading weights...')
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print('Weights loaded!')
    except Exception as e:
        print(f'Error: {e}')
        return

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'   Batch {batch_idx+1}/{len(train_loader)}')

        epoch_acc = 100 * correct / total
        print(f'   Train Acc: {epoch_acc:.2f}%')

        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f'   Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), NEW_CHECKPOINT_PATH)
            print(f'   New best model saved!')

    print('Fine-Tuning Complete!')

if __name__ == '__main__':
    train_finetune()
