"""
Training script for SimpleCNN with Batch Normalization
Run this instead of the notebook for more stability!
"""

import os
import sys
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
import config

print("="*60)
print("🚀 SimpleCNN with Batch Normalization - Training Script")
print("="*60)
print(f"🖥️  Device: {config.DEVICE}")
print(f"📦 Batch size: {config.BATCH_SIZE}")
print(f"📈 Learning rate: {config.LEARNING_RATE}")
print(f"🔢 Max Epochs: {config.NUM_EPOCHS}")
print(f"⏸️  Early Stopping Patience: {config.PATIENCE}")
print("="*60 + "\n")


# ============================================================================
# 1. Load Dataset Splits
# ============================================================================
print("📂 Loading dataset splits...")
splits_path = config.RESULTS_DIR / 'dataset_splits.json'

with open(splits_path, 'r') as f:
    splits_data = json.load(f)

train_indices = splits_data['train_indices']
val_indices = splits_data['val_indices']
test_indices = splits_data['test_indices']
class_names = splits_data['class_names']
num_classes = splits_data['num_classes']

print(f"✅ Splits loaded!")
print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")


# ============================================================================
# 2. Create DataLoaders
# ============================================================================
print("🔄 Creating DataLoaders...")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
    transforms.RandomCrop((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

data_path = config.DATA_PATH

train_dataset_full = ImageFolder(root=str(data_path), transform=train_transforms)
train_dataset = Subset(train_dataset_full, train_indices)

val_dataset_full = ImageFolder(root=str(data_path), transform=val_test_transforms)
val_dataset = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

print(f"✅ DataLoaders created!")
print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")


# ============================================================================
# 3. Define Model with Batch Normalization
# ============================================================================
print("🏗️  Building SimpleCNN_BatchNorm model...")

class SimpleCNN_BatchNorm(nn.Module):
    """SimpleCNN with Batch Normalization after each conv layer"""
    
    def __init__(self, num_classes=4):
        super(SimpleCNN_BatchNorm, self).__init__()
        
        # Conv Block 1: 3 → 16 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2: 16 → 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3: 32 → 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size: 224 → 112 → 56 → 28
        self.flatten_size = 28 * 28 * 64
        
        # Fully Connected Layers with Dropout
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


model = SimpleCNN_BatchNorm(num_classes=num_classes).to(config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created!")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}\n")


# ============================================================================
# 4. Setup Loss, Optimizer, Scheduler
# ============================================================================
print("🔧 Setting up training components...")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=5e-4  # Stronger than 02!
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.NUM_EPOCHS,
    eta_min=1e-6
)

print(f"✅ Training setup ready!")
print(f"   Loss: CrossEntropyLoss")
print(f"   Optimizer: Adam (lr={config.LEARNING_RATE}, weight_decay=5e-4)")
print(f"   Scheduler: CosineAnnealingLR\n")


# ============================================================================
# 5. Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ============================================================================
# 6. Training Loop with Early Stopping
# ============================================================================

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

best_val_acc = 0.0
patience_counter = 0
best_model_path = config.MODELS_DIR / 'simple_cnn_batchnorm_best.pth'

print("="*60)
print("🚀 Starting training...")
print("="*60 + "\n")

start_time = time.time()

for epoch in range(config.NUM_EPOCHS):
    print(f"{'='*60}")
    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    print(f"{'='*60}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
    
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
    
    # Update scheduler
    scheduler.step()
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # Print epoch summary
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"   ✅ New best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"   ⏸️  No improvement. Patience: {patience_counter}/{config.PATIENCE}")
        
        if patience_counter >= config.PATIENCE:
            print(f"\n⛔ Early stopping triggered! No improvement for {config.PATIENCE} epochs.")
            break
    
    print()

total_time = time.time() - start_time

print("="*60)
print("🎉 Training completed!")
print(f"⏱️  Total time: {total_time/60:.1f} minutes")
print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")
print("="*60 + "\n")


# ============================================================================
# 7. Save Results
# ============================================================================

print("💾 Saving results...")

results = {
    'model_name': 'SimpleCNN_BatchNorm',
    'total_params': total_params,
    'trainable_params': trainable_params,
    'num_epochs_trained': len(history['train_acc']),
    'best_val_accuracy': best_val_acc,
    'final_train_acc': history['train_acc'][-1],
    'final_val_acc': history['val_acc'][-1],
    'overfitting_gap': history['train_acc'][-1] - history['val_acc'][-1],
    'training_time_minutes': total_time / 60,
    'history': history,
    'hyperparameters': {
        'learning_rate': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
        'weight_decay': 5e-4,
        'scheduler': 'CosineAnnealingLR',
        'dropout_rates': [0.6, 0.5],
        'early_stopping_patience': config.PATIENCE
    }
}

results_path = config.LOGS_DIR / 'simple_cnn_batchnorm_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {results_path}")


# ============================================================================
# 8. Final Summary
# ============================================================================

print("\n" + "="*60)
print("📊 FINAL RESULTS - SimpleCNN + BatchNorm")
print("="*60)
print(f"\n🏆 Best Val Accuracy: {best_val_acc:.2f}%")
print(f"📈 Final Train Acc: {history['train_acc'][-1]:.2f}%")
print(f"📉 Final Val Acc: {history['val_acc'][-1]:.2f}%")
print(f"\n📊 Overfitting Analysis:")
print(f"   Gap: {results['overfitting_gap']:.2f}%")

if results['overfitting_gap'] < 2:
    print("   ✅ Excellent! Minimal overfitting.")
elif results['overfitting_gap'] < 5:
    print("   ✅ Good! Acceptable overfitting.")
else:
    print("   ⚠️  High overfitting. Consider more regularization.")

print(f"\n⏱️  Training Time: {results['training_time_minutes']:.1f} minutes")
print(f"📦 Model Size: {trainable_params:,} parameters")
print(f"\n💾 Best model saved to: {best_model_path}")
print(f"📄 Results saved to: {results_path}")
print("="*60)

print("\n✅ Training script completed successfully! 🎉")
