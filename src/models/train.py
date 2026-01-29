"""
Training pipeline for COVID-19 diagnosis models
Includes: training loop, validation, checkpointing, early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path
import json


class Trainer:
    """
    Trainer class for COVID-19 diagnosis models
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=50,
        save_dir='models/saved',
        model_name='covid_model',
        early_stopping_patience=10
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device (cuda/cpu)
            num_epochs: Number of training epochs
            save_dir: Directory to save models
            model_name: Name for saved model
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Save directory: {save_dir}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save last checkpoint
        last_path = self.save_dir / f'{self.model_name}_last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (val_acc: {self.best_val_acc:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                print(f"   No improvement for {self.early_stopping_patience} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        print("="*70)
        
        # Save training history
        history_path = self.save_dir / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✅ Training history saved to: {history_path}")
        
        return self.history


def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4):
    """
    Get optimizer
    
    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('adam', 'sgd', 'adamw')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        optimizer: PyTorch optimizer
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='plateau', **kwargs):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Scheduler name ('plateau', 'cosine', 'step')
        **kwargs: Additional scheduler parameters
        
    Returns:
        scheduler: PyTorch scheduler
    """
    if scheduler_name.lower() == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=kwargs.get('patience', 3),
            factor=kwargs.get('factor', 0.5)
            # Removed 'verbose=True' - not supported in newer PyTorch
        )
    elif scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    else:
        return None


def get_criterion(class_weights=None, device='cuda'):
    """
    Get loss function
    
    Args:
        class_weights: Weights for each class (for imbalanced datasets)
        device: Device
        
    Returns:
        criterion: Loss function
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()


# Test module
if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Available functions:")
    print("  - Trainer: Main training class")
    print("  - get_optimizer: Create optimizer (adam, sgd, adamw)")
    print("  - get_scheduler: Create LR scheduler (plateau, cosine, step)")
    print("  - get_criterion: Create loss function")