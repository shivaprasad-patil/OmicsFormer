"""
Training utilities and trainer classes for multi-omics transformers.

This module provides comprehensive training functionality including:
- Multi-omics trainer with advanced features
- Custom loss functions for multi-modal learning
- Training utilities and callbacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class MultiOmicsTrainer:
    """
    Comprehensive trainer for multi-omics transformer models.
    
    Features:
    - Support for both EnhancedMultiOmicsTransformer and AdvancedMultiOmicsTransformer
    - Custom loss functions including load balancing
    - Advanced metrics tracking and logging
    - Early stopping and learning rate scheduling
    - Gradient clipping and regularization
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None, scheduler: Optional[Any] = None,
                 device: str = 'auto', logger: Optional[logging.Logger] = None,
                 use_wandb: bool = False, project_name: str = 'omicsformer'):
        """
        Args:
            model: Multi-omics transformer model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (Adam if None)
            scheduler: Learning rate scheduler (optional)
            device: Device to use ('auto', 'cpu', 'cuda')
            logger: Logger instance (creates one if None)
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Setup W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name, config={
                'model': self.model.__class__.__name__,
                'optimizer': self.optimizer.__class__.__name__,
                'device': str(self.device)
            })
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Check if model uses MoE (for load balancing)
        self.use_moe = hasattr(model, 'use_moe') and model.use_moe
    
    def train_epoch(self, epoch: int, load_balance_weight: float = 0.01) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        total_lb_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if 'label' in batch:
                labels = batch['label'].squeeze()
                # Get model outputs
                if self.use_moe:
                    outputs = self.model(batch, return_embeddings=False)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Classification loss
                    classification_loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    # Load balancing loss for MoE
                    if isinstance(outputs, dict) and 'load_balance_loss' in outputs:
                        lb_loss = outputs['load_balance_loss']
                        total_lb_loss += lb_loss.item()
                        loss = classification_loss + load_balance_weight * lb_loss
                    else:
                        loss = classification_loss
                else:
                    logits = self.model(batch)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_samples += labels.size(0)
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if self.use_moe:
            metrics['load_balance_loss'] = total_lb_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                if 'label' in batch:
                    labels = batch['label'].squeeze()
                    
                    # Forward pass
                    if self.use_moe:
                        outputs = self.model(batch, return_embeddings=False)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    else:
                        logits = self.model(batch)
                    
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    total_loss += loss.item()
                    
                    # Track predictions
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC for binary classification
        if len(np.unique(all_labels)) == 2:
            probabilities_positive = np.array(all_probabilities)[:, 1]
            auc = roc_auc_score(all_labels, probabilities_positive)
            metrics['auc'] = auc
        
        return metrics
    
    def fit(self, num_epochs: int, early_stopping_patience: int = 10,
            load_balance_weight: float = 0.01, save_best_model: bool = True,
            model_save_path: str = 'best_model.pth') -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            load_balance_weight: Weight for load balancing loss
            save_best_model: Whether to save the best model
            model_save_path: Path to save the best model
            
        Returns:
            Dictionary with training history
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch, load_balance_weight)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validation
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    
                    if save_best_model:
                        torch.save(self.model.state_dict(), model_save_path)
                        self.logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")
                else:
                    self.patience_counter += 1
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
                
                # W&B logging
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy']
                    })
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}"
                )
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy']
                    })
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
        
        return history
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the appropriate device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
            else:
                moved_batch[key] = value
        return moved_batch
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multi-modal representation learning."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Labels for contrastive learning (diagonal should be positive)
        batch_size = embeddings1.size(0)
        labels = torch.arange(batch_size).to(embeddings1.device)
        
        # Compute contrastive loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss


def create_optimizer(model: nn.Module, optimizer_name: str = 'adamw', 
                    learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    """Create optimizer for the model."""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'cosine', 
                    **kwargs) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'auto') -> Dict[str, float]:
    """Evaluate model on test data."""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            moved_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(device)
                else:
                    moved_batch[key] = value
            
            if 'label' in moved_batch:
                labels = moved_batch['label'].squeeze()
                
                # Get predictions
                if hasattr(model, 'use_moe') and model.use_moe:
                    outputs = model(moved_batch)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                else:
                    logits = model(moved_batch)
                
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Add AUC for binary classification
    if len(np.unique(all_labels)) == 2:
        probabilities_positive = np.array(all_probabilities)[:, 1]
        auc = roc_auc_score(all_labels, probabilities_positive)
        metrics['auc'] = auc
    
    return metrics