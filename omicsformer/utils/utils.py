"""
Utility functions and helper classes for omicsformer.

This module provides various utility functions for data processing,
model utilities, and common operations.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import random
import os
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        PyTorch device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, Union[int, str]]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = param_size + buffer_size
    
    def format_bytes(bytes_val):
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} TB"
    
    return {
        'model_size_bytes': model_size,
        'model_size_formatted': format_bytes(model_size),
        'param_size_bytes': param_size,
        'param_size_formatted': format_bytes(param_size),
        'buffer_size_bytes': buffer_size,
        'buffer_size_formatted': format_bytes(buffer_size)
    }


def save_model_config(model: torch.nn.Module, config: Dict[str, Any], 
                     save_path: str) -> None:
    """
    Save model configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    full_config = {
        'model_class': model.__class__.__name__,
        'model_config': config,
        'parameter_count': count_parameters(model),
        'model_size': get_model_size(model),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(full_config, f, indent=2, default=str)


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_logger(name: str, level: int = logging.INFO, 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(results: Dict[str, Any], save_path: str) -> None:
    """
    Save results dictionary to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save the results
    """
    save_path = Path(save_path)
    
    if save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif save_path.suffix == '.pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported file format: {save_path.suffix}")


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Results dictionary
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            results = json.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return results


def validate_modality_data(modality_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    Validate multi-omics data and return validation results.
    
    Args:
        modality_data: Dictionary mapping modality names to DataFrames
        
    Returns:
        Dictionary with validation issues
    """
    issues = {}
    
    for modality, df in modality_data.items():
        modality_issues = []
        
        # Check for empty data
        if df.empty:
            modality_issues.append("DataFrame is empty")
        
        # Check for all NaN columns
        nan_cols = df.columns[df.isnull().all()].tolist()
        if nan_cols:
            modality_issues.append(f"Columns with all NaN values: {nan_cols}")
        
        # Check for constant columns
        constant_cols = df.columns[df.nunique() <= 1].tolist()
        if constant_cols:
            modality_issues.append(f"Constant columns: {constant_cols}")
        
        # Check for high missing value percentage
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            modality_issues.append(f"Columns with >50% missing values: {high_missing}")
        
        if modality_issues:
            issues[modality] = modality_issues
    
    return issues


def compute_memory_usage(data: Union[pd.DataFrame, torch.Tensor, Dict[str, Any]]) -> str:
    """
    Compute memory usage of data structures.
    
    Args:
        data: Data structure to analyze
        
    Returns:
        Formatted memory usage string
    """
    if isinstance(data, pd.DataFrame):
        memory_bytes = data.memory_usage(deep=True).sum()
    elif isinstance(data, torch.Tensor):
        memory_bytes = data.element_size() * data.nelement()
    elif isinstance(data, dict):
        memory_bytes = sum(compute_memory_usage_bytes(v) for v in data.values())
    else:
        # Fallback using sys.getsizeof
        import sys
        memory_bytes = sys.getsizeof(data)
    
    # Format bytes
    for unit in ['B', 'KB', 'MB', 'GB']:
        if memory_bytes < 1024.0:
            return f"{memory_bytes:.2f} {unit}"
        memory_bytes /= 1024.0
    return f"{memory_bytes:.2f} TB"


def compute_memory_usage_bytes(data: Any) -> int:
    """Helper function to compute memory usage in bytes."""
    if isinstance(data, pd.DataFrame):
        return data.memory_usage(deep=True).sum()
    elif isinstance(data, torch.Tensor):
        return data.element_size() * data.nelement()
    elif isinstance(data, (list, tuple)):
        return sum(compute_memory_usage_bytes(item) for item in data)
    elif isinstance(data, dict):
        return sum(compute_memory_usage_bytes(v) for v in data.values())
    else:
        import sys
        return sys.getsizeof(data)


class EarlyStopping:
    """
    Early stopping utility class.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class ModelCheckpoint:
    """
    Model checkpoint utility class.
    """
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, mode: str = 'min'):
        """
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            save_best_only: Whether to only save the best model
            mode: 'min' or 'max' for the monitored metric
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        ensure_dir(Path(filepath).parent)
    
    def __call__(self, metrics: Dict[str, float], model: torch.nn.Module, 
                 epoch: int) -> bool:
        """
        Check if model should be saved.
        
        Args:
            metrics: Dictionary of metrics
            model: Model to save
            epoch: Current epoch
            
        Returns:
            True if model was saved, False otherwise
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return False
        
        if self.save_best_only:
            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )
            
            if is_better:
                self.best_value = current_value
                self._save_model(model, epoch, metrics)
                return True
        else:
            self._save_model(model, epoch, metrics)
            return True
        
        return False
    
    def _save_model(self, model: torch.nn.Module, epoch: int, 
                   metrics: Dict[str, float]) -> None:
        """Save model with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.filepath)


def print_model_summary(model: torch.nn.Module, input_size: Optional[Tuple] = None) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for parameter counting
    """
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    
    # Parameter counts
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    
    # Model size
    size_info = get_model_size(model)
    print(f"Model size: {size_info['model_size_formatted']}")
    
    print("=" * 60)
    
    # Layer information
    print("Model Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module.__class__.__name__} ({num_params:,} params)")


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move a batch dictionary to the specified device.
    
    Args:
        batch: Batch dictionary
        device: Target device
        
    Returns:
        Batch dictionary with tensors moved to device
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = batch_to_device(value, device)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            moved_batch[key] = [tensor.to(device) for tensor in value]
        else:
            moved_batch[key] = value
    return moved_batch