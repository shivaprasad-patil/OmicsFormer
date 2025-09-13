"""
Training module for omicsformer.

This module provides comprehensive training functionality for multi-omics transformers.
"""

from .trainer import (
    MultiOmicsTrainer,
    FocalLoss,
    ContrastiveLoss,
    create_optimizer,
    create_scheduler,
    evaluate_model
)

__all__ = [
    'MultiOmicsTrainer',
    'FocalLoss',
    'ContrastiveLoss',
    'create_optimizer',
    'create_scheduler',
    'evaluate_model'
]