"""
Analysis module for omicsformer.

This module provides comprehensive analysis and interpretation tools for multi-omics transformers.
"""

from .analyzer import (
    MultiOmicsAnalyzer,
    compute_modality_statistics,
    plot_modality_distributions
)

__all__ = [
    'MultiOmicsAnalyzer',
    'compute_modality_statistics',
    'plot_modality_distributions'
]