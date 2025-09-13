"""
Data handling module for omicsformer.

This module provides flexible dataset classes and utilities for multi-omics data integration.
"""

from .dataset import (
    FlexibleMultiOmicsDataset,
    FlexibleMultiSampleDataset,
    create_synthetic_multiomics_data,
    normalize_modality_data,
    filter_features_by_variance,
    align_modalities_by_samples
)

__all__ = [
    'FlexibleMultiOmicsDataset',
    'FlexibleMultiSampleDataset',
    'create_synthetic_multiomics_data',
    'normalize_modality_data',
    'filter_features_by_variance',
    'align_modalities_by_samples'
]