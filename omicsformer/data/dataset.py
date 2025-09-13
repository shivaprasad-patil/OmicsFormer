"""
Flexible data handling for multi-omics integration.

This module provides dataset classes for various sample alignment strategies:
- FlexibleMultiOmicsDataset: Basic multi-omics dataset with alignment options
- FlexibleMultiSampleDataset: Advanced dataset with complex sample relationships
- Multi-omics data utilities and transformations
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
import warnings


class FlexibleMultiOmicsDataset(Dataset):
    """
    Dataset for multi-omics data with flexible sample alignment strategies.
    
    Supports multiple alignment strategies:
    - 'strict': Only samples present in all modalities
    - 'flexible': All samples (use missing tokens for absent modalities)
    - 'intersection': Samples in at least one modality (intersection-based)
    - 'union': All samples across all modalities
    """
    
    def __init__(self, modality_data: Dict[str, pd.DataFrame], labels: Optional[pd.Series] = None,
                 sample_col: str = 'sample_id', alignment: str = 'flexible',
                 missing_value_strategy: str = 'zero', normalize: bool = True,
                 feature_selection: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            modality_data: Dict mapping modality names to DataFrames (samples x features)
            labels: Series with sample labels (optional for unsupervised)
            sample_col: Column name for sample identifiers
            alignment: Sample alignment strategy ('strict', 'flexible', 'intersection', 'union')
            missing_value_strategy: How to handle missing values ('zero', 'mean', 'median')
            normalize: Whether to normalize features
            feature_selection: Optional dict mapping modalities to selected feature lists
        """
        self.modality_data = {}
        self.labels = labels
        self.sample_col = sample_col
        self.alignment = alignment
        self.missing_value_strategy = missing_value_strategy
        self.normalize = normalize
        
        # Process modality data
        for mod, df in modality_data.items():
            # Feature selection if specified
            if feature_selection and mod in feature_selection:
                selected_features = [col for col in feature_selection[mod] if col in df.columns]
                if selected_features:
                    df = df[selected_features]
                else:
                    warnings.warn(f"No selected features found for modality {mod}")
            
            # Set sample ID as index if it's a column
            if sample_col in df.columns:
                df = df.set_index(sample_col)
            
            self.modality_data[mod] = df
        
        # Determine sample alignment
        self._align_samples()
        
        # Process labels
        if labels is not None:
            if hasattr(labels, 'index'):
                # Align labels with samples
                self.labels = labels.reindex(self.sample_ids).fillna(-1)  # -1 for missing labels
            else:
                # Assume labels are already aligned
                self.labels = pd.Series(labels, index=self.sample_ids)
        
        # Prepare data tensors
        self._prepare_tensors()
    
    def _align_samples(self):
        """Determine which samples to include based on alignment strategy."""
        all_samples = set()
        modality_samples = {}
        
        for mod, df in self.modality_data.items():
            mod_samples = set(df.index)
            modality_samples[mod] = mod_samples
            all_samples.update(mod_samples)
        
        if self.alignment == 'strict':
            # Only samples present in ALL modalities
            self.sample_ids = list(all_samples)
            for mod_samples in modality_samples.values():
                self.sample_ids = [s for s in self.sample_ids if s in mod_samples]
        
        elif self.alignment == 'flexible':
            # All samples from all modalities
            self.sample_ids = list(all_samples)
        
        elif self.alignment == 'intersection':
            # Samples present in at least 2 modalities
            sample_counts = defaultdict(int)
            for mod_samples in modality_samples.values():
                for sample in mod_samples:
                    sample_counts[sample] += 1
            
            self.sample_ids = [s for s, count in sample_counts.items() if count >= 2]
        
        elif self.alignment == 'union':
            # Union of all samples
            self.sample_ids = list(all_samples)
        
        else:
            raise ValueError(f"Unknown alignment strategy: {self.alignment}")
        
        self.sample_ids.sort()  # For reproducibility
        
        # Track which samples are available for each modality
        self.sample_masks = {}
        for mod in self.modality_data.keys():
            self.sample_masks[mod] = [s in modality_samples[mod] for s in self.sample_ids]
    
    def _prepare_tensors(self):
        """Prepare data tensors with proper missing value handling."""
        self.tensors = {}
        self.feature_dims = {}
        
        for mod, df in self.modality_data.items():
            # Reindex to align with sample_ids
            aligned_df = df.reindex(self.sample_ids)
            
            # Handle missing values
            if self.missing_value_strategy == 'zero':
                aligned_df = aligned_df.fillna(0)
            elif self.missing_value_strategy == 'mean':
                aligned_df = aligned_df.fillna(df.mean())
            elif self.missing_value_strategy == 'median':
                aligned_df = aligned_df.fillna(df.median())
            
            # Normalize if requested
            if self.normalize:
                aligned_df = (aligned_df - aligned_df.mean()) / (aligned_df.std() + 1e-8)
            
            # Convert to tensor
            self.tensors[mod] = torch.FloatTensor(aligned_df.values)
            self.feature_dims[mod] = aligned_df.shape[1]
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Get modality data
        batch = {}
        for mod in self.modality_data.keys():
            batch[mod] = self.tensors[mod][idx]
        
        # Add sample metadata
        batch['sample_id'] = sample_id
        batch['sample_masks'] = {mod: mask[idx] for mod, mask in self.sample_masks.items()}
        
        # Add labels if available
        if self.labels is not None:
            batch['label'] = torch.LongTensor([self.labels.iloc[idx]])
        
        return batch
    
    def get_modality_info(self):
        """Get information about modalities and their dimensions."""
        return {
            'modalities': list(self.modality_data.keys()),
            'feature_dims': self.feature_dims,
            'sample_alignment': self.alignment,
            'num_samples': len(self.sample_ids)
        }


class FlexibleMultiSampleDataset(Dataset):
    """
    Advanced dataset for handling complex multi-sample, multi-omics relationships.
    
    This dataset can handle scenarios where:
    - Different modalities measure different features from the same samples
    - Same modalities measure same features from different samples
    - Complex sample relationships and groupings
    """
    
    def __init__(self, data_config: Dict[str, Any], sample_relationships: Optional[pd.DataFrame] = None,
                 batch_strategy: str = 'mixed', augmentation: bool = False):
        """
        Args:
            data_config: Configuration dictionary with modality data and metadata
            sample_relationships: DataFrame defining relationships between samples
            batch_strategy: How to form batches ('mixed', 'grouped', 'matched')
            augmentation: Whether to apply data augmentation
        """
        self.data_config = data_config
        self.sample_relationships = sample_relationships
        self.batch_strategy = batch_strategy
        self.augmentation = augmentation
        
        # Parse data configuration
        self._parse_data_config()
        
        # Setup sample relationships
        self._setup_sample_relationships()
        
        # Create sample groups based on strategy
        self._create_sample_groups()
    
    def _parse_data_config(self):
        """Parse the data configuration and prepare modality data."""
        self.modality_data = {}
        self.sample_metadata = {}
        
        for mod, config in self.data_config.items():
            if isinstance(config, dict):
                # Complex configuration
                self.modality_data[mod] = config['data']
                self.sample_metadata[mod] = config.get('metadata', {})
            else:
                # Simple data
                self.modality_data[mod] = config
                self.sample_metadata[mod] = {}
    
    def _setup_sample_relationships(self):
        """Setup relationships between samples across modalities."""
        if self.sample_relationships is None:
            # Create default relationships (assume 1:1 mapping by sample ID)
            all_samples = set()
            for mod_data in self.modality_data.values():
                if hasattr(mod_data, 'index'):
                    all_samples.update(mod_data.index)
                else:
                    all_samples.update(range(len(mod_data)))
            
            # Create identity relationships
            self.sample_relationships = pd.DataFrame({
                'sample_id': list(all_samples),
                'group_id': list(all_samples),
                'modality': 'all'
            })
    
    def _create_sample_groups(self):
        """Create sample groups based on batching strategy."""
        if self.batch_strategy == 'mixed':
            # Mixed batches with samples from different groups
            self.sample_groups = list(range(len(self.sample_relationships)))
        
        elif self.batch_strategy == 'grouped':
            # Group samples by group_id
            groups = self.sample_relationships.groupby('group_id').groups
            self.sample_groups = list(groups.keys())
        
        elif self.batch_strategy == 'matched':
            # Only perfectly matched samples across all modalities
            complete_samples = self.sample_relationships.groupby('sample_id').size()
            complete_samples = complete_samples[complete_samples == len(self.modality_data)]
            self.sample_groups = list(complete_samples.index)
        
        else:
            raise ValueError(f"Unknown batch strategy: {self.batch_strategy}")
    
    def __len__(self):
        return len(self.sample_groups)
    
    def __getitem__(self, idx):
        group_id = self.sample_groups[idx]
        
        # Get samples for this group
        if self.batch_strategy == 'grouped':
            group_samples = self.sample_relationships[
                self.sample_relationships['group_id'] == group_id
            ]
        else:
            # For mixed and matched strategies
            group_samples = self.sample_relationships.iloc[[idx]]
        
        # Collect data for each modality
        batch = {}
        sample_masks = {}
        
        for mod in self.modality_data.keys():
            mod_samples = group_samples[
                (group_samples['modality'] == mod) | (group_samples['modality'] == 'all')
            ]
            
            if len(mod_samples) > 0:
                # Get the first matching sample (can be extended for multiple samples)
                sample_id = mod_samples.iloc[0]['sample_id']
                
                if hasattr(self.modality_data[mod], 'loc'):
                    # DataFrame-like data
                    batch[mod] = torch.FloatTensor(
                        self.modality_data[mod].loc[sample_id].values
                    )
                else:
                    # Array-like data
                    batch[mod] = torch.FloatTensor(self.modality_data[mod][sample_id])
                
                sample_masks[mod] = True
            else:
                # Missing modality - create placeholder
                expected_dim = self._get_modality_dim(mod)
                batch[mod] = torch.zeros(expected_dim)
                sample_masks[mod] = False
        
        # Add metadata
        batch['sample_masks'] = sample_masks
        batch['group_id'] = group_id
        
        # Apply augmentation if enabled
        if self.augmentation:
            batch = self._apply_augmentation(batch)
        
        return batch
    
    def _get_modality_dim(self, modality):
        """Get the expected dimension for a modality."""
        mod_data = self.modality_data[modality]
        if hasattr(mod_data, 'shape'):
            return mod_data.shape[1] if len(mod_data.shape) > 1 else mod_data.shape[0]
        else:
            return len(mod_data[0]) if len(mod_data) > 0 else 100  # Default
    
    def _apply_augmentation(self, batch):
        """Apply data augmentation techniques."""
        augmented_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                # Add noise
                noise = torch.randn_like(value) * 0.01
                augmented_batch[key] = value + noise
            else:
                augmented_batch[key] = value
        
        return augmented_batch


def create_synthetic_multiomics_data(n_samples: int = 1000, n_features_per_modality: Dict[str, int] = None,
                                   n_classes: int = 2, missing_rate: float = 0.1, 
                                   correlation_strength: float = 0.3) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Create synthetic multi-omics data for testing and examples.
    
    Args:
        n_samples: Number of samples
        n_features_per_modality: Dict mapping modality names to feature counts
        n_classes: Number of classes
        missing_rate: Rate of missing samples per modality
        correlation_strength: Strength of correlation between modalities
    
    Returns:
        Tuple of (modality_data_dict, labels)
    """
    if n_features_per_modality is None:
        n_features_per_modality = {
            'genomics': 1000,
            'transcriptomics': 500,
            'proteomics': 200,
            'metabolomics': 150
        }
    
    # Generate sample IDs
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    # Generate labels
    labels = pd.Series(
        np.random.randint(0, n_classes, n_samples),
        index=sample_ids,
        name='label'
    )
    
    # Generate correlated base signals
    base_signal = np.random.randn(n_samples, 50)  # Shared latent factors
    
    modality_data = {}
    
    for mod, n_features in n_features_per_modality.items():
        # Create modality-specific features with some correlation to base signal
        mod_specific = np.random.randn(n_samples, n_features - 10)
        
        # Add some features correlated with the base signal
        correlated_features = base_signal[:, :10] * correlation_strength + \
                            np.random.randn(n_samples, 10) * (1 - correlation_strength)
        
        # Combine features
        features = np.hstack([correlated_features, mod_specific])
        
        # Add some class-dependent signal
        for class_idx in range(n_classes):
            class_mask = labels == class_idx
            if class_mask.sum() > 0:  # Only if there are samples in this class
                class_signal = np.random.randn(n_features) * 0.5
                features[class_mask] += class_signal
        
        # Create DataFrame
        feature_names = [f"{mod}_feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(features, index=sample_ids, columns=feature_names)
        
        # Introduce missing samples
        if missing_rate > 0:
            n_missing = int(n_samples * missing_rate)
            missing_samples = np.random.choice(sample_ids, n_missing, replace=False)
            df = df.drop(missing_samples)
        
        modality_data[mod] = df
    
    return modality_data, labels


# Utility functions for data processing
def normalize_modality_data(data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """Normalize modality data using specified method."""
    if method == 'zscore':
        return (data - data.mean()) / (data.std() + 1e-8)
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    elif method == 'robust':
        median = data.median()
        mad = (data - median).abs().median()
        return (data - median) / (mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_features_by_variance(data: pd.DataFrame, min_variance: float = 0.01) -> pd.DataFrame:
    """Filter features by minimum variance threshold."""
    variances = data.var()
    selected_features = variances[variances >= min_variance].index
    return data[selected_features]


def align_modalities_by_samples(modality_data: Dict[str, pd.DataFrame], 
                               strategy: str = 'intersection') -> Dict[str, pd.DataFrame]:
    """Align multiple modalities by common samples."""
    if strategy == 'intersection':
        # Find common samples
        common_samples = set(modality_data[list(modality_data.keys())[0]].index)
        for df in modality_data.values():
            common_samples &= set(df.index)
        
        # Align all modalities to common samples
        aligned_data = {}
        for mod, df in modality_data.items():
            aligned_data[mod] = df.loc[list(common_samples)].sort_index()
        
        return aligned_data
    
    elif strategy == 'union':
        # Find all samples
        all_samples = set()
        for df in modality_data.values():
            all_samples |= set(df.index)
        
        # Reindex all modalities to include all samples
        aligned_data = {}
        for mod, df in modality_data.items():
            aligned_data[mod] = df.reindex(list(all_samples)).sort_index()
        
        return aligned_data
    
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")