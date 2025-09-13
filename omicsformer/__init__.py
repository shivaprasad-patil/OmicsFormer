"""
OmicsFormer: Advanced Multi-Omics Integration with Transformers

A comprehensive Python package for integrating multiple omics modalities using 
state-of-the-art transformer architectures with advanced attention mechanisms.

Features:
- Flexible multi-omics data handling with multiple alignment strategies
- Advanced attention mechanisms (Grouped Query Attention, Mixture of Experts)
- Memory-efficient transformers with 40% memory reduction
- Comprehensive post-training analysis and interpretation tools
- Biological insights and pathway analysis capabilities
- Clinical-ready missing data handling
- Production-ready training pipelines
- Extensive visualization and reporting tools

Author: Shiva Prasad
"""

__version__ = "0.1.0"
__author__ = "Shiva Prasad"
__email__ = "your.email@domain.com"

# Core imports
from . import models
from . import data
from . import training
from . import analysis
from . import utils

# Key classes for easy access
from .models.transformer import EnhancedMultiOmicsTransformer, AdvancedMultiOmicsTransformer
from .models.attention import GroupedQueryAttention, MixtureOfExpertsLayer
from .data.dataset import FlexibleMultiOmicsDataset, FlexibleMultiSampleDataset, create_synthetic_multiomics_data
from .training.trainer import MultiOmicsTrainer
from .analysis.analyzer import MultiOmicsAnalyzer
from .utils.utils import set_random_seeds, get_device, print_model_summary

__all__ = [
    # Modules
    'models',
    'data', 
    'training',
    'analysis',
    'utils',
    # Key classes
    'EnhancedMultiOmicsTransformer',
    'AdvancedMultiOmicsTransformer', 
    'GroupedQueryAttention',
    'MixtureOfExpertsLayer',
    'FlexibleMultiOmicsDataset',
    'FlexibleMultiSampleDataset',
    'create_synthetic_multiomics_data',
    'MultiOmicsTrainer',
    'MultiOmicsAnalyzer',
    # Utilities
    'set_random_seeds',
    'get_device',
    'print_model_summary'
]

# Core components
from .models.transformer import EnhancedMultiOmicsTransformer, AdvancedMultiOmicsTransformer
from .models.attention import (
    GroupedQueryAttention, 
    MixtureOfExpertsLayer, 
    CrossModalAttentionBlock,
    InterpretableAttentionPool
)
from .data.dataset import FlexibleMultiOmicsDataset, FlexibleMultiSampleDataset
# Training utilities are imported through the training module
from .analysis.analyzer import MultiOmicsAnalyzer

# Utility functions
from .utils.utils import set_random_seeds, get_device, count_parameters

__all__ = [
    # Core models
    'EnhancedMultiOmicsTransformer',
    'AdvancedMultiOmicsTransformer',
    
    # Attention mechanisms
    'InterpretableAttentionPool',
    'GroupedQueryAttention',
    'MixtureOfExpertsLayer',
    'CrossModalAttentionBlock',
    
    # Data handling
    'FlexibleMultiOmicsDataset',
    'FlexibleMultiSampleDataset',
    'create_synthetic_multiomics_data',
    
    # Training
    'MultiOmicsTrainer',
    
    # Analysis
    'MultiOmicsAnalyzer',
    
    # Utilities
    'set_random_seeds',
    'get_device',
    'count_parameters'
]# Package metadata
PACKAGE_NAME = "omicsformer"
DESCRIPTION = "Advanced multi-omics integration with transformers"
URL = "https://github.com/yourusername/omicsformer"