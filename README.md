# OmicsFormer: Multi-Omics Integration with Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Deep learning framework for integrating multi-omics data using transformer architectures. Handles multiple modalities, missing data, and batch effects for disease classification and biomarker discovery.

## Key Features

- **Multi-Modal Integration**: Genomics, transcriptomics, proteomics, metabolomics
- **Flexible Alignment**: 4 strategies for handling missing modalities
- **Batch Effect Correction**: Integrate studies from different platforms/technologies
- **Feature Importance**: Gradient, attention, and permutation methods (scaled 0-1)
- **Advanced Architecture**: Grouped Query Attention (GQA) + Mixture of Experts (MoE)

## Quick Start

```bash
pip install torch pandas scikit-learn matplotlib seaborn
git clone https://github.com/shivaprasad-patil/omicsformer.git
cd omicsformer && pip install -e .
```

**Basic Usage:**
```python
from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer
from omicsformer.training.trainer import MultiOmicsTrainer

# Create dataset
dataset = FlexibleMultiOmicsDataset(
    modality_data={'genomics': genomics_df, 'transcriptomics': rna_df},
    labels=labels,
    alignment='flexible'  # handles missing modalities
)

# Build and train model
model = EnhancedMultiOmicsTransformer(
    input_dims=dataset.feature_dims, num_classes=2
)
trainer = MultiOmicsTrainer(model, train_loader, val_loader)
history = trainer.fit(num_epochs=20)
```

## Real-World Example: SLE Multi-Study Analysis

**[Complete Notebook](https://github.com/shivaprasad-patil/omicsformer/blob/main/examples/SLE_MultiStudy_OmicsFormer_Analysis.ipynb)** - Integrating 8 SLE RNA-seq studies (550 samples)

**Workflow:**
```python
# 1. Load 8 independent studies
study_ids = ['SRP062966', 'SRP095109', 'SRP131775', ...]  # 8 studies total

# 2. Batch correction with ComBat
from combat.pycombat import pycombat
expr_corrected = pycombat(expr_matrix.T, batch_labels=study_indices)

# 3. Train model
model = EnhancedMultiOmicsTransformer(
    input_dims={'combined': 2008},  # 2000 genes + 8 study indicators
    num_classes=2
)
```

**Results:**
- **90.91% test accuracy** (F1=0.91)
- **Batch effects removed**: PC1 variance 64.9% → 24.8%
- **Top biomarkers identified**: TNFSF13B, TPM2, PNLIPRP3, SLIT2, COL1A2

See [full notebook](https://github.com/shivaprasad-patil/omicsformer/blob/main/examples/SLE_MultiStudy_OmicsFormer_Analysis.ipynb) for complete analysis with visualizations.

## Feature Importance & Analysis

```python
from omicsformer.analysis.analyzer import MultiOmicsAnalyzer

analyzer = MultiOmicsAnalyzer(model, device='cpu')

# Compute importance (automatically scaled to [0, 1])
importance = analyzer.analyze_feature_importance(
    dataloader=train_loader,
    method='gradient'  # or 'attention', 'permutation'
)

# Visualize top features
analyzer.plot_feature_importance(importance, modality='genomics', top_n=20)
```

**Three methods available:**
- **Gradient-based**: Backpropagation sensitivity
- **Attention-based**: Model focus patterns  
- **Permutation-based**: Performance impact

All scores scaled to [0, 1] for easy comparison.

## Alignment Strategies

| Strategy | Use Case | Missing Data |
|----------|----------|--------------|
| **Strict** | All modalities required | ❌ Not allowed |
| **Flexible** | Research/exploratory | ✅ Zero-filled |
| **Intersection** | Balanced coverage | ✅ Partial OK |
| **Union** | Maximum data use | ✅ All handled |

Use `alignment='auto'` for automatic selection.

## Examples

**Real-World:**
- [SLE Multi-Study Analysis](https://github.com/shivaprasad-patil/omicsformer/blob/main/examples/SLE_MultiStudy_OmicsFormer_Analysis.ipynb) - 8 studies, 550 samples, 90.91% accuracy

**Quick Start:**
```bash
python examples/quick_reference.py                  # Basic usage
python examples/analyzer_comprehensive_test.py      # Full analysis pipeline
python examples/alignment_strategies_demo.py        # Alignment comparison
```

## Applications

- Cancer subtype classification
- Autoimmune disease biomarker discovery (SLE, RA, IBD)
- Drug response prediction
- Multi-center clinical trial integration
- Cross-platform data harmonization

## License & Contact

Apache License 2.0 - See LICENSE file

**Issues**: [github.com/shivaprasad-patil/omicsformer/issues](https://github.com/shivaprasad-patil/omicsformer/issues)  
**Email**: shivaprasad309319@gmail.com

## Citation

```bibtex
@software{omicsformer2025,
  title={OmicsFormer: Multi-Omics Integration with Transformers},
  author={Shivaprasad Patil},
  year={2025},
  url={https://github.com/shivaprasad-patil/omicsformer}
}
```