# Multi-Study Transcriptomics Integration Examples

This directory contains comprehensive examples demonstrating how to integrate multiple transcriptomics datasets from different studies using OmicsFormer.

## Quick Start

For a fast demonstration, run:
```bash
python quick_multi_study_demo.py
```

This will train models on 7 simulated lung cancer studies (500 total samples) in ~30 seconds.

## Available Examples

### 1. `quick_multi_study_demo.py` ⚡ (Recommended for Quick Demo)
**Runtime:** ~30 seconds  
**Purpose:** Fast demonstration of multi-study integration

**Features:**
- 7 lung cancer studies (TCGA_USA, GEO_Europe, Japan, UK, China, Canada, Australia)
- 500 total samples across studies
- 200 genes measured in common
- 3 cancer subtypes (LUAD, LUSC, SCLC)
- Different technologies (RNA-seq vs microarray)
- Varying batch effects (0.1 to 0.8)
- **Two integration approaches:**
  - Approach 1: Each study as separate modality (688K parameters)
  - Approach 2: Combined with batch encoding (585K parameters)

**Output:**
```
✅ SUCCESS! OmicsFormer can integrate multiple transcriptomics studies!

COMPARISON:
  Metric               Separate Modalities  Combined + Batch    
  -------------------- -------------------- --------------------
  Test Accuracy                     77.33%             100.00%
  Test F1                           0.6801              1.0000
  Parameters                       688,612             584,996
```

### 2. `multi_study_transcriptomics_integration.py` 🔬 (Full-Featured)
**Runtime:** ~5-10 minutes  
**Purpose:** Comprehensive multi-study integration with full training and visualization

**Features:**
- 7 lung cancer studies with realistic configurations
- 670 total samples
- 500 genes
- Full 20-epoch training with early stopping
- Cross-study generalization analysis
- **Visualizations generated:**
  - `lung_cancer_batch_effects.png` - PCA showing batch effects, biological signal, and technology differences
  - `lung_cancer_training_separate_modalities.png` - Training curves for Approach 1
  - `lung_cancer_training_combined_with_batch.png` - Training curves for Approach 2

**What it demonstrates:**
- Per-study performance evaluation
- Batch effect visualization
- Training history tracking
- Model comparison

### 3. `multi_study_integration_demo.py` 📊 (Original Conceptual Demo)
**Purpose:** Architectural comparison and conceptual demonstration

**Features:**
- Shows two main approaches
- Includes theoretical hybrid approach
- Parameter count comparisons
- Use case recommendations

## Integration Approaches

### Approach 1: Each Study as Separate Modality
```python
dataset = FlexibleMultiOmicsDataset(
    modality_data={
        'TCGA_USA': study1_df,
        'GEO_Europe': study2_df,
        'Japan_Cohort': study3_df,
        # ... more studies
    },
    labels=labels,
    alignment='union',
    normalize=True
)
```

**Advantages:**
- Explicit cross-study attention learning
- Batch effects isolated per study
- Can learn study-specific patterns
- Good for strong batch effects

**Use when:**
- Studies have very different technologies
- Strong batch effects expected
- Need to identify study-specific biomarkers
- Sample sizes vary significantly across studies

### Approach 2: Combined with Batch Encoding
```python
# Combine all study data
combined_data = pd.concat([study1_df, study2_df, study3_df, ...])

# Add batch indicators
batch_onehot = pd.get_dummies(study_labels, prefix='batch')
combined_with_batch = pd.concat([combined_data, batch_onehot], axis=1)

dataset = FlexibleMultiOmicsDataset(
    modality_data={'transcriptomics': combined_with_batch},
    labels=labels,
    alignment='strict',
    normalize=True
)
```

**Advantages:**
- 15-30% fewer parameters
- Faster training
- Explicit batch correction
- Better for homogeneous studies

**Use when:**
- Studies use similar technologies
- Batch effects are mild to moderate
- Computational resources are limited
- Need faster inference

## Real-World Usage Example

```python
import pandas as pd
from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer
from omicsformer.training.trainer import MultiOmicsTrainer

# Load your studies
tcga_data = pd.read_csv('tcga_expression.csv', index_col=0)
geo_data = pd.read_csv('geo_expression.csv', index_col=0)
your_data = pd.read_csv('your_expression.csv', index_col=0)

# Combine labels
labels = pd.concat([
    pd.read_csv('tcga_labels.csv', index_col=0)['subtype'],
    pd.read_csv('geo_labels.csv', index_col=0)['subtype'],
    pd.read_csv('your_labels.csv', index_col=0)['subtype']
])

# Create dataset (Approach 1)
dataset = FlexibleMultiOmicsDataset(
    modality_data={
        'TCGA': tcga_data,
        'GEO': geo_data,
        'YourStudy': your_data
    },
    labels=labels,
    alignment='union',
    normalize=True
)

# Train model
model = EnhancedMultiOmicsTransformer(
    input_dims={'TCGA': tcga_data.shape[1], 
                'GEO': geo_data.shape[1],
                'YourStudy': your_data.shape[1]},
    num_classes=len(labels.unique()),
    embed_dim=128,
    num_heads=8,
    num_layers=4
)

trainer = MultiOmicsTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

history = trainer.fit(num_epochs=20)
```

## Key Insights

### Batch Effect Handling
- OmicsFormer can handle batch effects through:
  1. **Separate modalities**: Isolates each study, learns cross-study patterns
  2. **Batch encoding**: Explicitly models batch as a feature
  3. **Attention mechanism**: Learns to focus on relevant patterns across batches

### When to Use Multi-Study Integration
✅ **Good use cases:**
- Meta-analysis across multiple published datasets
- Combining in-house data with public datasets
- Multi-center clinical trials
- Longitudinal studies with platform changes
- Validation across independent cohorts

⚠️ **Considerations:**
- Ensure common gene/feature set across studies
- Document technology differences
- Consider study-specific preprocessing
- Validate on held-out studies when possible

## Performance Expectations

Based on simulated data:
- **Training time:** 30 seconds (quick demo) to 10 minutes (full training)
- **Accuracy:** 70-100% depending on signal strength and batch effects
- **F1 Score:** 0.68-1.0 for multi-class classification
- **Cross-study generalization:** Varies by batch effect magnitude

## Troubleshooting

### Issue: Low accuracy with separate modalities
**Solution:** Try Approach 2 (combined with batch encoding) for more direct learning

### Issue: Perfect training accuracy but poor validation
**Solution:** Reduce model complexity, increase dropout, or use more data augmentation

### Issue: Different studies have different gene sets
**Solution:** 
- Use `alignment='union'` to include all genes
- Missing values handled automatically
- Consider feature selection to common genes

### Issue: Memory errors with many studies
**Solution:**
- Reduce `embed_dim` and `num_layers`
- Use gradient checkpointing (not yet implemented)
- Try Approach 2 (more parameter efficient)

## Citation

If you use OmicsFormer for multi-study integration, please cite:
```
[Citation information to be added]
```

## Additional Resources

- Main OmicsFormer documentation: `../README.md`
- Alignment strategies: `alignment_strategies_demo.py`
- Comprehensive analysis: `analyzer_comprehensive_test.py`
- Quick reference: `quick_reference.py`
