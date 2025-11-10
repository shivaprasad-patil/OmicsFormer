#!/usr/bin/env python3
"""
Multi-Study Transcriptomics Integration Demo

This demonstrates how OmicsFormer can integrate multiple transcriptomics datasets
from different studies measuring the same genes for the same disease/process.

Scenario: 5 different studies of the same cancer type, all measuring gene expression
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer
from omicsformer.training.trainer import MultiOmicsTrainer

def create_multi_study_data():
    """
    Create synthetic data simulating 5 different transcriptomics studies
    of the same disease (e.g., breast cancer) with batch effects.
    """
    print("🔬 Simulating 5 Transcriptomics Studies of Breast Cancer")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Common gene set (same genes measured across all studies)
    n_genes = 1000
    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
    
    # Study characteristics
    studies = {
        'Study1_USA': {'n_samples': 100, 'batch_effect': 0.0, 'tech': 'microarray'},
        'Study2_Europe': {'n_samples': 80, 'batch_effect': 0.5, 'tech': 'RNA-seq'},
        'Study3_Asia': {'n_samples': 120, 'batch_effect': 0.3, 'tech': 'RNA-seq'},
        'Study4_USA': {'n_samples': 90, 'batch_effect': 0.2, 'tech': 'microarray'},
        'Study5_Multi': {'n_samples': 110, 'batch_effect': 0.4, 'tech': 'RNA-seq'}
    }
    
    print(f"📊 Dataset Characteristics:")
    print(f"   • Common genes measured: {n_genes}")
    print(f"   • Number of studies: {len(studies)}")
    print(f"   • Total samples: {sum(s['n_samples'] for s in studies.values())}")
    print()
    
    # Generate data for each study with batch effects
    all_data = {}
    all_labels = []
    
    for study_name, study_info in studies.items():
        n_samples = study_info['n_samples']
        batch_effect = study_info['batch_effect']
        
        # Generate base gene expression with biological signal
        # 3 subtypes of breast cancer: Luminal (0), HER2+ (1), Triple-negative (2)
        labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        
        # Generate expression data with biological signal
        data = np.random.randn(n_samples, n_genes)
        
        # Add subtype-specific signals
        for i, label in enumerate(labels):
            if label == 0:  # Luminal
                data[i, :100] += 2.0  # ER/PR signature
            elif label == 1:  # HER2+
                data[i, 100:200] += 2.5  # HER2 signature
            else:  # Triple-negative
                data[i, 200:300] += 1.8  # Basal signature
        
        # Add batch effect (technical variation)
        data += np.random.randn(n_samples, n_genes) * batch_effect
        
        # Add study-specific mean shift (center bias)
        data += np.random.randn(1, n_genes) * batch_effect * 0.5
        
        # Create DataFrame
        sample_ids = [f'{study_name}_Sample_{i:03d}' for i in range(n_samples)]
        df = pd.DataFrame(data, index=sample_ids, columns=gene_names)
        
        all_data[study_name] = df
        all_labels.extend(labels)
        
        print(f"   📋 {study_name}:")
        print(f"      Samples: {n_samples}, Technology: {study_info['tech']}")
        print(f"      Batch effect: {batch_effect:.1f}, Label distribution: {np.bincount(labels)}")
    
    all_labels = pd.Series(all_labels, name='subtype')
    
    return all_data, all_labels, gene_names


def approach1_separate_modalities():
    """
    Approach 1: Treat each study as a separate modality
    
    Pros: 
    - Study-specific patterns can be learned
    - Batch effects are isolated per study
    - Can learn cross-study patterns via attention
    
    Cons:
    - Requires more parameters (one projection per study)
    """
    print("\n" + "="*80)
    print("🎯 APPROACH 1: Each Study as Separate Modality")
    print("="*80)
    
    data_dict, labels, gene_names = create_multi_study_data()
    
    print("\n📊 Dataset Configuration:")
    print(f"   • Treating {len(data_dict)} studies as {len(data_dict)} modalities")
    print(f"   • Each modality: {data_dict[list(data_dict.keys())[0]].shape[1]} genes")
    
    # Create dataset
    dataset = FlexibleMultiOmicsDataset(
        modality_data=data_dict,
        labels=labels,
        alignment='union'  # Combine all samples
    )
    
    print(f"\n✅ Dataset created:")
    print(f"   • Total samples: {len(dataset)}")
    print(f"   • Modalities: {len(dataset.modality_data)}")
    print(f"   • Strategy: {dataset.alignment}")
    
    # Create model
    input_dims = {mod: data_dict[mod].shape[1] for mod in data_dict.keys()}
    
    model = EnhancedMultiOmicsTransformer(
        input_dims=input_dims,
        num_classes=3,  # 3 cancer subtypes
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"\n🤖 Model created:")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Input modalities: {list(input_dims.keys())}")
    
    # Test forward pass
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    
    print(f"   • Output shape: {logits.shape}")
    print(f"   ✅ Forward pass successful!")
    
    return dataset, model


def approach2_combined_with_batch_info():
    """
    Approach 2: Combine studies into single modality with batch/study encoding
    
    Pros:
    - Shared parameters across studies
    - More efficient
    - Easier batch correction
    
    Cons:
    - Doesn't leverage cross-study attention patterns as explicitly
    """
    print("\n" + "="*80)
    print("🎯 APPROACH 2: Combined Transcriptomics with Batch Encoding")
    print("="*80)
    
    data_dict, labels, gene_names = create_multi_study_data()
    
    # Combine all studies into one transcriptomics modality
    combined_data = pd.concat(list(data_dict.values()), axis=0)
    
    # Create batch indicators
    batch_info = []
    for study_idx, (study_name, study_df) in enumerate(data_dict.items()):
        batch_info.extend([study_idx] * len(study_df))
    
    batch_series = pd.Series(batch_info, index=combined_data.index, name='batch')
    
    # Add batch information as additional features
    batch_onehot = pd.get_dummies(batch_series, prefix='batch')
    combined_with_batch = pd.concat([combined_data, batch_onehot], axis=1)
    
    print(f"\n📊 Combined Dataset:")
    print(f"   • Total samples: {len(combined_with_batch)}")
    print(f"   • Gene features: {len(gene_names)}")
    print(f"   • Batch features: {len(batch_onehot.columns)}")
    print(f"   • Total features: {combined_with_batch.shape[1]}")
    
    # Create dataset with single modality
    dataset = FlexibleMultiOmicsDataset(
        modality_data={'transcriptomics': combined_with_batch},
        labels=labels,
        alignment='strict'
    )
    
    print(f"\n✅ Dataset created:")
    print(f"   • Samples: {len(dataset)}")
    print(f"   • Modality: transcriptomics (with batch encoding)")
    
    # Create model
    model = EnhancedMultiOmicsTransformer(
        input_dims={'transcriptomics': combined_with_batch.shape[1]},
        num_classes=3,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"\n🤖 Model created:")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✅ More parameter-efficient than Approach 1")
    
    # Test forward pass
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    
    print(f"   • Output shape: {logits.shape}")
    print(f"   ✅ Forward pass successful!")
    
    return dataset, model


def approach3_hybrid():
    """
    Approach 3: Hybrid - Studies as modalities + shared gene embeddings
    
    This is the most sophisticated approach combining benefits of both.
    """
    print("\n" + "="*80)
    print("🎯 APPROACH 3: Hybrid Approach (Advanced)")
    print("="*80)
    print("   Combining study-specific learning with shared biological knowledge")
    print("   • Each study is a modality for study-specific patterns")
    print("   • But shares underlying gene embeddings")
    print("   • Can learn both batch effects and biological signals")
    print("   ⚠️  This would require custom model architecture (future work)")
    print()


def main():
    """Run all demonstrations."""
    print("🧬" + "="*78 + "🧬")
    print("      MULTI-STUDY TRANSCRIPTOMICS INTEGRATION WITH OMICSFORMER")
    print("🧬" + "="*78 + "🧬")
    print()
    print("📋 Use Case: Integrating 5 different transcriptomics studies")
    print("   • Same disease (e.g., Breast Cancer)")
    print("   • Same genes measured")
    print("   • Different batches, technologies, populations")
    print()
    
    # Approach 1
    dataset1, model1 = approach1_separate_modalities()
    
    # Approach 2
    dataset2, model2 = approach2_combined_with_batch_info()
    
    # Approach 3 (conceptual)
    approach3_hybrid()
    
    # Comparison
    print("="*80)
    print("📊 COMPARISON OF APPROACHES")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Approach': [
            'Separate Modalities',
            'Combined + Batch',
            'Hybrid (Future)'
        ],
        'Parameters': [
            f"{sum(p.numel() for p in model1.parameters()):,}",
            f"{sum(p.numel() for p in model2.parameters()):,}",
            'TBD'
        ],
        'Cross-Study Learning': ['✅ Explicit', '⚖️ Implicit', '✅✅ Best of both'],
        'Batch Handling': ['⚖️ Isolated', '✅ Explicit', '✅✅ Sophisticated'],
        'Efficiency': ['⚖️ Lower', '✅ Higher', '✅ Balanced'],
        'Use Case': [
            'Strong batch effects',
            'Mild batch effects',
            'Complex scenarios'
        ]
    })
    
    print()
    print(comparison.to_string(index=False))
    print()
    
    print("="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    print()
    print("✅ For 5 studies with different technologies/populations:")
    print("   → Use APPROACH 1 (Separate Modalities)")
    print("   → Treats each study as distinct modality")
    print("   → Transformer learns cross-study patterns via attention")
    print("   → Naturally handles batch effects")
    print()
    print("✅ For 5 studies with similar protocols:")
    print("   → Use APPROACH 2 (Combined + Batch Encoding)")
    print("   → More parameter efficient")
    print("   → Explicit batch correction via features")
    print()
    print("✅ Best Practice:")
    print("   → Try both approaches and compare")
    print("   → Use cross-validation to assess generalization")
    print("   → Check if model learns study-agnostic features")
    print()
    
    print("="*80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*80)
    print()
    print("Key Insight: OmicsFormer's flexible architecture allows multiple")
    print("strategies for multi-study integration, from treating studies as")
    print("separate modalities to sophisticated batch encoding.")
    print()
    print("✅ YES - OmicsFormer can integrate 5 different transcriptomics datasets!")
    print()


if __name__ == '__main__':
    main()
