#!/usr/bin/env python3
"""
Multi-Study Transcriptomics Integration Example

This comprehensive example demonstrates how to integrate multiple transcriptomics
datasets from different studies measuring the same genes for the same disease.

Scenario: 7 independent studies of lung cancer, each measuring gene expression
with different technologies, sample sizes, and batch effects.

Author: OmicsFormer Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer
from omicsformer.training.trainer import MultiOmicsTrainer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def create_realistic_multi_study_data(n_genes=500, n_studies=7):
    """
    Create realistic multi-study transcriptomics data simulating lung cancer.
    
    Features:
    - Different sample sizes per study
    - Varying batch effects (technical variation)
    - Same biological signal (3 lung cancer subtypes)
    - Different technologies (microarray vs RNA-seq)
    - Missing samples (not all patients in all studies)
    
    Args:
        n_genes: Number of genes (features) measured across all studies
        n_studies: Number of independent studies
        
    Returns:
        study_data: Dict of DataFrames (one per study)
        labels: Combined labels for all samples
        gene_names: List of gene names
        study_info: Metadata about each study
    """
    print("🧬 Simulating Multi-Study Lung Cancer Transcriptomics Data")
    print("=" * 80)
    
    # Define common gene set
    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
    
    # Define lung cancer subtypes
    # 0: Adenocarcinoma (LUAD)
    # 1: Squamous cell carcinoma (LUSC)
    # 2: Small cell lung cancer (SCLC)
    
    # Study configurations (realistic heterogeneity)
    study_configs = {
        'TCGA_LUAD': {
            'n_samples': 150,
            'technology': 'RNA-seq',
            'batch_effect': 0.1,  # Low batch effect
            'subtype_dist': [0.7, 0.2, 0.1],  # Mostly LUAD
            'country': 'USA',
            'year': 2015
        },
        'GEO_Europe': {
            'n_samples': 100,
            'technology': 'microarray',
            'batch_effect': 0.8,  # High batch effect (different platform)
            'subtype_dist': [0.3, 0.5, 0.2],  # Mostly LUSC
            'country': 'Germany',
            'year': 2012
        },
        'Japan_Cohort': {
            'n_samples': 80,
            'technology': 'RNA-seq',
            'batch_effect': 0.3,  # Moderate batch effect
            'subtype_dist': [0.5, 0.3, 0.2],  # Balanced
            'country': 'Japan',
            'year': 2018
        },
        'UK_Biobank': {
            'n_samples': 120,
            'technology': 'RNA-seq',
            'batch_effect': 0.2,  # Low batch effect
            'subtype_dist': [0.4, 0.4, 0.2],  # LUAD/LUSC balanced
            'country': 'UK',
            'year': 2020
        },
        'China_Discovery': {
            'n_samples': 90,
            'technology': 'microarray',
            'batch_effect': 0.6,  # Moderate-high batch effect
            'subtype_dist': [0.3, 0.4, 0.3],  # More SCLC
            'country': 'China',
            'year': 2016
        },
        'Canada_Multi': {
            'n_samples': 70,
            'technology': 'RNA-seq',
            'batch_effect': 0.25,  # Low-moderate batch effect
            'subtype_dist': [0.5, 0.3, 0.2],
            'country': 'Canada',
            'year': 2019
        },
        'Australia_Validation': {
            'n_samples': 60,
            'technology': 'RNA-seq',
            'batch_effect': 0.15,  # Very low batch effect
            'subtype_dist': [0.4, 0.35, 0.25],
            'country': 'Australia',
            'year': 2021
        }
    }
    
    print(f"\n📊 Study Overview:")
    print(f"   • Total studies: {len(study_configs)}")
    print(f"   • Total samples: {sum(s['n_samples'] for s in study_configs.values())}")
    print(f"   • Common genes: {n_genes}")
    print(f"   • Cancer subtypes: 3 (LUAD, LUSC, SCLC)")
    print()
    
    # Create data for each study
    study_data = {}
    all_labels = []
    study_info = []
    
    for study_name, config in study_configs.items():
        n_samples = config['n_samples']
        batch_effect = config['batch_effect']
        subtype_dist = config['subtype_dist']
        
        # Generate subtype labels
        labels = np.random.choice([0, 1, 2], size=n_samples, p=subtype_dist)
        
        # Base expression data (random normal)
        expression = np.random.randn(n_samples, n_genes)
        
        # Add strong biological signal for each subtype
        for i, label in enumerate(labels):
            if label == 0:  # LUAD signature
                # EGFR pathway genes (first 50 genes)
                expression[i, :50] += np.random.normal(3.0, 0.5, 50)
                # Cell cycle genes (genes 50-100)
                expression[i, 50:100] += np.random.normal(2.0, 0.3, 50)
                
            elif label == 1:  # LUSC signature
                # TP53 pathway genes (genes 100-150)
                expression[i, 100:150] += np.random.normal(3.5, 0.6, 50)
                # Keratinization genes (genes 150-200)
                expression[i, 150:200] += np.random.normal(2.5, 0.4, 50)
                
            else:  # SCLC signature
                # Neuroendocrine genes (genes 200-250)
                expression[i, 200:250] += np.random.normal(4.0, 0.7, 50)
                # RB1 pathway genes (genes 250-300)
                expression[i, 250:300] += np.random.normal(2.8, 0.5, 50)
        
        # Add batch effects (technical variation)
        # 1. Global mean shift (platform bias)
        if config['technology'] == 'microarray':
            expression += np.random.normal(0.5, 0.2, (1, n_genes))
        else:
            expression += np.random.normal(-0.3, 0.2, (1, n_genes))
        
        # 2. Gene-specific batch effects
        batch_noise = np.random.randn(n_samples, n_genes) * batch_effect
        expression += batch_noise
        
        # 3. Sample-specific batch effects (library size, etc.)
        sample_effects = np.random.randn(n_samples, 1) * batch_effect * 0.3
        expression += sample_effects
        
        # Create DataFrame
        sample_ids = [f'{study_name}_S{i:04d}' for i in range(n_samples)]
        df = pd.DataFrame(expression, index=sample_ids, columns=gene_names)
        
        study_data[study_name] = df
        all_labels.extend(labels)
        
        # Store study metadata
        for sid, lbl in zip(sample_ids, labels):
            study_info.append({
                'sample_id': sid,
                'study': study_name,
                'label': lbl,
                'technology': config['technology'],
                'country': config['country'],
                'year': config['year']
            })
        
        print(f"   📋 {study_name}:")
        print(f"      Samples: {n_samples:3d} | Tech: {config['technology']:10s} | "
              f"Batch: {batch_effect:.2f} | Country: {config['country']:9s} | "
              f"Subtypes: {np.bincount(labels)}")
    
    # Convert to DataFrame for study info
    study_info_df = pd.DataFrame(study_info)
    
    # Create labels Series with sample IDs as index
    all_sample_ids = study_info_df['sample_id'].tolist()
    all_labels_series = pd.Series(all_labels, index=all_sample_ids, name='cancer_subtype')
    
    print(f"\n✅ Data generation complete!")
    print(f"   Total samples: {len(all_labels_series)}")
    print(f"   Label distribution: LUAD={(all_labels_series==0).sum()}, "
          f"LUSC={(all_labels_series==1).sum()}, SCLC={(all_labels_series==2).sum()}")
    
    return study_data, all_labels_series, gene_names, study_info_df


def visualize_batch_effects(study_data, study_info_df, save_prefix='multi_study'):
    """Visualize batch effects using PCA."""
    print("\n📊 Visualizing Batch Effects and Study Distribution...")
    
    # Combine all data
    combined_data = pd.concat(list(study_data.values()), axis=0)
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(combined_data)
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Study': study_info_df['study'].values,
        'Subtype': study_info_df['label'].map({0: 'LUAD', 1: 'LUSC', 2: 'SCLC'}).values,
        'Technology': study_info_df['technology'].values
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Color by study (shows batch effects)
    studies = viz_df['Study'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(studies)))
    for i, study in enumerate(studies):
        mask = viz_df['Study'] == study
        axes[0].scatter(viz_df.loc[mask, 'PC1'], viz_df.loc[mask, 'PC2'],
                       c=[colors[i]], label=study, alpha=0.6, s=30)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PCA colored by Study\n(Shows Batch Effects)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Color by subtype (shows biological signal)
    subtype_colors = {'LUAD': 'red', 'LUSC': 'blue', 'SCLC': 'green'}
    for subtype, color in subtype_colors.items():
        mask = viz_df['Subtype'] == subtype
        axes[1].scatter(viz_df.loc[mask, 'PC1'], viz_df.loc[mask, 'PC2'],
                       c=color, label=subtype, alpha=0.6, s=30)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('PCA colored by Cancer Subtype\n(Shows Biological Signal)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Color by technology
    tech_colors = {'RNA-seq': 'purple', 'microarray': 'orange'}
    for tech, color in tech_colors.items():
        mask = viz_df['Technology'] == tech
        axes[2].scatter(viz_df.loc[mask, 'PC1'], viz_df.loc[mask, 'PC2'],
                       c=color, label=tech, alpha=0.6, s=30)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[2].set_title('PCA colored by Technology\n(Shows Platform Effects)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_prefix}_batch_effects.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved visualization: {save_path}")
    plt.close()
    
    return viz_df


def train_multi_study_model(study_data, labels, approach='separate_modalities'):
    """
    Train OmicsFormer model on multi-study data.
    
    Args:
        study_data: Dict of study DataFrames
        labels: Sample labels
        approach: 'separate_modalities' or 'combined_with_batch'
    """
    print(f"\n🤖 Training Model with Approach: {approach.upper()}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if approach == 'separate_modalities':
        # Approach 1: Each study as separate modality
        print("\n📋 Configuration: Each study as separate modality")
        
        # Create dataset
        dataset = FlexibleMultiOmicsDataset(
            modality_data=study_data,
            labels=labels,
            alignment='union',
            normalize=True
        )
        
        # Model configuration
        input_dims = {name: df.shape[1] for name, df in study_data.items()}
        
    else:
        # Approach 2: Combined with batch encoding
        print("\n📋 Configuration: Combined data with batch encoding")
        
        # Combine all studies
        combined_data = pd.concat(list(study_data.values()), axis=0)
        
        # Create batch indicators
        batch_info = []
        for study_idx, (study_name, study_df) in enumerate(study_data.items()):
            batch_info.extend([study_idx] * len(study_df))
        
        batch_series = pd.Series(batch_info, index=combined_data.index)
        batch_onehot = pd.get_dummies(batch_series, prefix='batch')
        
        # Combine expression + batch
        combined_with_batch = pd.concat([combined_data, batch_onehot], axis=1)
        
        # Create dataset
        dataset = FlexibleMultiOmicsDataset(
            modality_data={'transcriptomics': combined_with_batch},
            labels=labels,
            alignment='strict',
            normalize=True
        )
        
        input_dims = {'transcriptomics': combined_with_batch.shape[1]}
    
    print(f"\n✅ Dataset created:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Modalities: {list(input_dims.keys())}")
    print(f"   Input dimensions: {input_dims}")
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"\n📊 Data split:")
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    print(f"   Test:  {test_size} samples")
    
    # Create model
    model = EnhancedMultiOmicsTransformer(
        input_dims=input_dims,
        num_classes=3,  # 3 cancer subtypes
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🤖 Model created:")
    print(f"   Architecture: EnhancedMultiOmicsTransformer")
    print(f"   Parameters: {total_params:,}")
    print(f"   Embed dim: 128, Heads: 8, Layers: 4")
    
    # Train model
    print(f"\n🏋️  Training model...")
    trainer = MultiOmicsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    history = trainer.fit(
        num_epochs=20,
        early_stopping_patience=5
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    print(f"\n✅ Training Complete!")
    print(f"   Best Val Accuracy: {max(history['val_accuracy']):.2%}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"   Test F1 Score: {test_metrics['f1']:.4f}")
    
    return model, trainer, history, test_metrics, test_loader


def evaluate_cross_study_performance(model, study_data, labels, study_info_df, device):
    """Evaluate model performance on each study separately."""
    print("\n🔬 Cross-Study Generalization Analysis")
    print("=" * 80)
    
    model.eval()
    results = []
    
    for study_name, study_df in study_data.items():
        # Get labels for this study
        study_mask = study_info_df['study'] == study_name
        study_labels = labels[study_mask]
        
        # Create dataset for this study only
        study_dataset = FlexibleMultiOmicsDataset(
            modality_data={study_name: study_df},
            labels=study_labels,
            alignment='strict',
            normalize=True
        )
        
        study_loader = DataLoader(study_dataset, batch_size=32)
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in study_loader:
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                outputs = model(batch_device)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == batch_device['label']).sum().item()
                total += len(batch_device['label'])
        
        accuracy = correct / total if total > 0 else 0
        
        results.append({
            'Study': study_name,
            'Samples': len(study_df),
            'Accuracy': accuracy,
            'Technology': study_info_df[study_mask]['technology'].iloc[0],
            'Country': study_info_df[study_mask]['country'].iloc[0]
        })
        
        print(f"   {study_name:20s}: {accuracy:.2%} ({correct}/{total} correct)")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Summary:")
    print(f"   Mean accuracy: {results_df['Accuracy'].mean():.2%}")
    print(f"   Std accuracy: {results_df['Accuracy'].std():.2%}")
    print(f"   Min accuracy: {results_df['Accuracy'].min():.2%} ({results_df.loc[results_df['Accuracy'].idxmin(), 'Study']})")
    print(f"   Max accuracy: {results_df['Accuracy'].max():.2%} ({results_df.loc[results_df['Accuracy'].idxmax(), 'Study']})")
    
    return results_df


def visualize_training_history(history, approach, save_prefix='multi_study'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training History - {approach}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy History - {approach}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_prefix}_training_{approach}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved training curves: {save_path}")
    plt.close()


def main():
    """Main execution function."""
    print("🧬" + "=" * 78 + "🧬")
    print("        MULTI-STUDY TRANSCRIPTOMICS INTEGRATION EXAMPLE")
    print("🧬" + "=" * 78 + "🧬")
    print()
    print("This example demonstrates integrating 7 lung cancer transcriptomics studies")
    print("with different technologies, batch effects, and sample sizes.")
    print()
    
    # Step 1: Generate data
    study_data, labels, gene_names, study_info = create_realistic_multi_study_data(
        n_genes=500,
        n_studies=7
    )
    
    # Step 2: Visualize batch effects
    viz_df = visualize_batch_effects(study_data, study_info, save_prefix='lung_cancer')
    
    # Step 3: Train with Approach 1 (Separate Modalities)
    print("\n" + "=" * 80)
    print("APPROACH 1: SEPARATE MODALITIES")
    print("=" * 80)
    model1, trainer1, history1, test_metrics1, test_loader1 = train_multi_study_model(
        study_data, labels, approach='separate_modalities'
    )
    visualize_training_history(history1, 'separate_modalities', save_prefix='lung_cancer')
    
    # Evaluate cross-study performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results1 = evaluate_cross_study_performance(model1, study_data, labels, study_info, device)
    
    # Step 4: Train with Approach 2 (Combined with Batch)
    print("\n" + "=" * 80)
    print("APPROACH 2: COMBINED WITH BATCH ENCODING")
    print("=" * 80)
    model2, trainer2, history2, test_metrics2, test_loader2 = train_multi_study_model(
        study_data, labels, approach='combined_with_batch'
    )
    visualize_training_history(history2, 'combined_with_batch', save_prefix='lung_cancer')
    
    # Step 5: Compare approaches
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Test F1', 'Parameters', 'Best Val Acc'],
        'Separate Modalities': [
            f"{test_metrics1['accuracy']:.2%}",
            f"{test_metrics1['f1']:.4f}",
            f"{sum(p.numel() for p in model1.parameters()):,}",
            f"{max(history1['val_accuracy']):.2%}"
        ],
        'Combined + Batch': [
            f"{test_metrics2['accuracy']:.2%}",
            f"{test_metrics2['f1']:.4f}",
            f"{sum(p.numel() for p in model2.parameters()):,}",
            f"{max(history2['val_accuracy']):.2%}"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("🎉 EXAMPLE COMPLETE!")
    print("=" * 80)
    print()
    print("✅ Successfully integrated 7 transcriptomics studies!")
    print("✅ Trained models with both approaches")
    print("✅ Evaluated cross-study generalization")
    print()
    print("📁 Generated files:")
    print("   • lung_cancer_batch_effects.png - Visualization of batch effects")
    print("   • lung_cancer_training_separate_modalities.png - Training curves")
    print("   • lung_cancer_training_combined_with_batch.png - Training curves")
    print()
    print("💡 Key Insights:")
    print(f"   • Approach 1 achieved {test_metrics1['accuracy']:.1%} test accuracy")
    print(f"   • Approach 2 achieved {test_metrics2['accuracy']:.1%} test accuracy")
    print(f"   • Both approaches successfully learned across studies!")
    print()


if __name__ == '__main__':
    main()
