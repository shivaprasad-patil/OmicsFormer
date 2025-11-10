#!/usr/bin/env python3
"""
Quick Multi-Study Transcriptomics Integration Demo

A streamlined example showing how to integrate multiple transcriptomics datasets
from different studies. This version uses smaller models for faster demonstration.

Scenario: 7 lung cancer studies with different technologies and batch effects.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer
from omicsformer.training.trainer import MultiOmicsTrainer

np.random.seed(42)
torch.manual_seed(42)

def create_quick_multi_study_data(n_genes=200, n_studies=7):
    """Create simulated multi-study transcriptomics data."""
    print("🧬 Creating Multi-Study Dataset")
    print(f"   Studies: {n_studies} | Genes: {n_genes} | Subtypes: 3 (LUAD, LUSC, SCLC)\n")
    
    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
    
    studies = {
        'TCGA_USA': {'n': 100, 'tech': 'RNA-seq', 'batch': 0.1, 'dist': [0.7, 0.2, 0.1]},
        'GEO_Europe': {'n': 80, 'tech': 'microarray', 'batch': 0.8, 'dist': [0.3, 0.5, 0.2]},
        'Japan_Cohort': {'n': 60, 'tech': 'RNA-seq', 'batch': 0.3, 'dist': [0.5, 0.3, 0.2]},
        'UK_Biobank': {'n': 90, 'tech': 'RNA-seq', 'batch': 0.2, 'dist': [0.4, 0.4, 0.2]},
        'China_Discovery': {'n': 70, 'tech': 'microarray', 'batch': 0.6, 'dist': [0.3, 0.4, 0.3]},
        'Canada_Multi': {'n': 50, 'tech': 'RNA-seq', 'batch': 0.25, 'dist': [0.5, 0.3, 0.2]},
        'Australia_Valid': {'n': 50, 'tech': 'RNA-seq', 'batch': 0.15, 'dist': [0.4, 0.35, 0.25]}
    }
    
    study_data = {}
    all_labels = []
    study_info = []
    
    for name, config in studies.items():
        n_samples = config['n']
        labels = np.random.choice([0, 1, 2], size=n_samples, p=config['dist'])
        expression = np.random.randn(n_samples, n_genes)
        
        # Add subtype-specific signatures
        for i, label in enumerate(labels):
            if label == 0:  # LUAD
                expression[i, :30] += np.random.normal(2.5, 0.4, 30)
            elif label == 1:  # LUSC
                expression[i, 30:60] += np.random.normal(3.0, 0.5, 30)
            else:  # SCLC
                expression[i, 60:90] += np.random.normal(3.5, 0.6, 30)
        
        # Add batch effects
        if config['tech'] == 'microarray':
            expression += np.random.normal(0.5, 0.2, (1, n_genes))
        expression += np.random.randn(n_samples, n_genes) * config['batch']
        
        sample_ids = [f'{name}_S{i:03d}' for i in range(n_samples)]
        study_data[name] = pd.DataFrame(expression, index=sample_ids, columns=gene_names)
        
        all_labels.extend(labels)
        for sid, lbl in zip(sample_ids, labels):
            study_info.append({'sample_id': sid, 'study': name, 'label': lbl, 'technology': config['tech']})
        
        print(f"   ✓ {name:18s}: {n_samples:3d} samples, {config['tech']:10s}, "
              f"Subtypes={np.bincount(labels, minlength=3)}")
    
    study_info_df = pd.DataFrame(study_info)
    all_sample_ids = study_info_df['sample_id'].tolist()
    labels_series = pd.Series(all_labels, index=all_sample_ids, name='cancer_subtype')
    
    print(f"\n   Total: {len(labels_series)} samples")
    print(f"   Distribution: LUAD={sum(labels_series==0)}, LUSC={sum(labels_series==1)}, SCLC={sum(labels_series==2)}\n")
    
    return study_data, labels_series, study_info_df


def train_model(study_data, labels, approach='separate'):
    """Train model with specified approach."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if approach == 'separate':
        print("🔬 APPROACH 1: Each Study as Separate Modality")
        print("   (Enables cross-study attention learning)\n")
        
        dataset = FlexibleMultiOmicsDataset(
            modality_data=study_data,
            labels=labels,
            alignment='union',
            normalize=True
        )
        input_dims = {name: df.shape[1] for name, df in study_data.items()}
    
    else:
        print("🔬 APPROACH 2: Combined with Batch Encoding")
        print("   (More parameter efficient)\n")
        
        combined = pd.concat(list(study_data.values()), axis=0)
        batch_info = []
        for idx, (name, df) in enumerate(study_data.items()):
            batch_info.extend([idx] * len(df))
        
        batch_onehot = pd.get_dummies(pd.Series(batch_info, index=combined.index), prefix='batch')
        combined_with_batch = pd.concat([combined, batch_onehot], axis=1)
        
        dataset = FlexibleMultiOmicsDataset(
            modality_data={'transcriptomics': combined_with_batch},
            labels=labels,
            alignment='strict',
            normalize=True
        )
        input_dims = {'transcriptomics': combined_with_batch.shape[1]}
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    print(f"   Dataset: {len(dataset)} samples")
    print(f"   Modalities: {list(input_dims.keys())}")
    print(f"   Split: Train={train_size}, Val={val_size}, Test={test_size}\n")
    
    # Create smaller model for quick demo
    model = EnhancedMultiOmicsTransformer(
        input_dims=input_dims,
        num_classes=3,
        embed_dim=64,  # Smaller
        num_heads=4,   # Fewer heads
        num_layers=2,  # Fewer layers
        dropout=0.2
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {params:,} parameters\n")
    
    # Train
    print("   Training...")
    trainer = MultiOmicsTrainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device)
    history = trainer.fit(num_epochs=5, early_stopping_patience=3)  # Quick training
    
    # Test - manually evaluate on test set
    model.eval()
    all_preds = []
    all_labels_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch_device)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(batch_device['label'].cpu().numpy())
    
    from sklearn.metrics import f1_score, accuracy_score
    test_metrics = {
        'accuracy': accuracy_score(all_labels_list, all_preds),
        'f1': f1_score(all_labels_list, all_preds, average='weighted')
    }
    
    print(f"\n   ✅ Training Complete!")
    print(f"      Best Val Acc: {max(history['val_accuracy']):.2%}")
    print(f"      Test Acc: {test_metrics['accuracy']:.2%}")
    print(f"      Test F1: {test_metrics['f1']:.4f}\n")
    
    return model, history, test_metrics


def main():
    print("\n" + "="*80)
    print("  QUICK MULTI-STUDY TRANSCRIPTOMICS INTEGRATION DEMO")
    print("="*80 + "\n")
    
    # Generate data
    study_data, labels, study_info = create_quick_multi_study_data(n_genes=200, n_studies=7)
    
    # Train both approaches
    print("="*80)
    model1, history1, metrics1 = train_model(study_data, labels, approach='separate')
    
    print("="*80)
    model2, history2, metrics2 = train_model(study_data, labels, approach='combined')
    
    # Compare
    print("="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    
    print(f"  {'Metric':<20} {'Separate Modalities':<20} {'Combined + Batch':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'Test Accuracy':<20} {metrics1['accuracy']:>19.2%} {metrics2['accuracy']:>19.2%}")
    print(f"  {'Test F1':<20} {metrics1['f1']:>19.4f} {metrics2['f1']:>19.4f}")
    print(f"  {'Parameters':<20} {sum(p.numel() for p in model1.parameters()):>19,} "
          f"{sum(p.numel() for p in model2.parameters()):>19,}")
    
    print("\n" + "="*80)
    print("✅ SUCCESS! OmicsFormer can integrate multiple transcriptomics studies!")
    print("="*80 + "\n")
    
    print("Key Features Demonstrated:")
    print("  ✓ Integration of 7 studies with 500 total samples")
    print("  ✓ Handling different technologies (RNA-seq vs microarray)")
    print("  ✓ Managing varying batch effects across studies")
    print("  ✓ Two complementary integration approaches")
    print("  ✓ Cross-study learning and generalization")
    print()


if __name__ == '__main__':
    main()
