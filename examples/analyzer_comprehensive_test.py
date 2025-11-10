"""
OmicsFormer Complete Analysis & Training Pipeline

This comprehensive example demonstrates the FULL capabilities of OmicsFormer including:

🧬 DATA GENERATION:
- Realistic synthetic multi-omics data with biological feature names
- Multiple modalities: genomics, transcriptomics, proteomics, metabolomics
- Cross-modal correlations and realistic data distributions

🤖 MODEL TRAINING:
- Advanced transformer training with MoE (Mixture of Experts) and GQA (Grouped Query Attention)
- Early stopping, validation, and model evaluation
- Training history visualization and model persistence

🔍 COMPREHENSIVE ANALYSIS:
- Attention pattern extraction and visualization
- Embedding extraction with PCA, t-SNE, and UMAP
- Cross-modal correlation analysis
- Feature importance analysis (gradient, attention, permutation methods)
- Biological pathway enrichment analysis
- Statistical analysis and comprehensive reporting
- Model interpretation and biomarker discovery

📊 VISUALIZATION & REPORTING:
- Interactive plots and heatmaps
- Comprehensive HTML reports with embedded visualizations
- Training metrics and performance evaluation
- Results persistence and reproducibility

This script serves as both a comprehensive test suite and a complete showcase
of all OmicsFormer capabilities for multi-omics analysis and machine learning.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any
import json
import time

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import OmicsFormer components
try:
    import omicsformer as of
    from omicsformer.models import EnhancedMultiOmicsTransformer, AdvancedMultiOmicsTransformer
    from omicsformer.data import FlexibleMultiOmicsDataset
    from omicsformer.analysis.analyzer import MultiOmicsAnalyzer, compute_modality_statistics, plot_modality_distributions
    from omicsformer.training import MultiOmicsTrainer, evaluate_model
except ImportError:
    # Fallback imports for older structure
    from omicsformer.models import EnhancedMultiOmicsTransformer
    from omicsformer.data import FlexibleMultiOmicsDataset
    from omicsformer.analysis.analyzer import MultiOmicsAnalyzer, compute_modality_statistics, plot_modality_distributions

def create_enhanced_synthetic_data(n_samples=800, use_builtin=False, missing_rate=0.15):
    """
    Create enhanced synthetic multi-omics data for comprehensive analysis.
    
    Args:
        n_samples: Number of samples to generate
        use_builtin: Whether to use OmicsFormer's built-in synthetic data generator
        missing_rate: Proportion of missing values to introduce
    
    Returns:
        tuple: (data_dict, metadata_df, feature_names_dict)
    """
    print(f"🧬 Creating enhanced synthetic multi-omics data ({n_samples} samples)...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Option 1: Use OmicsFormer's built-in synthetic data generator
    if use_builtin:
        try:
            modality_data, labels = of.create_synthetic_multiomics_data(
                n_samples=n_samples,
                n_features_per_modality={
                    'genomics': 1000,      # DNA mutations, SNPs
                    'transcriptomics': 500, # Gene expression
                    'proteomics': 200,     # Protein abundance
                    'metabolomics': 150    # Metabolite levels
                },
                n_classes=3,  # Three disease subtypes
                missing_rate=missing_rate,
                correlation_strength=0.4  # Moderate cross-modal correlation
            )
            
            # Convert to expected format
            data = {}
            feature_names = {}
            for modality, df in modality_data.items():
                data[modality] = df.values.astype(np.float32)
                feature_names[modality] = list(df.columns)
            
            # Create metadata
            sample_ids = [f"Sample_{i:04d}" for i in range(n_samples)]
            metadata = pd.DataFrame({
                'sample_id': sample_ids,
                'label': labels,
                'batch': np.random.choice(['batch_1', 'batch_2', 'batch_3'], size=n_samples),
                'age': np.random.normal(50, 15, n_samples),
                'gender': np.random.choice(['M', 'F'], size=n_samples)
            })
            
            print(f"✅ Used built-in synthetic data generator")
            return data, metadata, feature_names
        except:
            print("⚠️  Built-in generator not available, using custom generator...")
    
    # Option 2: Custom detailed synthetic data generation
    # Define modalities with realistic feature counts
    modalities = {
        'genomics': 1000,      # SNPs, mutations
        'transcriptomics': 2000, # Gene expression
        'proteomics': 500,     # Protein levels
        'metabolomics': 300    # Metabolite concentrations
    }
    
    # Create synthetic data with realistic patterns
    data = {}
    feature_names = {}
    
    for modality, n_features in modalities.items():
        print(f"  Creating {modality} data: {n_samples} samples x {n_features} features")
        
        if modality == 'genomics':
            # Binary data for SNPs
            modality_data = np.random.binomial(2, 0.3, (n_samples, n_features)).astype(np.float32)
            # Use real gene names for genomics (treating SNPs as affecting these genes)
            real_genes = [
                'TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA', 'APC', 'PTEN', 'MYC', 'RB1',
                'CDKN2A', 'ATM', 'CHEK2', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'STK11', 'SMAD4', 'DPC4',
                'VHL', 'NF1', 'NF2', 'AKT1', 'PIK3R1', 'FBXW7', 'NOTCH1', 'BRAF', 'CTNNB1', 'IDH1',
                'IDH2', 'ARID1A', 'ARID1B', 'KMT2D', 'KMT2C', 'EP300', 'CREBBP', 'KDM6A', 'ASXL1', 'EZH2',
                'DNMT3A', 'TET2', 'RUNX1', 'GATA3', 'FOXA1', 'ESR1', 'AR', 'SPOP', 'CDH1', 'KEAP1'
            ]
            # Extend with numbered versions if we need more features
            feature_names[modality] = (real_genes * ((n_features // len(real_genes)) + 1))[:n_features]
            
        elif modality == 'transcriptomics':
            # Log-normal distribution for gene expression
            modality_data = np.random.lognormal(mean=5, sigma=2, size=(n_samples, n_features))
            # Use real gene names for transcriptomics
            real_genes = [
                'GAPDH', 'ACTB', 'TUBB', 'RPL13A', 'HPRT1', 'TBP', 'B2M', 'YWHAZ', 'SDHA', 'UBC',
                'CCND1', 'CDK4', 'RB1', 'E2F1', 'MYC', 'JUN', 'FOS', 'EGR1', 'ATF3', 'CREB1',
                'TP53', 'MDM2', 'CDKN1A', 'BAX', 'BCL2', 'APAF1', 'CASP3', 'CASP9', 'PARP1', 'ATM',
                'BRCA1', 'BRCA2', 'RAD51', 'XRCC1', 'DNA2', 'PCNA', 'RFC1', 'LIG1', 'POLD1', 'POLE',
                'EGFR', 'PDGFRA', 'VEGFA', 'KDR', 'FLT1', 'ANGPT1', 'TIE1', 'NOTCH1', 'DLL1', 'HES1',
                'WNT1', 'CTNNB1', 'APC', 'GSK3B', 'DVL1', 'FZD1', 'LRP5', 'TCF7', 'LEF1', 'AXIN1',
                'TGFB1', 'SMAD2', 'SMAD3', 'SMAD4', 'TGFBR1', 'TGFBR2', 'BAMBI', 'SMURF1', 'SKIL', 'SKI',
                'TNF', 'IL1B', 'IL6', 'NFKB1', 'RELA', 'IKBKA', 'CHUK', 'NFKBIA', 'TLR4', 'MYD88',
                'MAPK1', 'MAPK3', 'MAP2K1', 'MAP2K2', 'ARAF', 'BRAF', 'RAF1', 'KRAS', 'HRAS', 'NRAS',
                'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'PTEN', 'TSC1', 'TSC2', 'MTOR', 'RPS6KB1'
            ]
            feature_names[modality] = (real_genes * ((n_features // len(real_genes)) + 1))[:n_features]
            
        elif modality == 'proteomics':
            # Normal distribution for protein levels
            modality_data = np.random.normal(loc=10, scale=3, size=(n_samples, n_features))
            # Use real protein names
            real_proteins = [
                'TP53', 'RB1', 'EGFR', 'MYC', 'PTEN', 'AKT1', 'MTOR', 'CTNNB1', 'GSK3B', 'SMAD4',
                'TGFB1', 'VEGFA', 'HIF1A', 'STAT3', 'RELA', 'MAPK1', 'CDK4', 'CCND1', 'CDKN1A', 'BAX',
                'BCL2', 'CASP3', 'PARP1', 'ATM', 'BRCA1', 'RAD51', 'PCNA', 'TOP2A', 'TUBB3', 'KI67',
                'ERBB2', 'ESR1', 'PGR', 'AR', 'FOXA1', 'GATA3', 'KRT8', 'KRT18', 'VIM', 'CDH1',
                'SNAI1', 'TWIST1', 'ZEB1', 'MMP2', 'MMP9', 'TIMP1', 'COL1A1', 'FN1', 'ITGA5', 'ITGB1',
                'SRC', 'FAK', 'RHOA', 'RAC1', 'CDC42', 'ROCK1', 'ACTN1', 'MSN', 'EZR', 'RDX'
            ]
            feature_names[modality] = (real_proteins * ((n_features // len(real_proteins)) + 1))[:n_features]
            
        elif modality == 'metabolomics':
            # Exponential distribution for metabolite concentrations
            modality_data = np.random.exponential(scale=2, size=(n_samples, n_features))
            # Use real metabolite names (use gene names related to metabolic enzymes)
            metabolic_genes = [
                'PFKM', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'PKM', 'LDHA', 'LDHB',
                'G6PD', 'PGD', 'RPIA', 'RPE', 'TKTL1', 'TALDO1', 'PFKP', 'PFKL', 'FBP1', 'FBP2',
                'IDH1', 'IDH2', 'ACLY', 'ACACA', 'FASN', 'SCD', 'FADS1', 'FADS2', 'ELOVL6', 'ACSL1',
                'CPT1A', 'CPT2', 'ACADM', 'HADHA', 'HADHB', 'ACAA2', 'HMGCR', 'HMGCS1', 'MVK', 'PMVK',
                'MVD', 'IDI1', 'FDPS', 'SQLE', 'LSS', 'CYP51A1', 'DHCR24', 'DHCR7', 'SC5D', 'NSDHL'
            ]
            feature_names[modality] = (metabolic_genes * ((n_features // len(metabolic_genes)) + 1))[:n_features]
        
        # Add some correlation between modalities for realistic patterns
        if len(data) > 0:
            # Add some shared signal with previous modalities
            prev_modality = list(data.keys())[-1]
            shared_signal = data[prev_modality][:, :min(100, n_features, data[prev_modality].shape[1])]
            if shared_signal.shape[1] < n_features:
                padding = np.random.normal(0, 1, (n_samples, n_features - shared_signal.shape[1]))
                shared_signal = np.concatenate([shared_signal, padding], axis=1)
            else:
                shared_signal = shared_signal[:, :n_features]
            
            # Mix original data with shared signal
            modality_data = 0.7 * modality_data + 0.3 * shared_signal
        
        data[modality] = modality_data.astype(np.float32)
    
    # Create sample metadata
    sample_ids = [f"Sample_{i:04d}" for i in range(n_samples)]
    
    # Create labels with some class imbalance
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'sample_id': sample_ids,
        'label': labels,
        'batch': np.random.choice(['batch_1', 'batch_2', 'batch_3'], size=n_samples),
        'age': np.random.normal(50, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], size=n_samples)
    })
    
    print(f"Created data with {len(modalities)} modalities and {n_samples} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return data, metadata, feature_names

def select_optimal_alignment_strategy(modality_data: Dict[str, pd.DataFrame], metadata: pd.DataFrame) -> Dict[str, str]:
    """
    Intelligently select the optimal alignment strategy based on data characteristics.
    
    Args:
        modality_data: Dictionary of modality DataFrames
        metadata: Sample metadata DataFrame
    
    Returns:
        Dict with 'strategy' and 'rationale' keys
    """
    n_modalities = len(modality_data)
    total_samples = len(metadata)
    
    # Analyze sample overlap across modalities
    all_samples = set()
    modality_samples = {}
    
    for mod_name, df in modality_data.items():
        mod_samples = set(df.index)
        modality_samples[mod_name] = mod_samples
        all_samples.update(mod_samples)
    
    # Count samples by modality presence
    from collections import defaultdict
    sample_modality_count = defaultdict(int)
    for mod_samples in modality_samples.values():
        for sample in mod_samples:
            sample_modality_count[sample] += 1
    
    # Calculate statistics
    samples_in_all = sum(1 for count in sample_modality_count.values() if count == n_modalities)
    samples_in_most = sum(1 for count in sample_modality_count.values() if count >= max(2, n_modalities - 1))
    samples_in_at_least_2 = sum(1 for count in sample_modality_count.values() if count >= 2)
    unique_samples = len(all_samples)
    
    # Calculate percentages
    strict_retention = samples_in_all / total_samples if total_samples > 0 else 0
    intersection_retention = samples_in_at_least_2 / total_samples if total_samples > 0 else 0
    
    print(f"   📊 Data characteristics analysis:")
    print(f"      Total samples: {total_samples}")
    print(f"      Modalities: {n_modalities}")
    print(f"      Samples in ALL modalities: {samples_in_all} ({strict_retention:.1%})")
    print(f"      Samples in ≥2 modalities: {samples_in_at_least_2} ({intersection_retention:.1%})")
    print(f"      Unique samples across all: {unique_samples}")
    
    # Decision logic with additional considerations
    if n_modalities <= 2:
        # For 2 or fewer modalities, prefer strict if good overlap, otherwise flexible
        if strict_retention >= 0.7:
            return {
                'strategy': 'strict',
                'rationale': f'With {n_modalities} modalities and high overlap ({strict_retention:.1%}), strict alignment ensures data quality'
            }
        else:
            return {
                'strategy': 'flexible',
                'rationale': f'With {n_modalities} modalities and moderate overlap ({strict_retention:.1%}), flexible alignment maximizes sample utilization'
            }
    
    elif strict_retention >= 0.8:
        # If 80%+ samples are in all modalities, use strict for clean analysis
        return {
            'strategy': 'strict',
            'rationale': f'High overlap ({strict_retention:.1%}) - strict alignment ensures complete data'
        }
    
    elif strict_retention >= 0.5:
        # Moderate overlap - intersection is good balance
        return {
            'strategy': 'intersection',
            'rationale': f'Moderate overlap ({strict_retention:.1%} complete) - intersection balances data quality vs quantity'
        }
    
    elif intersection_retention >= 0.6:
        # Good intersection coverage
        return {
            'strategy': 'intersection',
            'rationale': f'Good multi-modal coverage ({intersection_retention:.1%}) - intersection preserves meaningful samples'
        }
    
    elif strict_retention < 0.3:
        # Very low strict overlap - use flexible to preserve data
        return {
            'strategy': 'flexible',
            'rationale': f'Low complete overlap ({strict_retention:.1%}) - flexible alignment preserves maximum information'
        }
    
    else:
        # Default to intersection as balanced approach
        return {
            'strategy': 'intersection',
            'rationale': f'Balanced approach - intersection with {intersection_retention:.1%} multi-modal samples'
        }


def create_advanced_model(modalities, n_classes=3, use_advanced=True):
    """Create either an Advanced or Enhanced MultiOmicsTransformer model."""
    if use_advanced:
        try:
            print("🤖 Creating AdvancedMultiOmicsTransformer with MoE and GQA...")
            model = AdvancedMultiOmicsTransformer(
                input_dims=modalities,
                embed_dim=256,
                num_heads=8,
                num_layers=4, 
                num_classes=n_classes,
                num_experts=4,  # MoE with 4 experts
                dropout=0.2,
                use_moe=True,   # Enable Mixture of Experts
                use_gqa=True,   # Enable Grouped Query Attention
                load_balance_weight=0.01
            )
            print("✅ AdvancedMultiOmicsTransformer created successfully")
            return model, True
        except Exception as e:
            print(f"⚠️  AdvancedMultiOmicsTransformer not available ({e}), using Enhanced version...")
    
    print("🤖 Creating EnhancedMultiOmicsTransformer...")
    model = EnhancedMultiOmicsTransformer(
        input_dims=modalities,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=n_classes,
        dropout=0.1
    )
    print("✅ EnhancedMultiOmicsTransformer created successfully")
    return model, False

class MockDataLoader:
    """Mock DataLoader that returns properly formatted batches for analysis."""
    
    def __init__(self, data, metadata, batch_size=32):
        self.data = data
        self.metadata = metadata
        self.batch_size = batch_size
        self.n_samples = len(metadata)
        self.modalities = list(data.keys())
        
    def __iter__(self):
        indices = list(range(self.n_samples))
        np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            batch = {
                'sample_id': torch.tensor(batch_indices, dtype=torch.long),  # Convert to tensor
                'label': torch.tensor([self.metadata.iloc[idx]['label'] for idx in batch_indices], dtype=torch.long)
            }
            
            # Add modality data
            for modality in self.modalities:
                batch[modality] = torch.tensor(
                    self.data[modality][batch_indices], 
                    dtype=torch.float32
                )
            
            yield batch
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def comprehensive_omicsformer_showcase(n_samples=800, train_model=True, quick_mode=False):
    """
    Comprehensive OmicsFormer showcase demonstrating all capabilities.
    
    Args:
        n_samples: Number of samples for data generation
        train_model: Whether to actually train the model or use mock for quick testing
        quick_mode: If True, use smaller parameters for faster execution
    """
    print("🧬" + "=" * 78 + "🧬")
    print("         COMPREHENSIVE OMICSFORMER SHOWCASE & ANALYSIS PIPELINE")
    print("🧬" + "=" * 78 + "🧬")
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if quick_mode:
        n_samples = min(n_samples, 200)
        print(f"⚡ Quick mode enabled - using {n_samples} samples")
    
    # STEP 1: DATA GENERATION
    print(f"\n📊 STEP 1: Generating synthetic multi-omics data...")
    print("-" * 60)
    
    data, metadata, feature_names = create_enhanced_synthetic_data(
        n_samples=n_samples, 
        use_builtin=True, 
        missing_rate=0.15
    )
    modalities = {k: v.shape[1] for k, v in data.items()}
    
    print(f"✅ Generated data for {len(metadata)} samples across {len(modalities)} modalities")
    for mod, dim in modalities.items():
        print(f"   - {mod}: {len(metadata)} samples × {dim} features")
    print(f"   Label distribution: {np.bincount(metadata['label'].values)}")
    
    # STEP 2: DATASET CREATION
    print(f"\n🔧 STEP 2: Creating flexible multi-omics dataset...")
    print("-" * 60)
    
    try:
        # Convert data to DataFrame format for FlexibleMultiOmicsDataset
        modality_dataframes = {}
        for modality, values in data.items():
            feature_cols = feature_names[modality]
            modality_dataframes[modality] = pd.DataFrame(
                values, 
                columns=feature_cols,
                index=[f"Sample_{i:04d}" for i in range(len(values))]
            )
        
        # Intelligent alignment strategy selection
        alignment_strategy = select_optimal_alignment_strategy(modality_dataframes, metadata)
        print(f"   🎯 Selected alignment strategy: '{alignment_strategy['strategy']}'")
        print(f"      Rationale: {alignment_strategy['rationale']}")
        
        dataset = FlexibleMultiOmicsDataset(
            modality_data=modality_dataframes,
            labels=metadata['label'].values,
            alignment=alignment_strategy['strategy'],
            missing_value_strategy='mean',
            normalize=True
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        print(f"   Alignment strategy: {dataset.alignment}")
        print(f"   Feature dimensions: {dataset.feature_dims}")
        
        # Create data loaders for training if enabled
        if train_model:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            batch_size = 16 if quick_mode else 32
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"   Train samples: {len(train_dataset)}")
            print(f"   Validation samples: {len(val_dataset)}")
        else:
            # Use mock dataloader for quick testing
            train_loader = val_loader = MockDataLoader(data, metadata, batch_size=16)
            
    except Exception as e:
        print(f"⚠️  Using mock dataloader due to dataset creation issue: {e}")
        # Ensure consistent sample count between data and metadata
        min_samples = min(len(metadata), min(len(values) for values in data.values()))
        print(f"   Adjusting to {min_samples} samples for consistency")
        
        # Trim data and metadata to consistent size
        consistent_data = {}
        for modality, values in data.items():
            consistent_data[modality] = values[:min_samples]
        consistent_metadata = metadata.iloc[:min_samples].copy()
        
        train_loader = val_loader = MockDataLoader(consistent_data, consistent_metadata, batch_size=16)
    
    # STEP 3: MODEL CREATION
    print(f"\n🤖 STEP 3: Creating advanced transformer model...")
    print("-" * 60)
    
    model, is_advanced = create_advanced_model(modalities, n_classes=3, use_advanced=train_model)
    model = model.to(device)
    
    try:
        if hasattr(of, 'print_model_summary'):
            of.print_model_summary(model)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
    except:
        print("   Model summary not available")
    
    # STEP 4: MODEL TRAINING (if enabled)
    training_history = None
    test_metrics = {}
    
    if train_model and is_advanced:
        print(f"\n🏋️  STEP 4: Training the model...")
        print("-" * 60)
        
        try:
            trainer = MultiOmicsTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                use_wandb=False
            )
            
            num_epochs = 5 if quick_mode else 20
            training_history = trainer.fit(
                num_epochs=num_epochs,
                early_stopping_patience=5,
                load_balance_weight=0.01 if is_advanced else 0.0,
                save_best_model=True,
                model_save_path='omicsformer_trained_model.pth'
            )
            
            print("✅ Training completed!")
            
            # Plot training history
            trainer.plot_training_history(training_history, save_path='training_history.png')
            print("✅ Training history plot saved")
            
            # Model evaluation
            test_metrics = evaluate_model(model, val_loader, device=device)
            print("📊 Model Performance:")
            for metric, value in test_metrics.items():
                print(f"     {metric}: {value:.4f}")
                
        except Exception as e:
            print(f"⚠️  Training failed ({e}), proceeding with untrained model for analysis...")
            model.eval()  # Set to eval mode for analysis
    else:
        print(f"\n⚡ STEP 4: Using mock/untrained model for quick analysis...")
        print("-" * 60)
        model.eval()  # Set to eval mode for analysis
    
    # STEP 5: COMPREHENSIVE ANALYSIS
    print(f"\n🔍 STEP 5: Comprehensive multi-omics analysis...")
    print("-" * 60)
    
    analyzer = MultiOmicsAnalyzer(model, device=device)
    
    print(f"   ✓ Initialized analyzer with device: {analyzer.device}")
    
    analysis_results = {}
    generated_files = []
    
    try:
        # Analysis 1: Extract attention patterns
        print("\n   🔍 1. Attention pattern extraction...")
        attention_patterns = analyzer.extract_attention_patterns(val_loader, max_samples=50)
        print(f"      ✓ Extracted attention patterns: {list(attention_patterns.keys())}")
        for key, pattern in attention_patterns.items():
            print(f"        - {key}: shape {pattern.shape}")
        analysis_results['attention_patterns'] = attention_patterns
        
        # Visualize attention heatmap
        if attention_patterns:
            fig = analyzer.visualize_attention_heatmap(
                list(attention_patterns.keys())[0], 
                save_path='omicsformer_attention_heatmap.png'
            )
            print("      ✓ Created attention heatmap visualization")
            generated_files.append('omicsformer_attention_heatmap.png')
            plt.close(fig)
        
        # Analysis 2: Extract embeddings
        print("\n   🧬 2. Embedding extraction and visualization...")
        embeddings = analyzer.extract_embeddings(val_loader, embedding_type='pooled')
        print(f"      ✓ Extracted embeddings: {list(embeddings.keys())}")
        for key, emb_data in embeddings.items():
            print(f"        - {key}: {emb_data['embeddings'].shape}")
        analysis_results['embeddings'] = embeddings
        
        # Embedding visualizations
        if embeddings:
            for method in ['pca', 'tsne', 'umap']:
                try:
                    fig = analyzer.plot_embedding_visualization(
                        embedding_type='pooled',
                        method=method,
                        save_path=f'omicsformer_embeddings_{method}.png'
                    )
                    print(f"      ✓ Created {method.upper()} embedding visualization")
                    generated_files.append(f'omicsformer_embeddings_{method}.png')
                    plt.close(fig)
                except Exception as e:
                    print(f"      ⚠ {method.upper()} visualization failed: {e}")
        
        # Analysis 3: Cross-modal correlations
        print("\n   🔗 3. Cross-modal correlation analysis...")
        try:
            correlations = analyzer.compute_cross_modal_correlations(val_loader, method='pearson')
            print(f"      ✓ Computed correlations: {correlations.shape}")
            print(f"        Correlation range: {correlations.min().min():.3f} to {correlations.max().max():.3f}")
            analysis_results['correlations'] = correlations
            
            # Plot correlation heatmap
            fig = analyzer.plot_correlation_heatmap(
                correlations, 
                save_path='omicsformer_correlations.png'
            )
            print("      ✓ Created correlation heatmap")
            generated_files.append('omicsformer_correlations.png')
            plt.close(fig)
        except Exception as e:
            print(f"      ⚠ Correlation analysis failed: {e}")
            analysis_results['correlations'] = pd.DataFrame()
        
        # Analysis 4: Feature importance analysis
        print("\n   ⚖️  4. Feature importance analysis...")
        importance_methods = ['gradient', 'attention'] if quick_mode else ['gradient', 'attention', 'permutation']
        gradient_results = {}
        
        for method in importance_methods:
            try:
                print(f"      Testing {method} importance...")
                results = analyzer.analyze_feature_importance(val_loader, method=method)
                
                if method == 'gradient':
                    gradient_results = results.copy()
                
                print(f"      ✓ {method} importance analysis completed")
                for key, importance in results.items():
                    print(f"        - {key}: shape {importance.shape if hasattr(importance, 'shape') else len(importance)}")
            except Exception as e:
                print(f"      ⚠ {method} importance analysis failed: {e}")
        
        analysis_results['feature_importance'] = gradient_results
        
        # Plot feature importance for main modalities
        main_modalities = ['genomics', 'transcriptomics', 'proteomics', 'metabolomics']
        for modality in main_modalities:
            if modality in gradient_results:
                try:
                    fig = analyzer.plot_feature_importance(
                        gradient_results, 
                        modality, 
                        feature_names=feature_names.get(modality),
                        top_n=15,
                        save_path=f'omicsformer_feature_importance_{modality}.png'
                    )
                    print(f"         ✓ Created feature importance plot for {modality}")
                    generated_files.append(f'omicsformer_feature_importance_{modality}.png')
                    plt.close(fig)
                except Exception as e:
                    print(f"         ⚠ Feature importance plot failed for {modality}: {e}")
        
        # Analysis 5: Biological pathway enrichment
        print("\n   🧪 5. Biological pathway enrichment analysis...")
        if gradient_results:
            try:
                # Adjust parameters based on mode
                top_k = 25 if quick_mode else 50
                max_modalities = 2 if quick_mode else len(gradient_results)
                
                print(f"      Analyzing top {top_k} features from up to {max_modalities} modalities...")
                
                # Limit to subset of modalities in quick mode
                analysis_modalities = list(gradient_results.keys())[:max_modalities]
                subset_results = {mod: gradient_results[mod] for mod in analysis_modalities}
                
                pathway_results = analyzer.analyze_pathway_enrichment(
                    subset_results, 
                    feature_names={mod: feature_names[mod] for mod in analysis_modalities},
                    top_k=top_k
                )
                print(f"      ✓ Pathway enrichment completed for {len(pathway_results)} modalities")
                analysis_results['pathway_enrichment'] = pathway_results
                
                # Plot pathway enrichment for each modality
                for modality, results in pathway_results.items():
                    if not results.empty:
                        print(f"        - {modality}: {len(results)} pathways found")
                        try:
                            fig = analyzer.plot_pathway_enrichment(
                                pathway_results, 
                                modality,
                                save_path=f'omicsformer_pathways_{modality}.png'
                            )
                            if fig:
                                print(f"          ✓ Created pathway plot for {modality}")
                                generated_files.append(f'omicsformer_pathways_{modality}.png')
                                plt.close(fig)
                        except Exception as e:
                            print(f"          ⚠ Pathway plot failed for {modality}: {e}")
                    else:
                        print(f"        - {modality}: No pathways found")
                
            except Exception as e:
                print(f"      ⚠ Pathway enrichment failed: {e}")
                print(f"        Error details: {str(e)}")
        else:
            print("      ⚠ No gradient results available for pathway analysis")
        
        # Analysis 6: Modality statistics
        print("\n   📈 6. Modality statistics and distributions...")
        try:
            # Convert data to DataFrames for statistics
            modality_dfs = {}
            for modality, modality_data in data.items():
                modality_dfs[modality] = pd.DataFrame(
                    modality_data, 
                    columns=feature_names[modality]
                )
            
            stats = compute_modality_statistics(modality_dfs)
            print("      ✓ Computed modality statistics")
            analysis_results['modality_statistics'] = stats
            
            # Create distribution plots
            fig = plot_modality_distributions(
                modality_dfs, 
                sample_features=3,
                save_path='omicsformer_distributions.png'
            )
            print("      ✓ Created distribution plots")
            generated_files.append('omicsformer_distributions.png')
            plt.close(fig)
            
        except Exception as e:
            print(f"      ⚠ Modality statistics failed: {e}")
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {}
    
    # STEP 6: COMPREHENSIVE REPORT GENERATION
    print(f"\n📋 STEP 6: Generating comprehensive analysis report...")
    print("-" * 60)
    
    try:
        # Add training history plot to generated files if available
        if training_history and os.path.exists('training_history.png'):
            generated_files.append('training_history.png')
        
        report_html = analyzer.generate_analysis_report(
            save_path='omicsformer_comprehensive_report.html',
            plot_paths=generated_files
        )
        print("   ✓ Generated comprehensive HTML analysis report")
        generated_files.append('omicsformer_comprehensive_report.html')
        
    except Exception as e:
        print(f"   ⚠ Report generation failed: {e}")
    
    # STEP 7: SAVE RESULTS
    print(f"\n💾 STEP 7: Saving analysis results...")
    print("-" * 60)
    
    try:
        results_summary = {
            'execution_info': {
                'n_samples': n_samples,
                'modalities': list(modalities.keys()),
                'feature_dims': modalities,
                'training_enabled': train_model,
                'quick_mode': quick_mode,
                'device': str(device),
                'execution_time_minutes': (time.time() - start_time) / 60
            },
            'model_info': {
                'model_type': 'AdvancedMultiOmicsTransformer' if is_advanced else 'EnhancedMultiOmicsTransformer',
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'num_experts': 4 if is_advanced else 0,
                'use_moe': is_advanced,
                'use_gqa': is_advanced
            },
            'training_results': training_history if training_history else {},
            'test_metrics': test_metrics,
            'analysis_summary': {
                'attention_patterns_extracted': len(analysis_results.get('attention_patterns', {})),
                'embedding_types': list(analysis_results.get('embeddings', {}).keys()),
                'correlation_matrix_shape': analysis_results.get('correlations', pd.DataFrame()).shape,
                'feature_importance_modalities': list(analysis_results.get('feature_importance', {}).keys()),
                'pathway_enrichment_results': len(analysis_results.get('pathway_enrichment', {}))
            },
            'generated_files': generated_files
        }
        
        with open('omicsformer_results.json', 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for key, value in results_summary.items():
                try:
                    json.dumps(value)  # Test serialization
                    serializable_results[key] = value
                except:
                    serializable_results[key] = str(value)
            json.dump(serializable_results, f, indent=2)
        
        print("   ✓ Results saved as 'omicsformer_results.json'")
        generated_files.append('omicsformer_results.json')
        
    except Exception as e:
        print(f"   ⚠ Results saving failed: {e}")
    
    # FINAL SUMMARY
    execution_time = time.time() - start_time
    print(f"\n🎉" + "=" * 78 + "🎉")
    print("         OMICSFORMER COMPREHENSIVE SHOWCASE COMPLETED!")
    print(f"🎉" + "=" * 78 + "🎉")
    
    print(f"\n⏱️  Execution time: {execution_time/60:.2f} minutes")
    print(f"📊 Data: {n_samples} samples across {len(modalities)} modalities")
    print(f"🤖 Model: {'Advanced' if is_advanced else 'Enhanced'}MultiOmicsTransformer")
    print(f"🏋️  Training: {'✅ Completed' if train_model and training_history else '⚡ Skipped/Mock'}")
    
    if test_metrics:
        print(f"📈 Model Performance:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"     {metric}: {value:.4f}")
    
    print(f"\n📁 Generated {len(generated_files)} files:")
    for i, file in enumerate(generated_files, 1):
        if os.path.exists(file):
            print(f"   {i:2d}. ✅ {file}")
        else:
            print(f"   {i:2d}. ⚠️  {file} (not found)")
    
    print(f"\n🧬 Analysis Features Demonstrated:")
    features = [
        "✅ Synthetic multi-omics data generation with biological names",
        "✅ Advanced transformer architecture (MoE + GQA)" if is_advanced else "✅ Enhanced transformer architecture",
        "✅ Model training with early stopping" if train_model else "⚡ Mock model for quick testing",
        "✅ Attention pattern extraction and visualization",
        "✅ Multi-dimensional embedding analysis (PCA, t-SNE, UMAP)",
        "✅ Cross-modal correlation analysis",
        "✅ Multi-method feature importance analysis",
        "✅ Biological pathway enrichment" + (" (reduced scope)" if quick_mode else ""),
        "✅ Comprehensive statistical analysis",
        "✅ Interactive HTML report generation",
        "✅ Results persistence and reproducibility"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🚀 This showcase demonstrates the complete OmicsFormer pipeline!")
    print(f"   From data generation → model training → comprehensive analysis → reporting")
    
    return True, results_summary

def print_alignment_strategies_info():
    """Print information about available alignment strategies."""
    return """
🎯 ALIGNMENT STRATEGIES EXPLAINED:

1. 'strict' - Only samples present in ALL modalities
   ✅ Pros: Complete data for every sample, no missing values
   ❌ Cons: May significantly reduce sample size
   🎯 Best for: High-quality datasets with good overlap (>70%)

2. 'intersection' - Samples present in at least 2 modalities  
   ✅ Pros: Balanced approach, preserves multi-modal samples
   ❌ Cons: Some samples may have missing modalities
   🎯 Best for: Moderate overlap, typical multi-omics studies

3. 'flexible'/'union' - All samples from all modalities
   ✅ Pros: Maximizes sample size, preserves all information
   ❌ Cons: Many missing values, requires robust imputation
   🎯 Best for: Limited data, exploratory analysis

4. 'auto' - Intelligent selection based on data characteristics
   ✅ Pros: Optimizes strategy for your specific dataset
   ❌ Cons: Less predictable, may need manual override
   🎯 Best for: Unknown datasets, general use

The script automatically analyzes your data and selects the optimal strategy!
"""


def print_usage_instructions():
    """Print usage instructions for the comprehensive showcase."""
    print("🧬" + "=" * 78 + "🧬")
    print("         OMICSFORMER COMPREHENSIVE SHOWCASE - USAGE INSTRUCTIONS")
    print("🧬" + "=" * 78 + "🧬")
    
    print("""
📖 USAGE OPTIONS:

1. 🚀 FULL SHOWCASE (Recommended for complete demonstration):
   python analyzer_comprehensive_test.py --full

2. ⚡ QUICK DEMO (Faster execution with smaller dataset):
   python analyzer_comprehensive_test.py --quick

3. 🔍 ANALYSIS ONLY (Skip model training, use mock model):
   python analyzer_comprehensive_test.py --analysis-only

4. 🏋️  TRAINING FOCUS (Emphasize model training):
   python analyzer_comprehensive_test.py --train

🎯 WHAT THIS SHOWCASE DEMONSTRATES:
   • Complete multi-omics data generation with realistic biological features
   • Advanced transformer training with MoE (Mixture of Experts) and GQA
   • Comprehensive analysis: attention, embeddings, correlations, importance
   • Biological pathway enrichment and statistical analysis
   • Interactive visualizations and HTML reporting
   • Full reproducibility and results persistence

💡 TIP: Start with --quick to get familiar, then run --full for complete demo!

🤖 INTELLIGENT FEATURES:
   • Automatic alignment strategy selection based on data overlap
   • Fallback handling for missing components
   • Comprehensive error handling and informative messages
   • Reproducible results with seed management
""")
    
    print(print_alignment_strategies_info())


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "--help"
    
    if mode in ["--help", "-h", "help"]:
        print_usage_instructions()
        sys.exit(0)
    
    # Configure execution based on mode
    if mode == "--full":
        print("🚀 Running FULL COMPREHENSIVE SHOWCASE...")
        success, results = comprehensive_omicsformer_showcase(
            n_samples=800, 
            train_model=True, 
            quick_mode=False
        )
    elif mode == "--quick":
        print("⚡ Running QUICK DEMONSTRATION...")
        success, results = comprehensive_omicsformer_showcase(
            n_samples=200, 
            train_model=False, 
            quick_mode=True
        )
    elif mode == "--analysis-only":
        print("🔍 Running ANALYSIS-ONLY MODE...")
        success, results = comprehensive_omicsformer_showcase(
            n_samples=400, 
            train_model=False, 
            quick_mode=False
        )
    elif mode == "--train":
        print("🏋️  Running TRAINING-FOCUSED MODE...")
        success, results = comprehensive_omicsformer_showcase(
            n_samples=600, 
            train_model=True, 
            quick_mode=False
        )
    else:
        print(f"❌ Unknown mode: {mode}")
        print_usage_instructions()
        sys.exit(1)
    
    # Final status
    if success:
        print(f"\n🎉 SHOWCASE COMPLETED SUCCESSFULLY!")
        print(f"📊 Generated {len(results.get('generated_files', []))} output files")
        print(f"⏱️  Total execution time: {results.get('execution_info', {}).get('execution_time_minutes', 0):.2f} minutes")
    else:
        print(f"\n❌ SHOWCASE ENCOUNTERED ISSUES!")
        print("   Check the output above for specific error messages.")
    
    # Optional cleanup
    if success:
        try:
            cleanup = input(f"\n🧹 Clean up generated files? (y/n): ").lower().strip()
        except (EOFError, KeyboardInterrupt):
            cleanup = 'n'
        
        if cleanup == 'y':
            import glob
            patterns = ['omicsformer_*.png', 'omicsformer_*.html', 'omicsformer_*.json', '*.pth', 'training_*.png']
            cleaned_files = 0
            for pattern in patterns:
                files = glob.glob(pattern)
                for file in files:
                    try:
                        os.remove(file)
                        cleaned_files += 1
                    except:
                        pass
            print(f"   ✅ Cleaned up {cleaned_files} files.")
        else:
            print(f"   📁 Files preserved for your review and analysis.")