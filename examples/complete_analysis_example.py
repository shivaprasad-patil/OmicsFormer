"""
OmicsFormer Example: Complete Multi-Omics Analysis Pipeline

This example demonstrates the full capabilities of OmicsFormer including:
- Synthetic multi-omics data generation
- Advanced transformer training with MoE and GQA
- Comprehensive analysis and visualization
- Model interpretation and biomarker discovery
"""

import omicsformer as of
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    print("🧬 OmicsFormer Complete Analysis Example")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    of.set_random_seeds(42)
    device = of.get_device()
    print(f"Using device: {device}")
    
    # Step 1: Generate synthetic multi-omics data
    print("\n📊 Step 1: Generating synthetic multi-omics data...")
    
    modality_data, labels = of.create_synthetic_multiomics_data(
        n_samples=800,
        n_features_per_modality={
            'genomics': 1000,      # DNA mutations, SNPs
            'transcriptomics': 500, # Gene expression
            'proteomics': 200,     # Protein abundance
            'metabolomics': 150    # Metabolite levels
        },
        n_classes=3,  # Three disease subtypes
        missing_rate=0.2,  # 20% missing data per modality
        correlation_strength=0.4  # Moderate cross-modal correlation
    )
    
    print(f"✅ Generated data for {len(labels)} samples across {len(modality_data)} modalities")
    for mod, df in modality_data.items():
        print(f"   - {mod}: {df.shape[0]} samples × {df.shape[1]} features")
    
    # Step 2: Create flexible dataset with intersection alignment
    print("\n🔧 Step 2: Creating flexible multi-omics dataset...")
    
    dataset = of.FlexibleMultiOmicsDataset(
        modality_data=modality_data,
        labels=labels,
        alignment='intersection',  # Samples present in ≥2 modalities
        missing_value_strategy='mean',
        normalize=True
    )
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"   Alignment strategy: {dataset.alignment}")
    print(f"   Feature dimensions: {dataset.feature_dims}")
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Step 3: Initialize advanced transformer model
    print("\n🤖 Step 3: Initializing AdvancedMultiOmicsTransformer...")
    
    model = of.AdvancedMultiOmicsTransformer(
        input_dims=dataset.feature_dims,
        embed_dim=256,
        num_heads=8,
        num_layers=4, 
        num_classes=3,
        num_experts=4,  # MoE with 4 experts
        dropout=0.2,
        use_moe=True,   # Enable Mixture of Experts
        use_gqa=True,   # Enable Grouped Query Attention
        load_balance_weight=0.01
    )
    
    # Print model summary
    of.print_model_summary(model)
    
    # Step 4: Train the model
    print("\n🏋️ Step 4: Training the model...")
    
    trainer = of.MultiOmicsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_wandb=False  # Set to True if you want W&B logging
    )
    
    # Train with early stopping
    history = trainer.fit(
        num_epochs=30,
        early_stopping_patience=8,
        load_balance_weight=0.01,
        save_best_model=True,
        model_save_path='best_omicsformer_model.pth'
    )
    
    print("✅ Training completed!")
    
    # Plot training history
    print("\n📈 Plotting training history...")
    trainer.plot_training_history(history, save_path='training_history.png')
    
    # Step 5: Comprehensive model analysis
    print("\n🔍 Step 5: Comprehensive model analysis...")
    
    analyzer = of.MultiOmicsAnalyzer(model, device=device)
    
    # Extract attention patterns
    print("   Extracting attention patterns...")
    attention_patterns = analyzer.extract_attention_patterns(
        val_loader, max_samples=100
    )
    
    # Visualize attention heatmap
    if attention_patterns:
        print("   Creating attention heatmap...")
        fig = analyzer.visualize_attention_heatmap(
            attention_key='main',
            figsize=(10, 8),
            save_path='attention_patterns.png'
        )
        plt.close(fig)
    
    # Extract embeddings for visualization
    print("   Extracting embeddings...")
    embeddings = analyzer.extract_embeddings(
        val_loader,
        embedding_type='pooled'
    )
    
    # Create UMAP visualization
    if embeddings:
        print("   Creating UMAP visualization...")
        fig = analyzer.plot_embedding_visualization(
            embedding_type='pooled',
            method='umap',
            color_by='label',
            figsize=(10, 8),
            save_path='embeddings_umap.png'
        )
        plt.close(fig)
    
    # Compute cross-modal correlations
    print("   Computing cross-modal correlations...")
    try:
        # Compute correlations from input data directly
        import numpy as np
        from scipy.stats import pearsonr
        
        # Collect data from validation loader
        modality_data = {mod: [] for mod in dataset.modality_data.keys()}
        
        for batch in val_loader:
            for mod in modality_data.keys():
                # Take mean across features for each sample
                mod_data = batch[mod].mean(dim=1).numpy()
                modality_data[mod].extend(mod_data)
        
        # Convert to arrays
        for mod in modality_data:
            modality_data[mod] = np.array(modality_data[mod])
        
        # Compute correlation matrix
        modalities = list(modality_data.keys())
        correlations_matrix = np.zeros((len(modalities), len(modalities)))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if len(modality_data[mod1]) > 0 and len(modality_data[mod2]) > 0:
                    corr, _ = pearsonr(modality_data[mod1], modality_data[mod2])
                    correlations_matrix[i, j] = corr
                else:
                    correlations_matrix[i, j] = 0.0
        
        # Create DataFrame
        import pandas as pd
        correlations = pd.DataFrame(
            correlations_matrix,
            index=modalities,
            columns=modalities
        )
        
        print("   Cross-modal correlations:")
        print(correlations.round(3))
        
        # Plot correlation heatmap
        if not correlations.empty and correlations.shape[0] > 0:
            fig = analyzer.plot_correlation_heatmap(
                correlations,
                figsize=(8, 6),
                save_path='cross_modal_correlations.png'
            )
            plt.close(fig)
            print("   ✅ Cross-modal correlation heatmap saved")
        else:
            print("   ⚠️  Cross-modal correlations are empty - skipping heatmap")
            
    except Exception as e:
        print(f"   ⚠️  Cross-modal correlation analysis failed: {str(e)}")
        print("   Continuing with other analyses...")
        correlations = pd.DataFrame()  # Empty DataFrame for results
    
    # Feature importance analysis
    print("   Analyzing feature importance...")
    try:
        importance = analyzer.analyze_feature_importance(
            val_loader,
            method='attention'  # Use attention-based importance
        )
        
        if importance:
            print("   Feature importance computed!")
            for key, values in importance.items():
                print(f"     {key}: shape {values.shape}")
    except Exception as e:
        print(f"   Feature importance analysis skipped: {e}")
    
    # Step 6: Model evaluation
    print("\n📊 Step 6: Model evaluation...")
    
    test_metrics = {}  # Initialize with empty dict
    try:
        from omicsformer.training import evaluate_model
        
        # Ensure model is on correct device
        model = model.to(device)
        
        test_metrics = evaluate_model(model, val_loader, device=device)
        
        print("   Test Performance:")
        for metric, value in test_metrics.items():
            print(f"     {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"   ⚠️  Model evaluation failed: {str(e)}")
        print("   Continuing with report generation...")
        test_metrics = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Step 7: Generate comprehensive report
    print("\n📋 Step 7: Generating analysis report...")
    
    try:
        report_html = analyzer.generate_analysis_report(
            save_path='omicsformer_analysis_report.html'
        )
        print("   ✅ Analysis report saved as 'omicsformer_analysis_report.html'")
    except Exception as e:
        print(f"   Report generation failed: {e}")
    
    # Step 8: Save results
    print("\n💾 Step 8: Saving results...")
    
    results = {
        'training_history': history,
        'test_metrics': test_metrics,
        'cross_modal_correlations': correlations.to_dict(),
        'model_config': {
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'num_experts': 4,
            'use_moe': True,
            'use_gqa': True
        },
        'dataset_info': dataset.get_modality_info()
    }
    
    import json
    try:
        with open('omicsformer_results.json', 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in results.items():
                if key == 'cross_modal_correlations' and hasattr(value, 'to_dict'):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        print("   ✅ Results saved as 'omicsformer_results.json'")
    except Exception as e:
        print(f"   ⚠️ Results saving failed: {str(e)}")
    
    # Summary
    print("\n🎉 Analysis Complete!")
    print("=" * 50)
    print("Generated files:")
    print("  - best_omicsformer_model.pth (trained model)")
    print("  - training_history.png (training curves)")
    print("  - attention_patterns.png (attention heatmap)")
    print("  - embeddings_umap.png (UMAP visualization)")
    print("  - cross_modal_correlations.png (correlation heatmap)")
    print("  - omicsformer_analysis_report.html (comprehensive report)")
    print("  - omicsformer_results.json (numerical results)")
    
    print(f"\nFinal model performance:")
    print(f"  Accuracy: {test_metrics.get('accuracy', 0):.1%}")
    print(f"  F1-Score: {test_metrics.get('f1', 0):.3f}")
    if 'auc' in test_metrics:
        print(f"  AUC: {test_metrics['auc']:.3f}")
    
    print("\n🧬 OmicsFormer analysis pipeline completed successfully!")


if __name__ == "__main__":
    main()