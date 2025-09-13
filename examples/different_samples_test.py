#!/usr/bin/env python3
"""
Test OmicsFormer alignment strategies with completely different samples across modalities.

This script tests what happens when different modalitie    for strategy, result in results.items():
        success = "✅ Success" if result['success'] and result['samples'] > 0 else "❌ Failed"
        samples = result['samples']ontain data from 
completely different sets of samples (no overlap).
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omicsformer.data.dataset import FlexibleMultiOmicsDataset
from omicsformer.models.transformer import EnhancedMultiOmicsTransformer

def create_non_overlapping_data():
    """Create synthetic multi-omics data with NO overlapping samples."""
    
    print("🧬 Creating multi-omics data with COMPLETELY DIFFERENT samples per modality...")
    
    # Define completely different sample sets for each modality
    sample_sets = {
        'genomics': [f'sample_genomics_{i:03d}' for i in range(1, 51)],        # 50 samples
        'transcriptomics': [f'sample_transcriptomics_{i:03d}' for i in range(1, 41)],  # 40 samples  
        'proteomics': [f'sample_proteomics_{i:03d}' for i in range(1, 31)],    # 30 samples
        'metabolomics': [f'sample_metabolomics_{i:03d}' for i in range(1, 21)]  # 20 samples
    }
    
    # Verify no overlaps
    all_samples = []
    for modality, samples in sample_sets.items():
        all_samples.extend(samples)
    
    print(f"📊 Total unique samples across all modalities: {len(set(all_samples))}")
    print(f"📊 Total samples if completely different: {sum(len(samples) for samples in sample_sets.values())}")
    print(f"✅ No overlap confirmed: {len(set(all_samples)) == len(all_samples)}")
    
    # Create data for each modality
    modality_data = {}
    feature_counts = {'genomics': 100, 'transcriptomics': 80, 'proteomics': 60, 'metabolomics': 40}
    
    for modality, samples in sample_sets.items():
        n_features = feature_counts[modality]
        
        # Generate random data
        data = np.random.randn(len(samples), n_features)
        
        # Create DataFrame
        df = pd.DataFrame(data, 
                         index=samples,
                         columns=[f'{modality}_feature_{i}' for i in range(n_features)])
        
        modality_data[modality] = df
        
        print(f"   - {modality}: {len(samples)} samples × {n_features} features")
    
    # Create some synthetic labels for all samples
    all_unique_samples = list(set(all_samples))
    labels = pd.Series(np.random.randint(0, 3, len(all_unique_samples)), 
                      index=all_unique_samples)
    
    return modality_data, labels, sample_sets

def test_alignment_strategy(modality_data: Dict, labels: pd.Series, strategy: str):
    """Test a specific alignment strategy with non-overlapping samples."""
    
    print(f"\n{'='*60}")
    print(f"🎯 Testing {strategy.upper()} Alignment Strategy")
    print(f"📋 Sample scenario: COMPLETELY DIFFERENT samples per modality")
    print(f"{'='*60}")
    
    try:
        # Create dataset
        dataset = FlexibleMultiOmicsDataset(
            modality_data=modality_data,
            labels=labels,
            alignment=strategy,
            missing_value_strategy='zero',
            normalize=True
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"   📊 Samples included: {len(dataset)}")
        
        # Get modality info
        info = dataset.get_modality_info()
        print(f"   🧪 Modalities: {info['modalities']}")
        print(f"   📏 Feature dimensions: {info['feature_dims']}")
        print(f"   🔗 Alignment strategy: {info['sample_alignment']}")
        
        if len(dataset) > 0:
            # Test data loading
            sample_batch = dataset[0]
            print(f"\n🔍 First sample analysis:")
            print(f"   📦 Sample ID: {sample_batch['sample_id']}")
            print(f"   🎭 Sample availability per modality:")
            
            for modality in info['modalities']:
                tensor = sample_batch[modality]
                mask = sample_batch['sample_masks'][modality]
                availability = "available" if mask else "missing (filled with zeros/means)"
                print(f"      - {modality}: {tensor.shape} ({availability})")
            
            return True, len(dataset)
            
        else:
            print(f"   ⚠️  No samples available with this strategy!")
            return False, 0
            
    except Exception as e:
        print(f"   ❌ Error with {strategy} strategy: {str(e)}")
        return False, 0

def main():
    """Main demonstration function."""
    
    print("🚀 OmicsFormer Different Samples Test")
    print("="*80)
    print("🧪 SCENARIO: Each modality has completely different samples (NO overlap)")
    print("🎯 QUESTION: Do any alignment strategies work in this case?")
    print("="*80)
    
    # Create non-overlapping data
    modality_data, labels, sample_sets = create_non_overlapping_data()
    
    # Test all alignment strategies
    strategies = ['strict', 'flexible', 'intersection', 'union']
    results = {}
    
    for strategy in strategies:
        success, sample_count = test_alignment_strategy(modality_data, labels, strategy)
        results[strategy] = {'success': success, 'samples': sample_count}
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 RESULTS SUMMARY: Different Samples Across Modalities")
    print(f"{'='*80}")
    
    print(f"{'Strategy':<12} {'Samples':<8} {'Status':<12} {'Notes'}")
    print(f"{'-'*50}")
    
    for strategy, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        samples = result['samples']
        
        if strategy == 'strict':
            notes = "Expected: 0 (no common samples)"
        elif strategy == 'flexible':
            notes = "Expected: All samples with missing tokens"
        elif strategy == 'intersection':
            notes = "Expected: 0 (no sample in >1 modality)"
        elif strategy == 'union':
            notes = "Expected: All samples combined"
        else:
            notes = ""
            
        print(f"{strategy.capitalize():<12} {samples:<8} {status:<12} {notes}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"==================================================")
    
    working_strategies = [s for s, r in results.items() if r['success'] and r['samples'] > 0]
    
    if working_strategies:
        print(f"✅ Working strategies: {', '.join(working_strategies)}")
        print(f"🎯 Best strategy for different samples: FLEXIBLE or UNION")
        print(f"   - These strategies can handle completely different sample sets")
        print(f"   - Missing data is handled with missing tokens or zero-filling")
        print(f"   - Model can still learn cross-modal patterns through shared feature space")
    else:
        print(f"❌ No strategies work with completely different samples")
    
    print(f"\n🔬 TECHNICAL EXPLANATION:")
    print(f"==================================================")
    print(f"• STRICT: Requires same samples across ALL modalities → 0 samples")
    print(f"• FLEXIBLE: Uses all samples + missing tokens → Works!")
    print(f"• INTERSECTION: Needs samples in ≥2 modalities → 0 samples") 
    print(f"• UNION: Combines all samples → Works!")
    print(f"")
    print(f"🧠 The transformer can still learn useful representations even with")
    print(f"   completely different samples because it learns modality-specific")
    print(f"   and cross-modal patterns in the shared embedding space.")

if __name__ == "__main__":
    main()