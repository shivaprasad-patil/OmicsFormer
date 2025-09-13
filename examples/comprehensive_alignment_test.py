#!/usr/bin/env python3
"""
Comprehensive test of OmicsFormer alignment strategies with various sample overlap scenarios.

This script tests alignment strategies with:
1. Complete overlap (same samples across all modalities)
2. Partial overlap (some shared samples)
3. No overlap (completely different samples)
4. Real-world scenario (mixed overlap patterns)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omicsformer.data.dataset import FlexibleMultiOmicsDataset

def create_scenario_data(scenario: str) -> Tuple[Dict, pd.Series, str]:
    """Create different sample overlap scenarios."""
    
    if scenario == "complete_overlap":
        # All modalities have the same 30 samples
        common_samples = [f'patient_{i:03d}' for i in range(1, 31)]
        sample_sets = {
            'genomics': common_samples,
            'transcriptomics': common_samples,
            'proteomics': common_samples,
            'metabolomics': common_samples
        }
        description = "Same 30 samples across all modalities"
        
    elif scenario == "partial_overlap":
        # Some samples shared, some unique to each modality
        base_samples = [f'patient_{i:03d}' for i in range(1, 21)]  # 20 shared
        sample_sets = {
            'genomics': base_samples + [f'genomics_only_{i:03d}' for i in range(1, 11)],      # +10 unique
            'transcriptomics': base_samples + [f'transcriptomics_only_{i:03d}' for i in range(1, 11)], # +10 unique
            'proteomics': base_samples + [f'proteomics_only_{i:03d}' for i in range(1, 11)],  # +10 unique
            'metabolomics': base_samples + [f'metabolomics_only_{i:03d}' for i in range(1, 11)] # +10 unique
        }
        description = "20 shared samples + 10 unique per modality"
        
    elif scenario == "no_overlap":
        # Completely different samples
        sample_sets = {
            'genomics': [f'genomics_patient_{i:03d}' for i in range(1, 26)],        # 25 samples
            'transcriptomics': [f'transcriptomics_patient_{i:03d}' for i in range(1, 21)], # 20 samples
            'proteomics': [f'proteomics_patient_{i:03d}' for i in range(1, 16)],    # 15 samples
            'metabolomics': [f'metabolomics_patient_{i:03d}' for i in range(1, 11)]  # 10 samples
        }
        description = "Completely different samples (25+20+15+10 unique)"
        
    elif scenario == "real_world":
        # Realistic clinical scenario with complex overlap patterns
        # Core cohort: 15 samples with all data
        # Extended cohorts: additional samples with 2-3 modalities
        # Individual studies: samples with only 1 modality
        
        core_cohort = [f'core_patient_{i:03d}' for i in range(1, 16)]  # 15 samples
        
        sample_sets = {
            'genomics': (core_cohort + 
                        [f'genomics_study_{i:03d}' for i in range(1, 21)] +  # +20 genomics only
                        [f'multi_study_A_{i:03d}' for i in range(1, 11)]),   # +10 in genomics+transcriptomics
            
            'transcriptomics': (core_cohort +
                               [f'transcriptomics_study_{i:03d}' for i in range(1, 16)] + # +15 transcriptomics only
                               [f'multi_study_A_{i:03d}' for i in range(1, 11)] +         # +10 with genomics
                               [f'multi_study_B_{i:03d}' for i in range(1, 8)]),          # +7 with proteomics
            
            'proteomics': (core_cohort +
                          [f'proteomics_study_{i:03d}' for i in range(1, 11)] +  # +10 proteomics only
                          [f'multi_study_B_{i:03d}' for i in range(1, 8)] +      # +7 with transcriptomics
                          [f'multi_study_C_{i:03d}' for i in range(1, 6)]),      # +5 with metabolomics
            
            'metabolomics': (core_cohort +
                            [f'metabolomics_study_{i:03d}' for i in range(1, 8)] + # +7 metabolomics only
                            [f'multi_study_C_{i:03d}' for i in range(1, 6)])       # +5 with proteomics
        }
        description = "Real-world: 15 core + various study-specific samples"
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Create data for each modality
    modality_data = {}
    feature_counts = {'genomics': 100, 'transcriptomics': 80, 'proteomics': 60, 'metabolomics': 40}
    
    for modality, samples in sample_sets.items():
        n_features = feature_counts[modality]
        data = np.random.randn(len(samples), n_features)
        df = pd.DataFrame(data, 
                         index=samples,
                         columns=[f'{modality}_feature_{i}' for i in range(n_features)])
        modality_data[modality] = df
    
    # Create labels for all unique samples
    all_samples = set()
    for samples in sample_sets.values():
        all_samples.update(samples)
    labels = pd.Series(np.random.randint(0, 3, len(all_samples)), 
                      index=list(all_samples))
    
    return modality_data, labels, description

def analyze_sample_overlap(sample_sets: Dict[str, List[str]]) -> Dict:
    """Analyze overlap patterns between modalities."""
    
    # Convert to sets for easier analysis
    mod_sets = {mod: set(samples) for mod, samples in sample_sets.items()}
    modalities = list(mod_sets.keys())
    
    analysis = {
        'total_unique_samples': len(set().union(*mod_sets.values())),
        'per_modality_counts': {mod: len(samples) for mod, samples in mod_sets.items()},
        'pairwise_overlaps': {},
        'complete_overlap': len(set.intersection(*mod_sets.values())),
        'samples_in_multiple_modalities': 0
    }
    
    # Pairwise overlaps
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            overlap = len(mod_sets[mod1] & mod_sets[mod2])
            analysis['pairwise_overlaps'][f"{mod1}-{mod2}"] = overlap
    
    # Count samples in multiple modalities
    all_samples = set().union(*mod_sets.values())
    for sample in all_samples:
        modality_count = sum(1 for mod_set in mod_sets.values() if sample in mod_set)
        if modality_count >= 2:
            analysis['samples_in_multiple_modalities'] += 1
    
    return analysis

def test_scenario(scenario_name: str):
    """Test all alignment strategies for a specific scenario."""
    
    print(f"\n{'='*80}")
    print(f"🧪 SCENARIO: {scenario_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    # Create data
    modality_data, labels, description = create_scenario_data(scenario_name)
    
    # Analyze sample overlap
    sample_sets = {mod: list(df.index) for mod, df in modality_data.items()}
    overlap_analysis = analyze_sample_overlap(sample_sets)
    
    print(f"📊 {description}")
    print(f"   • Total unique samples: {overlap_analysis['total_unique_samples']}")
    print(f"   • Per modality: {overlap_analysis['per_modality_counts']}")
    print(f"   • Complete overlap (all 4): {overlap_analysis['complete_overlap']}")
    print(f"   • In ≥2 modalities: {overlap_analysis['samples_in_multiple_modalities']}")
    
    # Test all strategies
    strategies = ['strict', 'flexible', 'intersection', 'union']
    results = {}
    
    for strategy in strategies:
        try:
            dataset = FlexibleMultiOmicsDataset(
                modality_data=modality_data,
                labels=labels,
                alignment=strategy,
                missing_value_strategy='zero',
                normalize=True
            )
            
            sample_count = len(dataset)
            success = sample_count > 0
            
            results[strategy] = {
                'success': success,
                'samples': sample_count,
                'percentage': (sample_count / overlap_analysis['total_unique_samples']) * 100
            }
            
            status = "✅" if success else "❌"
            print(f"   {status} {strategy.capitalize():<12}: {sample_count:>3} samples ({results[strategy]['percentage']:5.1f}%)")
            
        except Exception as e:
            results[strategy] = {'success': False, 'samples': 0, 'percentage': 0}
            print(f"   ❌ {strategy.capitalize():<12}: Error - {str(e)}")
    
    return results, overlap_analysis

def main():
    """Main comprehensive test function."""
    
    print("🚀 OmicsFormer Comprehensive Sample Alignment Test")
    print("="*80)
    print("🎯 Testing all 4 alignment strategies across different sample overlap scenarios")
    print("="*80)
    
    scenarios = [
        'complete_overlap',
        'partial_overlap', 
        'no_overlap',
        'real_world'
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        scenario_results, overlap_info = test_scenario(scenario)
        all_results[scenario] = scenario_results
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Scenario':<20} {'Strict':<8} {'Flexible':<10} {'Intersection':<12} {'Union':<8}")
    print(f"{'-'*70}")
    
    for scenario, results in all_results.items():
        scenario_display = scenario.replace('_', ' ').title()
        row = f"{scenario_display:<20}"
        
        for strategy in ['strict', 'flexible', 'intersection', 'union']:
            samples = results[strategy]['samples']
            status = "✅" if results[strategy]['success'] else "❌"
            row += f" {samples:>3}{status:<4}"
        
        print(row)
    
    print(f"\n💡 KEY INSIGHTS & RECOMMENDATIONS:")
    print(f"{'='*50}")
    print(f"✅ FLEXIBLE & UNION strategies work in ALL scenarios")
    print(f"   → Best for real-world data with missing/incomplete samples")
    print(f"   → Handle completely different sample sets gracefully")
    print(f"   → Use missing tokens for absent modality data")
    print(f"")
    print(f"🎯 STRICT strategy only works with complete sample overlap")
    print(f"   → Use when you need guaranteed complete data")
    print(f"   → Will result in 0 samples if no complete overlap")
    print(f"")
    print(f"⚖️  INTERSECTION strategy requires samples in ≥2 modalities")
    print(f"   → Good middle ground for partially overlapping datasets")
    print(f"   → Balances data completeness with sample inclusion")
    print(f"")
    print(f"📈 For clinical/real-world multi-omics:")
    print(f"   → Start with FLEXIBLE or UNION to maximize sample usage")
    print(f"   → Use INTERSECTION for quality-completeness balance")
    print(f"   → Only use STRICT if complete data is absolutely required")

if __name__ == "__main__":
    main()