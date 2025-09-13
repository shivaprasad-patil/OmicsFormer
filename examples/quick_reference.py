#!/usr/bin/env python3
"""
Quick demo showing which strategies work with different sample scenarios.
"""

def main():
    print("🧬 OmicsFormer Sample Alignment Strategies - Quick Reference")
    print("="*70)
    print()
    
    scenarios = [
        ("Same samples across all modalities", "✅", "✅", "✅", "✅"),
        ("Partial overlap (some shared)", "✅", "✅", "✅", "✅"), 
        ("No overlap (different samples)", "❌", "✅", "❌", "✅"),
        ("Real-world mixed patterns", "✅", "✅", "✅", "✅")
    ]
    
    print(f"{'Scenario':<35} {'Strict':<8} {'Flexible':<10} {'Intersection':<12} {'Union'}")
    print("-" * 70)
    
    for scenario, strict, flexible, intersection, union in scenarios:
        print(f"{scenario:<35} {strict:<8} {flexible:<10} {intersection:<12} {union}")
    
    print()
    print("🎯 ANSWER TO YOUR QUESTION:")
    print("YES! Both FLEXIBLE and UNION strategies work even when")
    print("data comes from completely different samples!")
    print()
    print("💡 HOW IT WORKS:")
    print("• Missing modality data → filled with zeros/missing tokens")
    print("• Model learns from available data per sample")
    print("• Cross-modal patterns learned through shared embedding space")
    print("• Each sample contributes what data it has available")

if __name__ == "__main__":
    main()