"""
Quick Results Checker for AES Experiments
==========================================
Use this to check your experiment results and extract values for dissertation tables.

Usage:
    python check_results.py results/bert_cpu
    python check_results.py results/bilstm_cpu results/bert_cpu results/roberta_cpu
"""

import json
import os
import sys

def check_experiment_results(results_dir):
    """Quick checker for experiment results"""
    results_path = os.path.join(results_dir, 'results.json')
    
    if not os.path.exists(results_path):
        print(f"❌ Results not found in {results_dir}")
        print("   Experiment may still be running or failed...")
        print(f"   Check if file exists: {os.path.abspath(results_path)}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"RESULTS FROM: {results_dir}")
    print(f"{'='*70}\n")
    
    # Test metrics
    if 'test_metrics' in results:
        print("📊 TEST METRICS (for Table 4.3):")
        for score_col, metrics in results['test_metrics'].items():
            print(f"\n  Score: {score_col}")
            print(f"    QWK:           {metrics['qwk']:.4f}")
            print(f"    MSE:           {metrics['mse']:.4f}")
            print(f"    MAE:           {metrics['mae']:.4f}")
            print(f"    Pearson r:     {metrics['pearson_r']:.4f}")
            print(f"    Exact Agr:     {metrics['exact_agreement']:.2%}")
            print(f"    Adjacent Agr:  {metrics['adjacent_agreement']:.2%}")
    
    # Fairness
    if 'fairness_stats' in results and results['fairness_stats']:
        print("\n⚖️  FAIRNESS METRICS (for Table 4.7):")
        for score_col, stats in results['fairness_stats'].items():
            print(f"\n  Score: {score_col}")
            print(f"    QWK Range:     {stats['qwk_range']:.4f}")
            print(f"    QWK Std Dev:   {stats['qwk_std']:.4f}")
    else:
        print("\n⚖️  FAIRNESS METRICS: Not available (need --do_fairness flag)")
    
    # Length bias
    if 'length_bias' in results:
        print("\n📏 LENGTH BIAS (for Table 4.9):")
        for score_col, bias in results['length_bias'].items():
            print(f"\n  Score: {score_col}")
            print(f"    Correlation:   {bias['correlation']:.4f}")
            print(f"    P-value:       {bias['p_value']:.4f}")
            print(f"    Interpretation: {bias['interpretation']}")
    
    print(f"\n{'='*70}\n")
    
    return results


def compare_models(results_dirs):
    """Compare results across multiple models"""
    all_results = {}
    
    for results_dir in results_dirs:
        model_name = os.path.basename(results_dir).replace('_cpu', '').replace('_gpu', '')
        all_results[model_name] = check_experiment_results(results_dir)
    
    # Filter out None results
    all_results = {k: v for k, v in all_results.items() if v is not None}
    
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("📊 MODEL COMPARISON (for Table 4.3)")
        print(f"{'='*70}\n")
        
        print(f"{'Model':<15} {'QWK':<8} {'MSE':<8} {'MAE':<8} {'Pearson':<8}")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            if 'test_metrics' in results:
                # Get first score column (usually only one for ASAP)
                score_col = list(results['test_metrics'].keys())[0]
                metrics = results['test_metrics'][score_col]
                
                print(f"{model_name:<15} {metrics['qwk']:<8.4f} {metrics['mse']:<8.4f} "
                      f"{metrics['mae']:<8.4f} {metrics['pearson_r']:<8.4f}")
        
        print("\n" + "="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo results directory specified. Checking default locations...\n")
        
        # Try to find results directories
        possible_dirs = [
            'results/bert_cpu',
            'results/bilstm_cpu', 
            'results/roberta_cpu',
            'artifacts'
        ]
        
        found_dirs = [d for d in possible_dirs if os.path.exists(os.path.join(d, 'results.json'))]
        
        if found_dirs:
            print(f"Found {len(found_dirs)} completed experiment(s):")
            for d in found_dirs:
                print(f"  - {d}")
            print()
            
            if len(found_dirs) == 1:
                check_experiment_results(found_dirs[0])
            else:
                compare_models(found_dirs)
        else:
            print("❌ No completed experiments found in default locations.")
            print("   Run with: python check_results.py <results_directory>")
    
    elif len(sys.argv) == 2:
        # Single directory
        check_experiment_results(sys.argv[1])
    
    else:
        # Multiple directories - compare
        compare_models(sys.argv[1:])


if __name__ == "__main__":
    main()