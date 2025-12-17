#!/usr/bin/env python3
"""
Analyze specific training run and validate AR/AUC calculations
"""

import json
import numpy as np
from pathlib import Path

def analyze_training_run(metadata_file):
    """Analyze a specific training run's metrics."""
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print(f" TRAINING RUN ANALYSIS: {data['model_id'][:8]}")
    print("="*80)
    print(f"Model: {data.get('model_name', 'N/A')}")
    print(f"Segment: {data.get('segment', 'N/A')}")
    print(f"Created: {data.get('created_at', 'N/A')}")
    print()
    
    # Architecture
    arch = data.get('architecture', {})
    print("ARCHITECTURE:")
    print(f"  Hidden Layers: {arch.get('hidden_layers', [])}")
    print(f"  Activation: {arch.get('activation_function', 'N/A')}")
    print(f"  Dropout: {arch.get('dropout_rate', 0)}")
    print()
    
    # Loss function
    loss = data.get('loss_function', {})
    print("LOSS FUNCTION:")
    print(f"  Type: {loss.get('loss_type', 'N/A')}")
    if loss.get('loss_alpha'):
        print(f"  Alpha: {loss.get('loss_alpha')}")
    if loss.get('auc_gamma'):
        print(f"  AUC Gamma: {loss.get('auc_gamma')}")
    print()
    
    # Training history
    history = data.get('training_history', {})
    epochs = history.get('epochs', [])
    
    if not epochs:
        print("No training history found!")
        return
    
    print(f"TRAINING HISTORY: {len(epochs)} epochs")
    print()
    
    # Validate AR = 2*AUC - 1 for each epoch
    print("VALIDATING AR = 2*AUC - 1 IDENTITY:")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train AUC':>10} | {'Train AR':>10} | {'2*AUC-1':>10} | {'Diff':>10} | {'Status':>6}")
    print("-" * 80)
    
    max_train_diff = 0
    max_test_diff = 0
    all_valid = True
    
    train_aucs = []
    test_aucs = []
    train_ars = []
    test_ars = []
    
    for epoch_data in epochs:
        epoch = epoch_data['epoch']
        train_auc = epoch_data['train_auc']
        train_ar = epoch_data['train_ar']
        test_auc = epoch_data['test_auc']
        test_ar = epoch_data['test_ar']
        
        # Calculate expected AR
        expected_train_ar = 2 * train_auc - 1
        expected_test_ar = 2 * test_auc - 1
        
        # Differences
        train_diff = abs(train_ar - expected_train_ar)
        test_diff = abs(test_ar - expected_test_ar)
        
        max_train_diff = max(max_train_diff, train_diff)
        max_test_diff = max(max_test_diff, test_diff)
        
        # Check validity (tolerance: 1e-6)
        train_valid = train_diff < 1e-6
        test_valid = test_diff < 1e-6
        
        if not (train_valid and test_valid):
            all_valid = False
        
        status = "✓" if (train_valid and test_valid) else "✗"
        
        # Print only every 5 epochs or if there's an issue
        if epoch % 5 == 0 or epoch == len(epochs) or not (train_valid and test_valid):
            print(f"{epoch:5d} | {train_auc:10.6f} | {train_ar:10.6f} | {expected_train_ar:10.6f} | {train_diff:10.2e} | {status:>6}")
        
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        train_ars.append(train_ar)
        test_ars.append(test_ar)
    
    print("-" * 80)
    print(f"Max Train Difference: {max_train_diff:.2e}")
    print(f"Max Test Difference: {max_test_diff:.2e}")
    
    if all_valid and max_train_diff < 1e-6 and max_test_diff < 1e-6:
        print("✅ PASS: AR = 2*AUC - 1 identity holds for all epochs")
    else:
        print("❌ FAIL: Identity violations detected")
    print()
    
    # Statistical summary
    print("METRICS SUMMARY:")
    print("-" * 80)
    print(f"{'Metric':<20} | {'Initial':>10} | {'Final':>10} | {'Best':>10} | {'Mean':>10}")
    print("-" * 80)
    
    print(f"{'Train AUC':<20} | {train_aucs[0]:10.4f} | {train_aucs[-1]:10.4f} | {max(train_aucs):10.4f} | {np.mean(train_aucs):10.4f}")
    print(f"{'Test AUC':<20} | {test_aucs[0]:10.4f} | {test_aucs[-1]:10.4f} | {max(test_aucs):10.4f} | {np.mean(test_aucs):10.4f}")
    print(f"{'Train AR':<20} | {train_ars[0]:10.4f} | {train_ars[-1]:10.4f} | {max(train_ars):10.4f} | {np.mean(train_ars):10.4f}")
    print(f"{'Test AR':<20} | {test_ars[0]:10.4f} | {test_ars[-1]:10.4f} | {max(test_ars):10.4f} | {np.mean(test_ars):10.4f}")
    print()
    
    # Convergence analysis
    print("CONVERGENCE ANALYSIS:")
    print("-" * 80)
    
    train_auc_improvement = train_aucs[-1] - train_aucs[0]
    test_auc_improvement = test_aucs[-1] - test_aucs[0]
    train_ar_improvement = train_ars[-1] - train_ars[0]
    test_ar_improvement = test_ars[-1] - test_ars[0]
    
    print(f"Train AUC Improvement: {train_auc_improvement:+.4f} ({train_auc_improvement/train_aucs[0]*100:+.2f}%)")
    print(f"Test AUC Improvement:  {test_auc_improvement:+.4f} ({test_auc_improvement/test_aucs[0]*100:+.2f}%)")
    print(f"Train AR Improvement:  {train_ar_improvement:+.4f} ({train_ar_improvement/train_ars[0]*100:+.2f}%)")
    print(f"Test AR Improvement:   {test_ar_improvement:+.4f} ({test_ar_improvement/test_ars[0]*100:+.2f}%)")
    print()
    
    # Check for overfitting
    train_test_gap = np.array(train_aucs) - np.array(test_aucs)
    print("OVERFITTING CHECK:")
    print("-" * 80)
    print(f"Initial Train-Test Gap (AUC): {train_test_gap[0]:.4f}")
    print(f"Final Train-Test Gap (AUC):   {train_test_gap[-1]:.4f}")
    print(f"Average Train-Test Gap (AUC): {np.mean(train_test_gap):.4f}")
    
    if train_test_gap[-1] < 0.02:
        print("✅ Model shows good generalization (gap < 0.02)")
    elif train_test_gap[-1] < 0.05:
        print("⚠️  Model shows slight overfitting (0.02 < gap < 0.05)")
    else:
        print("❌ Model shows significant overfitting (gap > 0.05)")
    print()
    
    # Final metrics
    final_metrics = data.get('final_metrics', {}).get('discrimination', {})
    if final_metrics:
        print("FINAL METRICS (from metadata):")
        print("-" * 80)
        final_auc = final_metrics.get('auc_roc', 0)
        final_ar = final_metrics.get('gini_ar', 0)
        final_ks = final_metrics.get('ks_statistic', 0)
        
        expected_final_ar = 2 * final_auc - 1
        final_diff = abs(final_ar - expected_final_ar)
        
        print(f"AUC: {final_auc:.4f}")
        print(f"AR (Gini): {final_ar:.4f}")
        print(f"Expected AR (2*AUC-1): {expected_final_ar:.4f}")
        print(f"Difference: {final_diff:.2e}")
        print(f"KS Statistic: {final_ks:.4f}")
        
        if final_diff < 1e-6:
            print("✅ Final metrics consistent with AR = 2*AUC - 1")
        else:
            print("❌ Final metrics show identity violation")
        print()
    
    # Best epoch
    best_epoch = history.get('best_epoch', 0)
    best_test_ar = history.get('best_test_ar', 0)
    early_stopping = history.get('early_stopping_triggered', False)
    
    print("TRAINING OUTCOME:")
    print("-" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test AR: {best_test_ar:.4f}")
    print(f"Early Stopping: {'Yes' if early_stopping else 'No'}")
    print(f"Total Training Time: {history.get('total_training_time_seconds', 0):.2f} seconds")
    print()
    
    return {
        'all_valid': all_valid,
        'max_train_diff': max_train_diff,
        'max_test_diff': max_test_diff,
        'final_test_auc': test_aucs[-1],
        'final_test_ar': test_ars[-1],
        'train_test_gap': train_test_gap[-1]
    }


def main():
    # Find most recent metadata file
    models_dir = Path(__file__).parent / "data" / "models"
    metadata_files = sorted(
        models_dir.glob("*_metadata.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not metadata_files:
        print("No training runs found!")
        return 1
    
    # Analyze most recent
    most_recent = metadata_files[0]
    print(f"\nAnalyzing most recent training run: {most_recent.name}\n")
    
    result = analyze_training_run(most_recent)
    
    print("="*80)
    print(" FINAL VERDICT")
    print("="*80)
    
    if result['all_valid']:
        print("✅ ALL CHECKS PASSED")
        print(f"   - AR = 2*AUC - 1 identity: VALID (max error: {max(result['max_train_diff'], result['max_test_diff']):.2e})")
        print(f"   - Final Test AUC: {result['final_test_auc']:.4f}")
        print(f"   - Final Test AR: {result['final_test_ar']:.4f}")
        print(f"   - Train-Test Gap: {result['train_test_gap']:.4f}")
        print("\n🎉 Training history and AR/AUC calculations are CORRECT!\n")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print(f"   - AR = 2*AUC - 1 identity: VIOLATED (max error: {max(result['max_train_diff'], result['max_test_diff']):.2e})")
        print("\n⚠️  Please review the metrics calculation logic.\n")
        return 1


if __name__ == "__main__":
    exit(main())

