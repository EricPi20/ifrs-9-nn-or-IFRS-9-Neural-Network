#!/usr/bin/env python3
"""
Validation Script: Compare AUC/AR Implementations and Check Training History

This script:
1. Compares sklearn AUC vs manual AUC implementation
2. Validates AR = 2*AUC - 1 identity
3. Analyzes recent training runs for consistency
"""

import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.neural_network import calculate_auc as manual_calculate_auc
from app.services.metrics import MetricsCalculator


def compare_auc_implementations(n_samples=1000, n_tests=10):
    """Compare sklearn AUC vs manual implementation."""
    print("="*70)
    print("TEST 1: Comparing AUC Implementations")
    print("="*70)
    
    differences = []
    
    for i in range(n_tests):
        np.random.seed(i)
        
        # Generate test data with varying AUC levels
        bad_rate = np.random.uniform(0.1, 0.4)
        n_bad = int(n_samples * bad_rate)
        n_good = n_samples - n_bad
        
        # Generate scores with good separation
        separation = np.random.uniform(10, 30)
        good_scores = np.clip(np.random.normal(55 + separation, 15, n_good), 0, 100)
        bad_scores = np.clip(np.random.normal(45 - separation, 15, n_bad), 0, 100)
        
        y_pred = np.concatenate([good_scores, bad_scores]) / 100.0  # Normalize to 0-1
        y_true = np.concatenate([np.zeros(n_good), np.ones(n_bad)])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        y_pred = y_pred[shuffle_idx]
        y_true = y_true[shuffle_idx]
        
        # Calculate with both methods
        sklearn_auc = roc_auc_score(y_true, y_pred)
        manual_auc = manual_calculate_auc(y_true, y_pred)
        
        difference = abs(sklearn_auc - manual_auc)
        differences.append(difference)
        
        status = "✓" if difference < 1e-6 else "⚠"
        print(f"Test {i+1:2d}: sklearn={sklearn_auc:.6f}, manual={manual_auc:.6f}, "
              f"diff={difference:.2e} {status}")
    
    print(f"\nMax difference: {max(differences):.2e}")
    print(f"Mean difference: {np.mean(differences):.2e}")
    
    if max(differences) < 1e-4:
        print("✅ PASS: Both implementations produce nearly identical results")
    else:
        print("❌ FAIL: Significant differences detected")
    
    return max(differences) < 1e-4


def validate_ar_identity():
    """Validate that AR = 2*AUC - 1 always holds."""
    print("\n" + "="*70)
    print("TEST 2: Validating AR = 2*AUC - 1 Identity")
    print("="*70)
    
    calculator = MetricsCalculator()
    test_cases = [
        ("Perfect Model", 0.0, 1.0, 1.0),  # Perfect separation
        ("Random Model", 0.5, 0.5, 0.0),   # Random
        ("Good Model", 0.3, 0.75, 0.5),    # Good performance
        ("Poor Model", 0.7, 0.60, 0.2),    # Poor performance
    ]
    
    all_passed = True
    
    for name, bad_rate, target_auc, expected_ar in test_cases:
        np.random.seed(42)
        n = 1000
        n_bad = int(n * bad_rate)
        n_good = n - n_bad
        
        # Generate data to achieve target AUC
        separation = (target_auc - 0.5) * 60
        good_scores = np.clip(np.random.normal(55 + separation, 15, n_good), 0.01, 0.99)
        bad_scores = np.clip(np.random.normal(45 - separation, 15, n_bad), 0.01, 0.99)
        
        y_pred = np.concatenate([good_scores, bad_scores])
        y_true = np.concatenate([np.zeros(n_good), np.ones(n_bad)])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n)
        y_pred = y_pred[shuffle_idx]
        y_true = y_true[shuffle_idx]
        
        # Calculate metrics
        metrics = calculator.calculate_all(y_true, y_pred)
        auc = metrics.discrimination.auc_roc
        ar = metrics.discrimination.gini_ar
        
        # Verify identity
        calculated_ar = 2 * auc - 1
        identity_holds = abs(ar - calculated_ar) < 1e-6
        
        status = "✓" if identity_holds else "✗"
        print(f"{name:15s}: AUC={auc:.4f}, AR={ar:.4f}, 2*AUC-1={calculated_ar:.4f} {status}")
        
        if not identity_holds:
            all_passed = False
    
    if all_passed:
        print("✅ PASS: AR = 2*AUC - 1 identity holds for all test cases")
    else:
        print("❌ FAIL: Identity violation detected")
    
    return all_passed


def analyze_recent_training_runs(limit=5):
    """Analyze recent training runs for metric consistency."""
    print("\n" + "="*70)
    print("TEST 3: Analyzing Recent Training Runs")
    print("="*70)
    
    models_dir = Path(__file__).parent / "data" / "models"
    
    if not models_dir.exists():
        print("⚠ No training data found")
        return True
    
    # Get all metadata files
    metadata_files = sorted(
        models_dir.glob("*_metadata.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:limit]
    
    if not metadata_files:
        print("⚠ No training runs found")
        return True
    
    print(f"Analyzing {len(metadata_files)} most recent training runs:\n")
    
    all_consistent = True
    
    for i, meta_file in enumerate(metadata_files, 1):
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            job_id = data.get('job_id', meta_file.stem.replace('_metadata', ''))[:8]
            config = data.get('config', {})
            
            # Check if we have metrics
            if 'metrics' in data:
                metrics = data['metrics']
                train_metrics = metrics.get('train', {})
                test_metrics = metrics.get('test', {})
                
                # Get AUC and AR
                train_auc = train_metrics.get('discrimination', {}).get('auc_roc')
                train_ar = train_metrics.get('discrimination', {}).get('gini_ar')
                test_auc = test_metrics.get('discrimination', {}).get('auc_roc')
                test_ar = test_metrics.get('discrimination', {}).get('gini_ar')
                
                if train_auc is not None and train_ar is not None:
                    # Validate identity
                    expected_train_ar = 2 * train_auc - 1
                    train_consistent = abs(train_ar - expected_train_ar) < 1e-4
                    
                    expected_test_ar = 2 * test_auc - 1 if test_auc else None
                    test_consistent = abs(test_ar - expected_test_ar) < 1e-4 if test_ar and test_auc else True
                    
                    status = "✓" if (train_consistent and test_consistent) else "✗"
                    
                    print(f"Run {i} [{job_id}]:")
                    print(f"  Train: AUC={train_auc:.4f}, AR={train_ar:.4f}, "
                          f"2*AUC-1={expected_train_ar:.4f} {status}")
                    
                    if test_auc:
                        test_status = "✓" if test_consistent else "✗"
                        print(f"  Test:  AUC={test_auc:.4f}, AR={test_ar:.4f}, "
                              f"2*AUC-1={expected_test_ar:.4f} {test_status}")
                    
                    if not (train_consistent and test_consistent):
                        all_consistent = False
                        print(f"  ⚠ WARNING: Identity violation detected!")
                else:
                    print(f"Run {i} [{job_id}]: Incomplete metrics")
            else:
                print(f"Run {i} [{job_id}]: No metrics found")
        
        except Exception as e:
            print(f"Run {i}: Error reading {meta_file.name}: {e}")
    
    print()
    if all_consistent:
        print("✅ PASS: All training runs show consistent AR = 2*AUC - 1")
    else:
        print("❌ FAIL: Some training runs show identity violations")
    
    return all_consistent


def check_training_history_consistency():
    """Check if training history tracks metrics correctly."""
    print("\n" + "="*70)
    print("TEST 4: Training History Structure Check")
    print("="*70)
    
    from app.services.trainer import EpochMetrics, TrainingHistory
    
    # Check that EpochMetrics has all required fields
    required_fields = [
        'epoch', 'train_loss', 'test_loss',
        'train_auc', 'test_auc', 
        'train_ar', 'test_ar',
        'train_ks', 'test_ks',
        'learning_rate', 'epoch_time_seconds'
    ]
    
    epoch_fields = EpochMetrics.__dataclass_fields__.keys()
    
    print("Checking EpochMetrics dataclass:")
    all_present = True
    for field in required_fields:
        present = field in epoch_fields
        status = "✓" if present else "✗"
        print(f"  {field:20s}: {status}")
        if not present:
            all_present = False
    
    print("\nChecking TrainingHistory dataclass:")
    history_fields = ['epochs', 'best_epoch', 'best_test_ar', 
                     'total_training_time_seconds', 'early_stopping_triggered']
    
    for field in history_fields:
        present = field in TrainingHistory.__dataclass_fields__.keys()
        status = "✓" if present else "✗"
        print(f"  {field:30s}: {status}")
        if not present:
            all_present = False
    
    if all_present:
        print("\n✅ PASS: Training history structure is complete")
    else:
        print("\n❌ FAIL: Training history missing required fields")
    
    return all_present


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print(" METRICS VALIDATION SUITE")
    print("="*70)
    print()
    
    results = {}
    
    # Run all tests
    results['auc_comparison'] = compare_auc_implementations()
    results['ar_identity'] = validate_ar_identity()
    results['training_runs'] = analyze_recent_training_runs()
    results['history_structure'] = check_training_history_consistency()
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():30s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed! Metrics calculations are correct.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())

