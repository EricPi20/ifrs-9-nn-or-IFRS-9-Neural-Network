"""
RIFT Data Validation Script
============================
Validates that frontend displays match actual CSV data calculations.

Run: python test_upload_validation.py <path_to_csv>
Or: pytest test_upload_validation.py::validate_csv_data -v
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def validate_csv_data(csv_path: str) -> dict:
    """
    Load CSV and compute all statistics that should match frontend display.
    """
    print(f"\n{'='*60}")
    print(f" RIFT Data Validation")
    print(f" File: {csv_path}")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    results = {
        'file': csv_path,
        'validation_passed': True,
        'segment_stats': [],
        'feature_stats': [],
        'errors': []
    }
    
    # ==================== SEGMENT VALIDATION ====================
    print("=" * 40)
    print(" SEGMENT STATISTICS")
    print("=" * 40)
    
    # Expected values from frontend
    expected_segments = {
        'ALL': {'records': 10000, 'bads': 1269, 'bad_rate': 0.127},
        'CONSUMER': {'records': 8000, 'bads': 1009, 'bad_rate': 0.126},
        'ENTERPRISE': {'records': 500, 'bads': 50, 'bad_rate': 0.10},
        'SME': {'records': 1500, 'bads': 210, 'bad_rate': 0.14},
    }
    
    # Calculate actual values
    target_col = settings.DEFAULT_TARGET_COLUMN
    segment_col = settings.DEFAULT_SEGMENT_COLUMN
    
    # ALL segment
    all_records = len(df)
    all_bads = df[target_col].sum()
    all_bad_rate = df[target_col].mean()
    
    print(f"\n{'Segment':<12} {'Records':>10} {'Bads':>10} {'Bad Rate':>12} {'Status':<10}")
    print("-" * 60)
    
    # Validate ALL
    status = "✓ PASS" if all_records == expected_segments['ALL']['records'] else "✗ FAIL"
    print(f"{'ALL':<12} {all_records:>10,} {all_bads:>10,} {all_bad_rate:>11.2%} {status:<10}")
    
    results['segment_stats'].append({
        'segment': 'ALL',
        'expected_records': expected_segments['ALL']['records'],
        'actual_records': all_records,
        'expected_bads': expected_segments['ALL']['bads'],
        'actual_bads': int(all_bads),
        'expected_bad_rate': expected_segments['ALL']['bad_rate'],
        'actual_bad_rate': round(all_bad_rate, 4),
        'match': all_records == expected_segments['ALL']['records']
    })
    
    if all_records != expected_segments['ALL']['records']:
        results['validation_passed'] = False
        results['errors'].append(f"ALL records mismatch: expected {expected_segments['ALL']['records']}, got {all_records}")
    
    if int(all_bads) != expected_segments['ALL']['bads']:
        results['validation_passed'] = False
        results['errors'].append(f"ALL bads mismatch: expected {expected_segments['ALL']['bads']}, got {int(all_bads)}")
    
    # Validate each segment
    for seg_name in ['CONSUMER', 'ENTERPRISE', 'SME']:
        if segment_col in df.columns:
            seg_df = df[df[segment_col] == seg_name]
            seg_records = len(seg_df)
            seg_bads = seg_df[target_col].sum()
            seg_bad_rate = seg_df[target_col].mean() if len(seg_df) > 0 else 0
            
            expected = expected_segments.get(seg_name, {})
            records_match = seg_records == expected.get('records', 0)
            bads_match = int(seg_bads) == expected.get('bads', 0)
            
            status = "✓ PASS" if records_match and bads_match else "✗ FAIL"
            print(f"{seg_name:<12} {seg_records:>10,} {seg_bads:>10,} {seg_bad_rate:>11.2%} {status:<10}")
            
            results['segment_stats'].append({
                'segment': seg_name,
                'expected_records': expected.get('records', 0),
                'actual_records': seg_records,
                'expected_bads': expected.get('bads', 0),
                'actual_bads': int(seg_bads),
                'expected_bad_rate': expected.get('bad_rate', 0),
                'actual_bad_rate': round(seg_bad_rate, 4),
                'match': records_match and bads_match
            })
            
            if not records_match or not bads_match:
                results['validation_passed'] = False
                results['errors'].append(f"{seg_name} mismatch: records={seg_records} (expected {expected.get('records', 0)}), bads={int(seg_bads)} (expected {expected.get('bads', 0)})")
    
    # ==================== FEATURE VALIDATION ====================
    print(f"\n{'='*40}")
    print(" FEATURE STATISTICS")
    print("=" * 40)
    
    # Expected values from frontend
    expected_features = {
        'payment_history': {'bins': 5, 'min': -60, 'max': 75, 'correlation': -0.114},
        'credit_utilization': {'bins': 5, 'min': -55, 'max': 65, 'correlation': -0.059},
        'account_tenure': {'bins': 5, 'min': -50, 'max': 55, 'correlation': -0.033},
        'monthly_income': {'bins': 5, 'min': -45, 'max': 60, 'correlation': -0.025},
        'recent_inquiries': {'bins': 4, 'min': -45, 'max': 45, 'correlation': -0.024},
        'num_credit_lines': {'bins': 4, 'min': -40, 'max': 40, 'correlation': -0.009},
    }
    
    # Identify feature columns (exclude target, segment, account_id)
    exclude_cols = {target_col, segment_col, settings.DEFAULT_ID_COLUMN}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\n{'Feature':<20} {'Bins':>6} {'Min':>8} {'Max':>8} {'Corr':>10} {'Status':<10}")
    print("-" * 70)
    
    for feat in feature_cols:
        unique_vals = df[feat].dropna().unique()
        num_bins = len(unique_vals)
        min_val = df[feat].min()
        max_val = df[feat].max()
        correlation = df[feat].corr(df[target_col])
        
        expected = expected_features.get(feat, {})
        
        bins_match = num_bins == expected.get('bins', 0) if expected else False
        min_match = abs(min_val - expected.get('min', 0)) < 0.01 if expected else False
        max_match = abs(max_val - expected.get('max', 0)) < 0.01 if expected else False
        corr_match = abs(correlation - expected.get('correlation', 0)) < 0.01 if expected else False  # Allow small tolerance
        
        all_match = bins_match and min_match and max_match and corr_match if expected else False
        status = "✓ PASS" if all_match else "✗ FAIL" if expected else "? SKIP"
        
        print(f"{feat:<20} {num_bins:>6} {min_val:>8.0f} {max_val:>8.0f} {correlation:>10.3f} {status:<10}")
        
        results['feature_stats'].append({
            'feature': feat,
            'expected_bins': expected.get('bins', 0) if expected else None,
            'actual_bins': num_bins,
            'expected_min': expected.get('min', 0) if expected else None,
            'actual_min': int(min_val),
            'expected_max': expected.get('max', 0) if expected else None,
            'actual_max': int(max_val),
            'expected_correlation': expected.get('correlation', 0) if expected else None,
            'actual_correlation': round(correlation, 3),
            'match': all_match if expected else None
        })
        
        if expected and not all_match:
            results['validation_passed'] = False
            if not bins_match:
                results['errors'].append(f"{feat} bins mismatch: expected {expected.get('bins')}, got {num_bins}")
            if not min_match:
                results['errors'].append(f"{feat} min mismatch: expected {expected.get('min')}, got {min_val:.0f}")
            if not max_match:
                results['errors'].append(f"{feat} max mismatch: expected {expected.get('max')}, got {max_val:.0f}")
            if not corr_match:
                results['errors'].append(f"{feat} correlation mismatch: expected {expected.get('correlation')}, got {correlation:.3f}")
    
    # ==================== UNIQUE VALUES CHECK ====================
    print(f"\n{'='*40}")
    print(" UNIQUE BIN VALUES PER FEATURE")
    print("=" * 40)
    
    for feat in feature_cols:
        unique_vals = sorted(df[feat].dropna().unique().tolist())
        print(f"\n{feat}:")
        print(f"  Values: {unique_vals}")
    
    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print(" VALIDATION SUMMARY")
    print("=" * 60)
    
    if results['validation_passed']:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("   Frontend displays match actual CSV data.")
    else:
        print("\n❌ SOME VALIDATIONS FAILED!")
        print("\nErrors found:")
        for error in results['errors']:
            print(f"   - {error}")
    
    # Save results to JSON (convert numpy types to native Python types)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results_serializable = convert_to_native(results)
    output_path = Path(csv_path).parent / (Path(csv_path).stem + "_validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")
    
    return results


def compare_with_frontend(frontend_response: dict, csv_path: str) -> dict:
    """
    Compare frontend API response with actual CSV calculations.
    
    Args:
        frontend_response: The JSON response from /api/upload endpoint
        csv_path: Path to the CSV file
    
    Returns:
        Comparison results
    """
    df = pd.read_csv(csv_path)
    target_col = settings.DEFAULT_TARGET_COLUMN
    
    comparison = {'matches': [], 'mismatches': []}
    
    # Compare segment stats
    for seg_stat in frontend_response.get('segment_stats', []):
        seg_name = seg_stat['segment']
        
        if seg_name == 'ALL':
            actual_count = len(df)
            actual_bads = df[target_col].sum()
        else:
            seg_df = df[df['segment'] == seg_name]
            actual_count = len(seg_df)
            actual_bads = seg_df[target_col].sum()
        
        if seg_stat['count'] == actual_count and seg_stat['bad_count'] == actual_bads:
            comparison['matches'].append(f"{seg_name}: count and bad_count match")
        else:
            comparison['mismatches'].append({
                'segment': seg_name,
                'frontend_count': seg_stat['count'],
                'actual_count': actual_count,
                'frontend_bads': seg_stat['bad_count'],
                'actual_bads': int(actual_bads)
            })
    
    return comparison


if __name__ == "__main__":
    # Default path or command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try common locations
        possible_paths = [
            "rift_sample_data.csv",
            "data/uploads/rift_sample_data.csv",
            "../data/uploads/rift_sample_data.csv",
            "nn-scorecard/backend/data/uploads/rift_sample_data.csv",
            str(Path(__file__).parent.parent / "data" / "uploads" / "rift_sample_data.csv"),
        ]
        csv_path = None
        for p in possible_paths:
            if Path(p).exists():
                csv_path = p
                break
        
        if not csv_path:
            print("Usage: python test_upload_validation.py <path_to_csv>")
            print("Please provide the path to rift_sample_data.csv")
            print("\nTried the following paths:")
            for p in possible_paths:
                print(f"  - {p}")
            sys.exit(1)
    
    results = validate_csv_data(csv_path)
    
    # Exit with error code if validation failed
    sys.exit(0 if results['validation_passed'] else 1)

