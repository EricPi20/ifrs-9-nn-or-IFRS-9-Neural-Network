"""
Tests for SegmentedDataProcessor

Comprehensive unit tests for data validation and processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

# Import directly from module file to avoid __init__.py dependencies
module_path = Path(__file__).parent.parent / "app" / "services" / "data_processor.py"
spec = importlib.util.spec_from_file_location("data_processor", module_path)
data_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processor_module)
SegmentedDataProcessor = data_processor_module.SegmentedDataProcessor


# === TEST FIXTURES ===

@pytest.fixture
def sample_data():
    """Create sample DataFrame with segments and WoE features."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'account_id': [f'ACC{i}' for i in range(100)],
        'segment': ['CONSUMER']*60 + ['SME']*40,
        'woe_feature1': np.random.choice([-0.5, 0.0, 0.5], 100),
        'woe_feature2': np.random.choice([-1.0, 0.0, 1.0], 100),
        'target': [0]*85 + [1]*15  # 15% bad rate
    })


@pytest.fixture
def processor():
    """Create SegmentedDataProcessor instance."""
    return SegmentedDataProcessor()


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame with required columns."""
    return pd.DataFrame(columns=['account_id', 'segment', 'woe_feature1', 'target'])


@pytest.fixture
def bin_mapping_data():
    """Create bin mapping DataFrame."""
    return pd.DataFrame({
        'feature': ['woe_feature1', 'woe_feature1', 'woe_feature1', 
                    'woe_feature2', 'woe_feature2', 'woe_feature2'],
        'bin_label': ['Low Risk', 'Medium Risk', 'High Risk',
                      'Low Risk', 'Medium Risk', 'High Risk'],
        'woe_value': [-0.5, 0.0, 0.5, -1.0, 0.0, 1.0]
    })


# === VALIDATION TESTS ===

def test_validate_valid_data(processor, sample_data):
    """Test validation with valid data returns empty error list."""
    errors = processor.validate(sample_data)
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_missing_target_column(processor, sample_data):
    """Test validation fails when target column is missing."""
    df = sample_data.drop(columns=['target'])
    errors = processor.validate(df)
    assert len(errors) > 0, "Expected errors for missing target column"
    assert any('target' in err.lower() for err in errors), "Error should mention target column"


def test_validate_missing_id_column(processor, sample_data):
    """Test validation fails when ID column is missing."""
    df = sample_data.drop(columns=['account_id'])
    errors = processor.validate(df)
    assert len(errors) > 0, "Expected errors for missing ID column"
    assert any('account_id' in err.lower() for err in errors), "Error should mention account_id"


def test_validate_non_binary_target(processor, sample_data):
    """Test validation fails when target is not binary."""
    df = sample_data.copy()
    df['target'] = [0, 1, 2] * 33 + [1]  # Contains value 2
    errors = processor.validate(df)
    assert len(errors) > 0, "Expected errors for non-binary target"
    assert any('binary' in err.lower() for err in errors), "Error should mention binary requirement"


def test_validate_duplicate_ids(processor, sample_data):
    """Test validation fails when IDs are duplicated."""
    df = sample_data.copy()
    df.loc[0, 'account_id'] = 'ACC1'  # Create duplicate
    errors = processor.validate(df)
    assert len(errors) > 0, "Expected errors for duplicate IDs"
    assert any('duplicate' in err.lower() for err in errors), "Error should mention duplicates"


def test_validate_non_numeric_woe(processor, sample_data):
    """Test validation fails when WoE features are not numeric."""
    df = sample_data.copy()
    df['woe_feature1'] = ['a', 'b', 'c'] * 33 + ['d']  # Non-numeric
    errors = processor.validate(df)
    assert len(errors) > 0, "Expected errors for non-numeric WoE"
    assert any('numeric' in err.lower() for err in errors), "Error should mention numeric requirement"


def test_validate_empty_dataframe(processor, empty_dataframe):
    """Test validation handles empty DataFrame."""
    errors = processor.validate(empty_dataframe)
    assert isinstance(errors, list), "Should return list of errors"
    # Empty DataFrame should have errors for missing columns/data


def test_validate_extreme_woe_values(processor, sample_data):
    """Test validation warns about extreme WoE values."""
    df = sample_data.copy()
    df['woe_feature1'] = 10.0  # Extreme value > 5
    errors = processor.validate(df)
    assert any('extreme' in err.lower() or 'warning' in err.lower() for err in errors), \
        "Should warn about extreme WoE values"


def test_validate_all_null_feature(processor, sample_data):
    """Test validation fails when feature is entirely null."""
    df = sample_data.copy()
    df['woe_feature1'] = np.nan
    errors = processor.validate(df)
    assert any('null' in err.lower() for err in errors), "Should error on all-null feature"


# === SEGMENT METHOD TESTS ===

def test_get_segments(processor, sample_data):
    """Test get_segments returns correct segment list."""
    segments = processor.get_segments(sample_data)
    assert segments == ['CONSUMER', 'SME'], f"Expected ['CONSUMER', 'SME'], got {segments}"


def test_get_segments_all_when_no_segment_column(processor, sample_data):
    """Test get_segments returns ['ALL'] when no segment column."""
    df = sample_data.drop(columns=['segment'])
    segments = processor.get_segments(df)
    assert segments == ['ALL'], f"Expected ['ALL'], got {segments}"


def test_get_segment_stats(processor, sample_data):
    """Test get_segment_stats returns correct counts and bad rates."""
    stats = processor.get_segment_stats(sample_data)
    
    assert len(stats) == 2, f"Expected 2 segments, got {len(stats)}"
    
    # Find CONSUMER segment
    consumer_stats = next(s for s in stats if s['segment'] == 'CONSUMER')
    assert consumer_stats['count'] == 60, f"Expected 60 CONSUMER records, got {consumer_stats['count']}"
    
    # Find SME segment
    sme_stats = next(s for s in stats if s['segment'] == 'SME')
    assert sme_stats['count'] == 40, f"Expected 40 SME records, got {sme_stats['count']}"
    
    # Check bad_rate is calculated correctly
    for stat in stats:
        expected_bad_rate = stat['bad_count'] / stat['count']
        assert abs(stat['bad_rate'] - expected_bad_rate) < 0.01, \
            f"Bad rate mismatch for {stat['segment']}"


def test_get_segment_stats_all_when_no_segment(processor, sample_data):
    """Test get_segment_stats returns ALL stats when no segment column."""
    df = sample_data.drop(columns=['segment'])
    stats = processor.get_segment_stats(df)
    
    assert len(stats) == 1, f"Expected 1 stat entry, got {len(stats)}"
    assert stats[0]['segment'] == 'ALL', "Should return ALL segment"
    assert stats[0]['count'] == 100, f"Expected 100 total records, got {stats[0]['count']}"


def test_filter_segment(processor, sample_data):
    """Test filter_segment correctly filters data."""
    consumer_df = processor.filter_segment(sample_data, 'CONSUMER')
    assert len(consumer_df) == 60, f"Expected 60 CONSUMER records, got {len(consumer_df)}"
    assert all(consumer_df['segment'] == 'CONSUMER'), "All records should be CONSUMER"
    
    sme_df = processor.filter_segment(sample_data, 'SME')
    assert len(sme_df) == 40, f"Expected 40 SME records, got {len(sme_df)}"
    assert all(sme_df['segment'] == 'SME'), "All records should be SME"


def test_filter_segment_all(processor, sample_data):
    """Test filter_segment with 'ALL' returns full dataset."""
    all_df = processor.filter_segment(sample_data, 'ALL')
    assert len(all_df) == len(sample_data), "ALL should return full dataset"
    assert all_df.equals(sample_data), "ALL should return identical data"


def test_filter_segment_all_when_no_segment_column(processor, sample_data):
    """Test filter_segment returns full dataset when no segment column."""
    df = sample_data.drop(columns=['segment'])
    filtered = processor.filter_segment(df, 'ALL')
    assert len(filtered) == len(df), "Should return full dataset when no segment column"


# === FEATURE ANALYSIS TESTS ===

def test_analyze_features_returns_all_features(processor, sample_data):
    """Test analyze_features returns all features."""
    analysis = processor.analyze_features(sample_data)
    feature_names = [f['name'] for f in analysis]
    assert 'woe_feature1' in feature_names, "Should include woe_feature1"
    assert 'woe_feature2' in feature_names, "Should include woe_feature2"
    assert len(feature_names) == 2, f"Expected 2 features, got {len(feature_names)}"


def test_analyze_features_sorted_by_correlation(processor, sample_data):
    """Test analyze_features results sorted by absolute correlation."""
    analysis = processor.analyze_features(sample_data)
    
    # Check sorting: absolute correlations should be descending
    abs_corrs = [abs(f['target_correlation']) for f in analysis]
    assert abs_corrs == sorted(abs_corrs, reverse=True), \
        "Features should be sorted by absolute correlation (descending)"


def test_analyze_features_num_bins_matches_unique_values(processor, sample_data):
    """Test num_bins matches unique WoE values."""
    analysis = processor.analyze_features(sample_data)
    
    for feat_analysis in analysis:
        num_bins = feat_analysis['num_bins']
        bins_list = feat_analysis['bins']
        
        assert num_bins == len(bins_list), \
            f"num_bins ({num_bins}) should match bins list length ({len(bins_list)})"
        
        # Check unique WoE values
        unique_woe = set(b['woe_value'] for b in bins_list)
        assert len(unique_woe) == num_bins, \
            f"Number of unique WoE values should match num_bins"


def test_analyze_features_segment_filtering(processor, sample_data):
    """Test analyze_features correctly filters by segment."""
    consumer_analysis = processor.analyze_features(sample_data, segment='CONSUMER')
    all_analysis = processor.analyze_features(sample_data, segment='ALL')
    
    # Should have same number of features
    assert len(consumer_analysis) == len(all_analysis), \
        "Should have same number of features regardless of segment"
    
    # But statistics might differ
    assert isinstance(consumer_analysis[0]['mean_woe'], float), \
        "mean_woe should be a float"


# === DATA PREPARATION TESTS ===

def test_prepare_data_returns_correct_shapes(processor, sample_data):
    """Test prepare_data returns correct shapes."""
    X, y, feature_names = processor.prepare_data(sample_data)
    
    assert X.shape[0] == len(sample_data), \
        f"X should have {len(sample_data)} rows, got {X.shape[0]}"
    assert X.shape[1] == len(feature_names), \
        f"X should have {len(feature_names)} columns, got {X.shape[1]}"
    assert y.shape[0] == len(sample_data), \
        f"y should have {len(sample_data)} elements, got {y.shape[0]}"
    assert len(feature_names) == 2, \
        f"Expected 2 feature names, got {len(feature_names)}"


def test_prepare_data_selected_features(processor, sample_data):
    """Test prepare_data filters correctly with selected_features."""
    X, y, feature_names = processor.prepare_data(
        sample_data, 
        selected_features=['woe_feature1']
    )
    
    assert len(feature_names) == 1, \
        f"Expected 1 feature, got {len(feature_names)}"
    assert feature_names[0] == 'woe_feature1', \
        f"Expected 'woe_feature1', got {feature_names[0]}"
    assert X.shape[1] == 1, \
        f"X should have 1 column, got {X.shape[1]}"


def test_prepare_data_data_types_float32(processor, sample_data):
    """Test prepare_data returns float32 data types."""
    X, y, feature_names = processor.prepare_data(sample_data)
    
    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert y.dtype == np.float32, f"y should be float32, got {y.dtype}"


def test_prepare_data_segment_filtering(processor, sample_data):
    """Test prepare_data correctly filters by segment."""
    X_all, y_all, _ = processor.prepare_data(sample_data, segment='ALL')
    X_consumer, y_consumer, _ = processor.prepare_data(sample_data, segment='CONSUMER')
    
    assert X_consumer.shape[0] == 60, \
        f"CONSUMER segment should have 60 rows, got {X_consumer.shape[0]}"
    assert X_all.shape[0] == 100, \
        f"ALL segment should have 100 rows, got {X_all.shape[0]}"


# === TRAIN/TEST SPLIT TESTS ===

def test_split_train_test_proportions(processor, sample_data):
    """Test split proportions are correct (~70/30)."""
    X, y, _ = processor.prepare_data(sample_data)
    splits = processor.split_train_test(X, y, test_size=0.30, random_state=42)
    
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']
    
    total = len(X_train) + len(X_test)
    train_prop = len(X_train) / total
    test_prop = len(X_test) / total
    
    assert abs(train_prop - 0.70) < 0.05, \
        f"Expected ~70% train, got {train_prop:.2%}"
    assert abs(test_prop - 0.30) < 0.05, \
        f"Expected ~30% test, got {test_prop:.2%}"


def test_split_train_test_stratification(processor, sample_data):
    """Test stratification maintains bad rate."""
    X, y, _ = processor.prepare_data(sample_data)
    splits = processor.split_train_test(X, y, test_size=0.30, random_state=42)
    
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']
    
    overall_bad_rate = y.mean()
    train_bad_rate = y_train.mean()
    test_bad_rate = y_test.mean()
    
    # Stratified split should maintain similar bad rates
    assert abs(train_bad_rate - overall_bad_rate) < 0.05, \
        f"Train bad rate ({train_bad_rate:.3f}) should be close to overall ({overall_bad_rate:.3f})"
    assert abs(test_bad_rate - overall_bad_rate) < 0.05, \
        f"Test bad rate ({test_bad_rate:.3f}) should be close to overall ({overall_bad_rate:.3f})"


def test_split_train_test_reproducible(processor, sample_data):
    """Test random state produces reproducible splits."""
    X, y, _ = processor.prepare_data(sample_data)
    
    splits1 = processor.split_train_test(X, y, test_size=0.30, random_state=42)
    splits2 = processor.split_train_test(X, y, test_size=0.30, random_state=42)
    
    X_train1, y_train1 = splits1['train']
    X_test1, y_test1 = splits1['test']
    X_train2, y_train2 = splits2['train']
    X_test2, y_test2 = splits2['test']
    
    # Check arrays are identical
    assert np.array_equal(X_train1, X_train2), "Train X should be identical with same random_state"
    assert np.array_equal(y_train1, y_train2), "Train y should be identical with same random_state"
    assert np.array_equal(X_test1, X_test2), "Test X should be identical with same random_state"
    assert np.array_equal(y_test1, y_test2), "Test y should be identical with same random_state"


def test_split_train_test_different_random_states(processor, sample_data):
    """Test different random states produce different splits."""
    X, y, _ = processor.prepare_data(sample_data)
    
    splits1 = processor.split_train_test(X, y, test_size=0.30, random_state=42)
    splits2 = processor.split_train_test(X, y, test_size=0.30, random_state=123)
    
    X_train1, _ = splits1['train']
    X_train2, _ = splits2['train']
    
    # With different random states, splits should be different
    assert not np.array_equal(X_train1, X_train2), \
        "Different random states should produce different splits"


# === BIN MAPPING TESTS ===

def test_load_bin_mapping_stores_correctly(processor, bin_mapping_data):
    """Test load_bin_mapping stores mapping correctly."""
    processor.load_bin_mapping(bin_mapping_data)
    
    assert processor._bin_mapping is not None, "Bin mapping should be stored"
    assert 'woe_feature1' in processor._bin_mapping, "Should have woe_feature1 mapping"
    assert 'woe_feature2' in processor._bin_mapping, "Should have woe_feature2 mapping"
    
    # Check structure
    feature1_mapping = processor._bin_mapping['woe_feature1']
    assert len(feature1_mapping) == 3, f"Expected 3 bins for woe_feature1, got {len(feature1_mapping)}"
    assert all('bin_label' in bin_info for bin_info in feature1_mapping), \
        "Each bin should have bin_label"
    assert all('woe_value' in bin_info for bin_info in feature1_mapping), \
        "Each bin should have woe_value"


def test_get_bin_label_returns_correct_labels(processor, bin_mapping_data, sample_data):
    """Test _get_bin_label returns correct labels from mapping."""
    processor.load_bin_mapping(bin_mapping_data)
    
    # Test with known WoE value
    label = processor._get_bin_label('woe_feature1', -0.5)
    assert label == 'Low Risk', f"Expected 'Low Risk', got '{label}'"
    
    label = processor._get_bin_label('woe_feature1', 0.0)
    assert label == 'Medium Risk', f"Expected 'Medium Risk', got '{label}'"
    
    label = processor._get_bin_label('woe_feature1', 0.5)
    assert label == 'High Risk', f"Expected 'High Risk', got '{label}'"


def test_get_bin_label_default_when_no_mapping(processor):
    """Test _get_bin_label generates default labels when no mapping."""
    # No mapping loaded
    label = processor._get_bin_label('woe_feature1', 0.8)
    assert 'Low Risk' in label, f"Should generate default label, got '{label}'"
    assert 'WoE' in label, "Default label should include WoE value"
    
    label = processor._get_bin_label('woe_feature1', -0.8)
    assert 'High Risk' in label, f"Should generate default label for negative WoE, got '{label}'"


def test_get_bin_label_tolerance(processor, bin_mapping_data):
    """Test _get_bin_label handles small floating point differences."""
    processor.load_bin_mapping(bin_mapping_data)
    
    # Test with value slightly different from stored (-0.5)
    label = processor._get_bin_label('woe_feature1', -0.5001)
    assert label == 'Low Risk', "Should match within tolerance"
    
    label = processor._get_bin_label('woe_feature1', -0.4999)
    assert label == 'Low Risk', "Should match within tolerance"


def test_analyze_features_uses_bin_mapping(processor, bin_mapping_data, sample_data):
    """Test analyze_features uses bin mapping when available."""
    processor.load_bin_mapping(bin_mapping_data)
    analysis = processor.analyze_features(sample_data)
    
    # Find woe_feature1 in analysis
    feat1_analysis = next(f for f in analysis if f['name'] == 'woe_feature1')
    
    # Check that bins have labels from mapping
    bin_labels = [b['bin_label'] for b in feat1_analysis['bins']]
    assert 'Low Risk' in bin_labels or 'Medium Risk' in bin_labels or 'High Risk' in bin_labels, \
        "Should use labels from mapping"
