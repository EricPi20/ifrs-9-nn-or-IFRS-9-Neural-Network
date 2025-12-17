"""
Tests for MetricsCalculator

Comprehensive unit tests for model evaluation metrics calculation.
"""

import pytest
import numpy as np
from app.services.metrics import (
    MetricsCalculator,
    DiscriminationMetrics,
    CalibrationMetrics,
    ClassificationMetrics,
    ConfusionMatrixMetrics,
    CaptureRateMetrics,
    LiftMetrics,
    CompleteMetrics
)


# === TEST FIXTURES ===

@pytest.fixture
def calculator():
    """Create MetricsCalculator instance."""
    return MetricsCalculator(n_deciles=10)


@pytest.fixture
def perfect_model_data():
    """Perfect model: all predictions match labels exactly."""
    np.random.seed(42)
    n = 1000
    y_true = np.array([0] * 700 + [1] * 300)
    # Perfect model: high scores for bads, low scores for goods
    y_pred_proba = np.concatenate([
        np.random.uniform(0.0, 0.3, 700),  # Goods get low scores
        np.random.uniform(0.7, 1.0, 300)  # Bads get high scores
    ])
    # Shuffle to test ranking
    indices = np.random.permutation(n)
    return y_true[indices], y_pred_proba[indices]


@pytest.fixture
def random_model_data():
    """Random model: predictions are random."""
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)
    y_pred_proba = np.random.uniform(0, 1, n)
    return y_true, y_pred_proba


@pytest.fixture
def all_zeros_data():
    """Edge case: all predictions are 0."""
    n = 100
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred_proba = np.zeros(n)
    return y_true, y_pred_proba


@pytest.fixture
def all_ones_data():
    """Edge case: all predictions are 1."""
    n = 100
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred_proba = np.ones(n)
    return y_true, y_pred_proba


@pytest.fixture
def all_zeros_labels():
    """Edge case: all labels are 0."""
    n = 100
    y_true = np.zeros(n)
    y_pred_proba = np.random.uniform(0, 1, n)
    return y_true, y_pred_proba


@pytest.fixture
def all_ones_labels():
    """Edge case: all labels are 1."""
    n = 100
    y_true = np.ones(n)
    y_pred_proba = np.random.uniform(0, 1, n)
    return y_true, y_pred_proba


# === TEST PERFECT MODEL ===

def test_perfect_model_auc(calculator, perfect_model_data):
    """Test perfect model has AUC = 1.0."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert metrics.discrimination.auc_roc == pytest.approx(1.0, abs=0.01), \
        "Perfect model should have AUC = 1.0"


def test_perfect_model_gini(calculator, perfect_model_data):
    """Test perfect model has Gini = 1.0."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert metrics.discrimination.gini_ar == pytest.approx(1.0, abs=0.01), \
        "Perfect model should have Gini = 1.0"


# === TEST RANDOM MODEL ===

def test_random_model_auc(calculator, random_model_data):
    """Test random model has AUC ≈ 0.5."""
    y_true, y_pred_proba = random_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert metrics.discrimination.auc_roc == pytest.approx(0.5, abs=0.1), \
        f"Random model should have AUC ≈ 0.5, got {metrics.discrimination.auc_roc}"


def test_random_model_gini(calculator, random_model_data):
    """Test random model has Gini ≈ 0."""
    y_true, y_pred_proba = random_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert metrics.discrimination.gini_ar == pytest.approx(0.0, abs=0.1), \
        f"Random model should have Gini ≈ 0, got {metrics.discrimination.gini_ar}"


# === TEST GINI = 2*AUC - 1 IDENTITY ===

def test_gini_auc_identity(calculator, perfect_model_data):
    """Test Gini = 2*AUC - 1 identity."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    expected_gini = 2 * metrics.discrimination.auc_roc - 1
    assert metrics.discrimination.gini_ar == pytest.approx(expected_gini, abs=1e-6), \
        f"Gini should equal 2*AUC - 1. Got Gini={metrics.discrimination.gini_ar}, " \
        f"2*AUC-1={expected_gini}"


def test_gini_auc_identity_random(calculator, random_model_data):
    """Test Gini = 2*AUC - 1 identity for random model."""
    y_true, y_pred_proba = random_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    expected_gini = 2 * metrics.discrimination.auc_roc - 1
    assert metrics.discrimination.gini_ar == pytest.approx(expected_gini, abs=1e-6), \
        f"Gini should equal 2*AUC - 1. Got Gini={metrics.discrimination.gini_ar}, " \
        f"2*AUC-1={expected_gini}"


# === TEST KS STATISTIC ===

def test_ks_bounds(calculator, perfect_model_data):
    """Test KS is between 0 and 1."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert 0 <= metrics.discrimination.ks_statistic <= 1, \
        f"KS statistic should be between 0 and 1, got {metrics.discrimination.ks_statistic}"


def test_ks_bounds_random(calculator, random_model_data):
    """Test KS is between 0 and 1 for random model."""
    y_true, y_pred_proba = random_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert 0 <= metrics.discrimination.ks_statistic <= 1, \
        f"KS statistic should be between 0 and 1, got {metrics.discrimination.ks_statistic}"


def test_ks_decile_range(calculator, perfect_model_data):
    """Test KS decile is between 1 and 10."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    assert 1 <= metrics.discrimination.ks_decile <= 10, \
        f"KS decile should be between 1 and 10, got {metrics.discrimination.ks_decile}"


# === TEST CAPTURE RATES ===

def test_capture_rates_monotonic(calculator, perfect_model_data):
    """Test capture rates are monotonically increasing."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    rates = [
        metrics.capture_rates.bad_rate_top_5pct,
        metrics.capture_rates.bad_rate_top_10pct,
        metrics.capture_rates.bad_rate_top_20pct,
        metrics.capture_rates.bad_rate_top_30pct
    ]
    
    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i + 1], \
            f"Capture rates should be monotonically increasing. " \
            f"Got {rates[i]} > {rates[i + 1]}"


def test_capture_rates_bounds(calculator, perfect_model_data):
    """Test capture rates are between 0 and 1."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    assert 0 <= metrics.capture_rates.bad_rate_top_5pct <= 1
    assert 0 <= metrics.capture_rates.bad_rate_top_10pct <= 1
    assert 0 <= metrics.capture_rates.bad_rate_top_20pct <= 1
    assert 0 <= metrics.capture_rates.bad_rate_top_30pct <= 1


# === TEST DECILE TABLE ===

def test_decile_table_rows(calculator, perfect_model_data):
    """Test decile table has 10 rows."""
    y_true, y_pred_proba = perfect_model_data
    table = calculator.calculate_decile_table(y_true, y_pred_proba)
    assert len(table) == 10, f"Decile table should have 10 rows, got {len(table)}"


def test_decile_table_structure(calculator, perfect_model_data):
    """Test decile table has correct structure."""
    y_true, y_pred_proba = perfect_model_data
    table = calculator.calculate_decile_table(y_true, y_pred_proba)
    
    required_keys = ['decile', 'count', 'bad_count', 'bad_rate', 
                     'cum_bad_pct', 'cum_good_pct', 'ks', 'lift', 
                     'min_prob', 'max_prob']
    
    for row in table:
        for key in required_keys:
            assert key in row, f"Decile table row missing key: {key}"


def test_decile_table_decile_numbers(calculator, perfect_model_data):
    """Test decile table has decile numbers 1-10."""
    y_true, y_pred_proba = perfect_model_data
    table = calculator.calculate_decile_table(y_true, y_pred_proba)
    
    deciles = [row['decile'] for row in table]
    assert deciles == list(range(1, 11)), \
        f"Decile numbers should be 1-10, got {deciles}"


# === TEST EDGE CASES ===

def test_all_zeros_predictions(calculator, all_zeros_data):
    """Test edge case: all predictions are 0."""
    y_true, y_pred_proba = all_zeros_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Should not crash
    assert isinstance(metrics, CompleteMetrics)
    assert metrics.discrimination.auc_roc >= 0
    assert metrics.discrimination.ks_statistic >= 0


def test_all_ones_predictions(calculator, all_ones_data):
    """Test edge case: all predictions are 1."""
    y_true, y_pred_proba = all_ones_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Should not crash
    assert isinstance(metrics, CompleteMetrics)
    assert metrics.discrimination.auc_roc >= 0
    assert metrics.discrimination.ks_statistic >= 0


def test_all_zeros_labels(calculator, all_zeros_labels):
    """Test edge case: all labels are 0."""
    y_true, y_pred_proba = all_zeros_labels
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Should not crash
    assert isinstance(metrics, CompleteMetrics)
    # When all labels are 0, capture rates should be 0
    assert metrics.capture_rates.bad_rate_top_10pct == 0


def test_all_ones_labels(calculator, all_ones_labels):
    """Test edge case: all labels are 1."""
    y_true, y_pred_proba = all_ones_labels
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Should not crash
    assert isinstance(metrics, CompleteMetrics)
    # When all labels are 1, capture rates should be 1.0 (or close)
    assert metrics.capture_rates.bad_rate_top_10pct == pytest.approx(0.1, abs=0.01)


# === TEST RETURN TYPES ===

def test_return_types(calculator, perfect_model_data):
    """Test all return types match dataclass fields."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Discrimination metrics
    assert isinstance(metrics.discrimination, DiscriminationMetrics)
    assert isinstance(metrics.discrimination.auc_roc, float)
    assert isinstance(metrics.discrimination.gini_ar, float)
    assert isinstance(metrics.discrimination.ks_statistic, float)
    assert isinstance(metrics.discrimination.ks_decile, int)
    assert isinstance(metrics.discrimination.c_statistic, float)
    
    # Calibration metrics
    assert isinstance(metrics.calibration, CalibrationMetrics)
    assert isinstance(metrics.calibration.log_loss, float)
    assert isinstance(metrics.calibration.brier_score, float)
    
    # Classification metrics
    assert isinstance(metrics.classification, ClassificationMetrics)
    assert isinstance(metrics.classification.accuracy, float)
    assert isinstance(metrics.classification.precision, float)
    assert isinstance(metrics.classification.recall, float)
    assert isinstance(metrics.classification.f1_score, float)
    assert isinstance(metrics.classification.specificity, float)
    assert isinstance(metrics.classification.balanced_accuracy, float)
    
    # Confusion matrix
    assert isinstance(metrics.confusion_matrix, ConfusionMatrixMetrics)
    assert isinstance(metrics.confusion_matrix.true_positives, int)
    assert isinstance(metrics.confusion_matrix.true_negatives, int)
    assert isinstance(metrics.confusion_matrix.false_positives, int)
    assert isinstance(metrics.confusion_matrix.false_negatives, int)
    assert isinstance(metrics.confusion_matrix.total, int)
    
    # Capture rates
    assert isinstance(metrics.capture_rates, CaptureRateMetrics)
    assert isinstance(metrics.capture_rates.bad_rate_top_5pct, float)
    assert isinstance(metrics.capture_rates.bad_rate_top_10pct, float)
    assert isinstance(metrics.capture_rates.bad_rate_top_20pct, float)
    assert isinstance(metrics.capture_rates.bad_rate_top_30pct, float)
    
    # Lift metrics
    assert isinstance(metrics.lift, LiftMetrics)
    assert isinstance(metrics.lift.lift_top_decile, float)
    assert isinstance(metrics.lift.lift_top_2_deciles, float)
    assert isinstance(metrics.lift.lift_top_3_deciles, float)
    assert isinstance(metrics.lift.cumulative_lift, list)
    assert len(metrics.lift.cumulative_lift) == 10
    assert all(isinstance(x, float) for x in metrics.lift.cumulative_lift)


# === TEST DIVISION BY ZERO PROTECTION ===

def test_division_by_zero_capture_rates(calculator, all_zeros_labels):
    """Test division by zero protection in capture rates."""
    y_true, y_pred_proba = all_zeros_labels
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # When total_bads = 0, all capture rates should be 0
    assert metrics.capture_rates.bad_rate_top_5pct == 0
    assert metrics.capture_rates.bad_rate_top_10pct == 0
    assert metrics.capture_rates.bad_rate_top_20pct == 0
    assert metrics.capture_rates.bad_rate_top_30pct == 0


def test_division_by_zero_lift(calculator, all_zeros_labels):
    """Test division by zero protection in lift calculation."""
    y_true, y_pred_proba = all_zeros_labels
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # When overall_bad_rate = 0, lift should be 0
    assert metrics.lift.lift_top_decile == 0
    assert metrics.lift.lift_top_2_deciles == 0
    assert metrics.lift.lift_top_3_deciles == 0
    assert all(x == 0 for x in metrics.lift.cumulative_lift)


def test_division_by_zero_classification(calculator):
    """Test division by zero protection in classification metrics."""
    # Case: no true positives or negatives
    y_true = np.array([0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3])
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    # Should not crash
    assert isinstance(metrics.classification.specificity, float)
    assert isinstance(metrics.classification.recall, float)
    assert isinstance(metrics.classification.balanced_accuracy, float)


# === TEST C_STATISTIC ===

def test_c_statistic_equals_auc(calculator, perfect_model_data):
    """Test C-statistic equals AUC for binary classification."""
    y_true, y_pred_proba = perfect_model_data
    metrics = calculator.calculate_all(y_true, y_pred_proba)
    
    assert metrics.discrimination.c_statistic == pytest.approx(
        metrics.discrimination.auc_roc, abs=1e-6
    ), "C-statistic should equal AUC for binary classification"

