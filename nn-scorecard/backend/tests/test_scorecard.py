"""
Tests for Scorecard Generator

Comprehensive unit tests for ScorecardGenerator class covering:
1. Mock model with known weights
2. Scorecard generation with discrete bin values
3. Points calculation verification: Points = Weight × Input_Value × scale_factor
4. Total score range validation (0-100)
5. Min score corresponds to worst combination
6. Max score corresponds to best combination
7. calculate_score returns correct breakdown
"""

import pytest
import torch
import numpy as np
from app.services.scorecard import ScorecardGenerator, Scorecard
from app.services.nn_scorecard import LinearScorecardNN


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_model_2_features():
    """Create a mock linear model with known weights for 2 features.
    
    Weights: [2.0, -1.5]
    Bias: 0.0
    
    This allows us to verify exact point calculations.
    """
    model = LinearScorecardNN(input_dim=2)
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[2.0, -1.5]])
        model.linear.bias.data = torch.tensor([0.0])
    return model


@pytest.fixture
def discrete_bin_values_2_features():
    """Create discrete bin values for 2 features.
    
    Feature 1: [10.0, 5.0, -5.0, -10.0]  (4 bins)
    Feature 2: [8.0, 4.0, -4.0, -8.0]    (4 bins)
    
    These represent the standardized log odds × -50 values.
    """
    return {
        'feature_1': [10.0, 5.0, -5.0, -10.0],
        'feature_2': [8.0, 4.0, -4.0, -8.0]
    }


@pytest.fixture
def mock_model_3_features():
    """Create a mock linear model with known weights for 3 features.
    
    Weights: [1.0, 0.5, -0.8]
    Bias: 0.0
    """
    model = LinearScorecardNN(input_dim=3)
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[1.0, 0.5, -0.8]])
        model.linear.bias.data = torch.tensor([0.0])
    return model


@pytest.fixture
def discrete_bin_values_3_features():
    """Create discrete bin values for 3 features."""
    return {
        'feature_1': [15.0, 10.0, -10.0, -15.0],
        'feature_2': [12.0, 6.0, -6.0, -12.0],
        'feature_3': [9.0, 4.5, -4.5, -9.0]
    }


@pytest.fixture
def generator():
    """Create a ScorecardGenerator instance with scale_factor=1.0 for testing."""
    return ScorecardGenerator(score_min=0, score_max=100, scale_factor=1.0)


# ============================================================================
# Test 1: Create Mock Model with Known Weights
# ============================================================================

def test_mock_model_creation(mock_model_2_features):
    """Test that mock model is created with correct weights."""
    weights, bias = mock_model_2_features.get_coefficients()
    
    assert len(weights) == 2
    assert weights[0] == pytest.approx(2.0, abs=1e-6)
    assert weights[1] == pytest.approx(-1.5, abs=1e-6)
    assert bias == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Test 2: Generate Scorecard with Discrete Bin Values
# ============================================================================

def test_generate_scorecard_with_discrete_bins(
    mock_model_2_features, 
    discrete_bin_values_2_features, 
    generator
):
    """Test that scorecard is generated with discrete bin values."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    assert isinstance(scorecard, Scorecard)
    assert scorecard.segment == 'TEST'
    assert len(scorecard.features) == 2
    
    # Check feature 1
    feat1 = scorecard.features[0] if scorecard.features[0].feature_name == 'feature_1' else scorecard.features[1]
    assert feat1.feature_name == 'feature_1'
    assert len(feat1.bins) == 4
    assert feat1.weight == pytest.approx(2.0, abs=1e-6)
    
    # Check feature 2
    feat2 = scorecard.features[1] if scorecard.features[1].feature_name == 'feature_2' else scorecard.features[0]
    assert feat2.feature_name == 'feature_2'
    assert len(feat2.bins) == 4
    assert feat2.weight == pytest.approx(-1.5, abs=1e-6)


# ============================================================================
# Test 3: Verify Points = Weight × Input_Value × scale_factor
# ============================================================================

def test_points_calculation_formula(
    mock_model_2_features,
    discrete_bin_values_2_features,
    generator
):
    """Test that points are calculated as: Points = Weight × Input_Value × scale_factor."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    # Extract weights
    weights, _ = mock_model_2_features.get_coefficients()
    
    # Calculate expected raw score range
    # Feature 1 (weight=2.0): values [10.0, 5.0, -5.0, -10.0]
    #   contributions: [20.0, 10.0, -10.0, -20.0]
    # Feature 2 (weight=-1.5): values [8.0, 4.0, -4.0, -8.0]
    #   contributions: [-12.0, -6.0, 6.0, 12.0]
    # raw_min = -20.0 + (-12.0) = -32.0
    # raw_max = 20.0 + 12.0 = 32.0
    raw_min = -32.0
    raw_max = 32.0
    raw_range = raw_max - raw_min  # 64.0
    scale_factor = 100.0 / raw_range  # 100 / 64 = 1.5625
    
    # Verify scale_factor in scorecard
    assert scorecard.scale_factor == pytest.approx(scale_factor, abs=1e-4)
    
    # Verify points for each bin
    for feature in scorecard.features:
        weight = feature.weight
        for bin_score in feature.bins:
            # Calculate expected raw points
            expected_raw_points = weight * bin_score.input_value
            
            # Calculate expected scaled points
            expected_scaled_points = int(round(expected_raw_points * scale_factor))
            
            # Verify raw_points
            assert bin_score.raw_points == pytest.approx(expected_raw_points, abs=1e-6), \
                f"Raw points mismatch for {feature.feature_name}, bin value {bin_score.input_value}"
            
            # Verify scaled_points (allow for rounding differences)
            assert bin_score.scaled_points == expected_scaled_points, \
                f"Scaled points mismatch for {feature.feature_name}, bin value {bin_score.input_value}. " \
                f"Expected {expected_scaled_points}, got {bin_score.scaled_points}"


# ============================================================================
# Test 4: Total Score is in [0, 100]
# ============================================================================

def test_total_score_in_range(
    mock_model_3_features,
    discrete_bin_values_3_features,
    generator
):
    """Test that all calculated scores are within [0, 100] range."""
    feature_names = ['feature_1', 'feature_2', 'feature_3']
    
    scorecard = generator.generate(
        model=mock_model_3_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_3_features,
        segment='TEST'
    )
    
    # Test multiple combinations of input values
    test_combinations = [
        # Best case (highest values)
        {'feature_1': 15.0, 'feature_2': 12.0, 'feature_3': 9.0},
        # Worst case (lowest values)
        {'feature_1': -15.0, 'feature_2': -12.0, 'feature_3': -9.0},
        # Mixed case 1
        {'feature_1': 10.0, 'feature_2': -6.0, 'feature_3': 4.5},
        # Mixed case 2
        {'feature_1': -10.0, 'feature_2': 6.0, 'feature_3': -4.5},
        # Middle values
        {'feature_1': 0.0, 'feature_2': 0.0, 'feature_3': 0.0},
    ]
    
    for input_values in test_combinations:
        score, _ = generator.calculate_score(scorecard, input_values)
        assert 0 <= score <= 100, \
            f"Score {score} should be within [0, 100] range for input values {input_values}"
    
    # Also verify min_possible_score and max_possible_score are in range
    assert 0 <= scorecard.min_possible_score <= 100, \
        f"min_possible_score {scorecard.min_possible_score} should be in [0, 100]"
    assert 0 <= scorecard.max_possible_score <= 100, \
        f"max_possible_score {scorecard.max_possible_score} should be in [0, 100]"
    assert scorecard.min_possible_score <= scorecard.max_possible_score, \
        f"min_possible_score should be <= max_possible_score"


# ============================================================================
# Test 5: Min Score Corresponds to Worst Combination
# ============================================================================

def test_min_score_is_worst_combination(
    mock_model_2_features,
    discrete_bin_values_2_features,
    generator
):
    """Test that minimum score corresponds to worst combination."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    # Find worst combination (lowest input values for positive weights, highest for negative)
    # Feature 1: weight=2.0 (positive), worst = minimum value = -10.0
    # Feature 2: weight=-1.5 (negative), worst = maximum value = 8.0
    worst_input_values = {
        'feature_1': -10.0,  # Lowest value (gives lowest contribution for positive weight)
        'feature_2': 8.0     # Highest value (gives lowest contribution for negative weight)
    }
    
    # Find best combination (opposite)
    best_input_values = {
        'feature_1': 10.0,   # Highest value (gives highest contribution for positive weight)
        'feature_2': -8.0    # Lowest value (gives highest contribution for negative weight)
    }
    
    worst_score, _ = generator.calculate_score(scorecard, worst_input_values)
    best_score, _ = generator.calculate_score(scorecard, best_input_values)
    
    # Worst should score lower than best
    assert worst_score <= best_score, \
        f"Worst combination should score <= best, got {worst_score} vs {best_score}"
    
    # Worst should be at or near the minimum
    assert worst_score == scorecard.min_possible_score or abs(worst_score - scorecard.min_possible_score) <= 1, \
        f"Worst combination should score at minimum ({scorecard.min_possible_score}), got {worst_score}"


# ============================================================================
# Test 6: Max Score Corresponds to Best Combination
# ============================================================================

def test_max_score_is_best_combination(
    mock_model_2_features,
    discrete_bin_values_2_features,
    generator
):
    """Test that maximum score corresponds to best combination."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    # Find best combination
    # Feature 1: weight=2.0 (positive), best = maximum value = 10.0
    # Feature 2: weight=-1.5 (negative), best = minimum value = -8.0
    best_input_values = {
        'feature_1': 10.0,   # Highest value (gives highest contribution for positive weight)
        'feature_2': -8.0    # Lowest value (gives highest contribution for negative weight)
    }
    
    best_score, _ = generator.calculate_score(scorecard, best_input_values)
    
    # Best should be at or near the maximum
    assert best_score == scorecard.max_possible_score or abs(best_score - scorecard.max_possible_score) <= 1, \
        f"Best combination should score at maximum ({scorecard.max_possible_score}), got {best_score}"
    
    # Best should be 100 (or very close)
    assert best_score == 100 or abs(best_score - 100) <= 1, \
        f"Best combination should score 100 (or very close), got {best_score}"


# ============================================================================
# Test 7: Calculate Score Returns Correct Breakdown
# ============================================================================

def test_calculate_score_breakdown(
    mock_model_2_features,
    discrete_bin_values_2_features,
    generator
):
    """Test that calculate_score returns correct breakdown."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    # Test with specific input values
    input_values = {
        'feature_1': 5.0,
        'feature_2': -4.0
    }
    
    total_score, breakdown = generator.calculate_score(scorecard, input_values)
    
    # Verify breakdown structure
    assert isinstance(breakdown, dict)
    assert 'feature_1' in breakdown
    assert 'feature_2' in breakdown
    
    # Verify breakdown values match bin points
    for feature in scorecard.features:
        feat_name = feature.feature_name
        input_val = input_values[feat_name]
        
        # Find matching bin
        matched_bin = None
        for bin_score in feature.bins:
            if abs(bin_score.input_value - input_val) < 0.01:
                matched_bin = bin_score
                break
        
        assert matched_bin is not None, f"Could not find bin for {feat_name} with value {input_val}"
        assert breakdown[feat_name] == matched_bin.scaled_points, \
            f"Breakdown for {feat_name} should be {matched_bin.scaled_points}, got {breakdown[feat_name]}"
    
    # Verify total score calculation
    # The calculate_score method uses raw_points to compute total_raw,
    # then applies: total_score = round(total_raw * scale_factor + offset)
    # Breakdown contains scaled_points (for display), but total is calculated from raw_points
    
    # Calculate expected total from raw_points
    total_raw_expected = 0.0
    for feature in scorecard.features:
        feat_name = feature.feature_name
        input_val = input_values.get(feat_name)
        if input_val is not None:
            # Find matching bin
            for bin_score in feature.bins:
                if abs(bin_score.input_value - input_val) < 0.01:
                    total_raw_expected += bin_score.raw_points
                    break
    
    expected_score = int(round(total_raw_expected * scorecard.scale_factor + scorecard.offset))
    expected_score = max(0, min(100, expected_score))
    
    assert total_score == expected_score or abs(total_score - expected_score) <= 1, \
        f"Total score {total_score} should match expected {expected_score}. " \
        f"Raw total: {total_raw_expected}, scale_factor: {scorecard.scale_factor}, offset: {scorecard.offset}"


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

def test_single_feature_scorecard(generator):
    """Test scorecard generation with a single feature."""
    model = LinearScorecardNN(input_dim=1)
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[1.0]])
        model.linear.bias.data = torch.tensor([0.0])
    
    unique_values = {
        'feature_1': [10.0, 5.0, -5.0, -10.0]
    }
    feature_names = ['feature_1']
    
    scorecard = generator.generate(
        model=model,
        feature_names=feature_names,
        unique_values_original=unique_values,
        segment='TEST'
    )
    
    assert len(scorecard.features) == 1
    assert scorecard.features[0].feature_name == 'feature_1'
    assert len(scorecard.features[0].bins) == 4
    
    # Test scoring
    best_score, _ = generator.calculate_score(scorecard, {'feature_1': 10.0})
    worst_score, _ = generator.calculate_score(scorecard, {'feature_1': -10.0})
    
    assert best_score >= worst_score, "Best value should score higher than worst value"
    assert 0 <= best_score <= 100
    assert 0 <= worst_score <= 100


def test_missing_feature_in_input_values(
    mock_model_3_features,
    discrete_bin_values_3_features,
    generator
):
    """Test that missing features in input_values are handled gracefully."""
    feature_names = ['feature_1', 'feature_2', 'feature_3']
    
    scorecard = generator.generate(
        model=mock_model_3_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_3_features,
        segment='TEST'
    )
    
    # Test with missing feature
    input_values = {
        'feature_1': 10.0,
        'feature_2': 6.0
        # feature_3 is missing
    }
    
    score, breakdown = generator.calculate_score(scorecard, input_values)
    
    # Should still calculate score (missing feature contributes 0 points)
    assert 0 <= score <= 100
    assert 'feature_1' in breakdown
    assert 'feature_2' in breakdown
    assert 'feature_3' not in breakdown  # Missing feature not in breakdown


def test_all_bins_have_correct_structure(
    mock_model_2_features,
    discrete_bin_values_2_features,
    generator
):
    """Test that all bins have correct structure with input_value, raw_points, and scaled_points."""
    feature_names = ['feature_1', 'feature_2']
    
    scorecard = generator.generate(
        model=mock_model_2_features,
        feature_names=feature_names,
        unique_values_original=discrete_bin_values_2_features,
        segment='TEST'
    )
    
    for feature in scorecard.features:
        assert len(feature.bins) > 0, f"Feature {feature.feature_name} should have at least one bin"
        
        for bin_score in feature.bins:
            # Verify bin has all required attributes
            assert hasattr(bin_score, 'input_value')
            assert hasattr(bin_score, 'raw_points')
            assert hasattr(bin_score, 'scaled_points')
            assert isinstance(bin_score.input_value, (int, float))
            assert isinstance(bin_score.raw_points, (int, float, np.floating))
            assert isinstance(bin_score.scaled_points, (int, np.integer))
            
            # Verify raw_points = weight × input_value
            expected_raw = feature.weight * bin_score.input_value
            assert bin_score.raw_points == pytest.approx(expected_raw, abs=1e-6), \
                f"Raw points should equal weight × input_value for {feature.feature_name}"
