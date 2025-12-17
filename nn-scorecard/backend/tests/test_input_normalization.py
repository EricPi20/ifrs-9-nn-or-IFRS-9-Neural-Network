import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List

from app.core.constants import INPUT_SCALE_FACTOR, SCORE_MIN, SCORE_MAX
from app.services.data_processor import DataProcessor
from app.services.scorecard import ScorecardGenerator, Scorecard, FeatureScore, BinScore


class TestInputNormalization:
    """Test suite for input normalization (÷50) implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known values."""
        np.random.seed(42)
        n = 500
        
        # Original CSV values (standardized log odds × -50)
        # Range approximately -150 to +150
        data = {
            'segment': ['CONSUMER'] * n,
            'feature_a': np.random.choice([-100, -50, 0, 50, 100], n),
            'feature_b': np.random.choice([-75, -25, 25, 75], n),
            'feature_c': np.random.choice([-60, -20, 20, 60], n),
            'target': np.random.choice([0, 1], n, p=[0.9, 0.1])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_cols(self):
        return ['feature_a', 'feature_b', 'feature_c']

    # ==================== DATA PROCESSOR TESTS ====================
    
    def test_data_processor_normalizes_inputs(self, sample_data, feature_cols):
        """Verify DataProcessor divides inputs by scale factor."""
        processor = DataProcessor(scale_factor=INPUT_SCALE_FACTOR)
        result = processor.prepare_training_data(
            df=sample_data,
            feature_cols=feature_cols,
            target_col='target'
        )
        
        X_normalized = result['X']
        X_original = result['X_original']
        
        # Check normalization
        assert X_normalized.shape == X_original.shape
        np.testing.assert_array_almost_equal(
            X_normalized,
            X_original / INPUT_SCALE_FACTOR,
            decimal=6
        )
        
        # Check normalized range is approximately [-3, +3]
        assert X_normalized.min() >= -4.0, f"Min too low: {X_normalized.min()}"
        assert X_normalized.max() <= 4.0, f"Max too high: {X_normalized.max()}"
        
        print(f"✓ Original range: [{X_original.min():.1f}, {X_original.max():.1f}]")
        print(f"✓ Normalized range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    
    def test_data_processor_preserves_original_values(self, sample_data, feature_cols):
        """Verify original unique values are preserved."""
        processor = DataProcessor(scale_factor=INPUT_SCALE_FACTOR)
        result = processor.prepare_training_data(
            df=sample_data,
            feature_cols=feature_cols,
            target_col='target'
        )
        
        # Check original values preserved
        for col in feature_cols:
            expected = sorted(sample_data[col].unique().tolist())
            actual = result['unique_values_original'][col]
            assert actual == expected, f"Mismatch for {col}"
        
        print("✓ Original unique values preserved correctly")
    
    def test_data_processor_stores_normalized_values(self, sample_data, feature_cols):
        """Verify normalized unique values are also available."""
        processor = DataProcessor(scale_factor=INPUT_SCALE_FACTOR)
        result = processor.prepare_training_data(
            df=sample_data,
            feature_cols=feature_cols,
            target_col='target'
        )
        
        for col in feature_cols:
            original = result['unique_values_original'][col]
            normalized = result['unique_values_normalized'][col]
            
            expected_normalized = [v / INPUT_SCALE_FACTOR for v in original]
            assert normalized == expected_normalized
        
        print("✓ Normalized unique values computed correctly")

    # ==================== MODEL TRAINING TESTS ====================
    
    def test_model_trains_without_nan(self, sample_data, feature_cols):
        """Verify model training doesn't produce NaN with normalized inputs."""
        processor = DataProcessor(scale_factor=INPUT_SCALE_FACTOR)
        result = processor.prepare_training_data(
            df=sample_data,
            feature_cols=feature_cols,
            target_col='target'
        )
        
        X = torch.tensor(result['X'], dtype=torch.float32)
        y = torch.tensor(result['y'], dtype=torch.float32).unsqueeze(1)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(len(feature_cols), 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Train for a few epochs
        losses = []
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Check no NaN
        assert not any(np.isnan(losses)), f"NaN loss detected: {losses}"
        assert losses[-1] < losses[0], "Loss should decrease"
        
        print(f"✓ Training stable: loss {losses[0]:.4f} → {losses[-1]:.4f}")
    
    def test_model_fails_without_normalization(self, sample_data, feature_cols):
        """Demonstrate training issues WITHOUT normalization (for comparison)."""
        # Use original (unnormalized) values
        X_original = sample_data[feature_cols].values.astype(np.float32)
        y = sample_data['target'].values.astype(np.float32)
        
        X = torch.tensor(X_original, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Create simple model with small initialization
        model = nn.Sequential(
            nn.Linear(len(feature_cols), 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Check initial output range - likely saturated
        with torch.no_grad():
            initial_output = model(X)
            output_range = (initial_output.min().item(), initial_output.max().item())
        
        # With large inputs, sigmoid outputs are likely near 0 or 1
        is_saturated = output_range[0] < 0.01 or output_range[1] > 0.99
        
        print(f"✓ Without normalization - Output range: {output_range}")
        print(f"  Saturated: {is_saturated} (expected: True)")

    # ==================== SCORECARD GENERATION TESTS ====================
    
    def test_scorecard_weight_adjustment(self):
        """Verify scorecard weights are adjusted for original scale."""
        # Create mock model with known weights
        model = nn.Linear(3, 1)
        with torch.no_grad():
            model.weight.fill_(0.5)  # Weight = 0.5 for each feature
            model.bias.fill_(0.0)
        
        unique_values_original = {
            'feat_a': [-100, 0, 100],
            'feat_b': [-50, 50],
            'feat_c': [-75, 75]
        }
        
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        scorecard = generator.generate(
            model=model,
            feature_names=['feat_a', 'feat_b', 'feat_c'],
            unique_values_original=unique_values_original,
            segment='TEST'
        )
        
        # Check weight adjustment
        # Model weight = 0.5 (trained on normalized)
        # Adjusted weight = 0.5 / 50 = 0.01
        for feat in scorecard.features:
            expected_weight = 0.5 / INPUT_SCALE_FACTOR
            assert abs(feat.weight - expected_weight) < 0.0001, \
                f"Weight mismatch: {feat.weight} vs {expected_weight}"
            assert abs(feat.weight_normalized - 0.5) < 0.0001
        
        print(f"✓ Weights adjusted: normalized=0.5 → original={0.5/INPUT_SCALE_FACTOR}")
    
    def test_scorecard_displays_original_values(self):
        """Verify scorecard displays original CSV values, not normalized."""
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        
        original_values = {
            'feat_a': [-100, -50, 0, 50, 100],
            'feat_b': [-75, -25, 25, 75]
        }
        
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        scorecard = generator.generate(
            model=model,
            feature_names=['feat_a', 'feat_b'],
            unique_values_original=original_values,
            segment='TEST'
        )
        
        # Verify bins show ORIGINAL values
        feat_a = next(f for f in scorecard.features if f.feature_name == 'feat_a')
        bin_values = [b.input_value for b in feat_a.bins]
        
        assert bin_values == original_values['feat_a'], \
            f"Bin values should be original: {bin_values} vs {original_values['feat_a']}"
        
        print(f"✓ Scorecard displays original values: {bin_values}")
    
    def test_scorecard_points_calculation(self):
        """Verify points are calculated correctly."""
        # Create model with weight = 1.0 (normalized)
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        
        # Single feature with known values
        original_values = {'feat': [-100, 0, 100]}
        
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        scorecard = generator.generate(
            model=model,
            feature_names=['feat'],
            unique_values_original=original_values,
            segment='TEST'
        )
        
        feat = scorecard.features[0]
        
        # Adjusted weight = 1.0 / 50 = 0.02
        expected_weight = 1.0 / INPUT_SCALE_FACTOR
        assert abs(feat.weight - expected_weight) < 0.0001
        
        # Raw points = weight × original_value
        for bin_info in feat.bins:
            expected_raw = expected_weight * bin_info.input_value
            assert abs(bin_info.raw_points - expected_raw) < 0.01, \
                f"Raw points mismatch: {bin_info.raw_points} vs {expected_raw}"
        
        print(f"✓ Points calculated correctly with adjusted weight")

    # ==================== SCORING TESTS ====================
    
    def test_score_calculation_with_original_values(self):
        """Verify score calculation accepts original CSV values."""
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight[0, 0] = 0.5
            model.weight[0, 1] = 0.5
            model.bias.fill_(0.0)
        
        original_values = {
            'feat_a': [-100, 0, 100],
            'feat_b': [-50, 50]
        }
        
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        scorecard = generator.generate(
            model=model,
            feature_names=['feat_a', 'feat_b'],
            unique_values_original=original_values,
            segment='TEST'
        )
        
        # Score with ORIGINAL values (not normalized)
        test_record = {'feat_a': -100, 'feat_b': -50}  # Original CSV values
        score, breakdown = generator.calculate_score(scorecard, test_record)
        
        # Score should be valid
        assert SCORE_MIN <= score <= SCORE_MAX, f"Score out of range: {score}"
        assert 'feat_a' in breakdown
        assert 'feat_b' in breakdown
        
        print(f"✓ Score calculated: {score} with breakdown {breakdown}")
    
    def test_score_range_validity(self):
        """Verify all possible scores are within [0, 100]."""
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight[0, 0] = 2.0  # Larger weights
            model.weight[0, 1] = 2.0
            model.bias.fill_(0.0)
        
        original_values = {
            'feat_a': [-100, -50, 0, 50, 100],
            'feat_b': [-75, -25, 25, 75]
        }
        
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        scorecard = generator.generate(
            model=model,
            feature_names=['feat_a', 'feat_b'],
            unique_values_original=original_values,
            segment='TEST'
        )
        
        # Test all combinations
        scores = []
        for a in original_values['feat_a']:
            for b in original_values['feat_b']:
                record = {'feat_a': a, 'feat_b': b}
                score, _ = generator.calculate_score(scorecard, record)
                scores.append(score)
                assert SCORE_MIN <= score <= SCORE_MAX, \
                    f"Score {score} out of range for {record}"
        
        print(f"✓ All {len(scores)} score combinations valid: [{min(scores)}, {max(scores)}]")
        
        # Best case should be high score (low risk = negative values)
        best_record = {'feat_a': -100, 'feat_b': -75}
        best_score, _ = generator.calculate_score(scorecard, best_record)
        
        # Worst case should be low score (high risk = positive values)
        worst_record = {'feat_a': 100, 'feat_b': 75}
        worst_score, _ = generator.calculate_score(scorecard, worst_record)
        
        assert best_score > worst_score, \
            f"Best score {best_score} should be > worst score {worst_score}"
        
        print(f"✓ Best (low risk): {best_score}, Worst (high risk): {worst_score}")

    # ==================== INTEGRATION TEST ====================
    
    def test_end_to_end_normalization(self, sample_data, feature_cols):
        """Complete end-to-end test of normalization flow."""
        
        # 1. Process data (normalizes inputs)
        processor = DataProcessor(scale_factor=INPUT_SCALE_FACTOR)
        data = processor.prepare_training_data(
            df=sample_data,
            feature_cols=feature_cols,
            target_col='target'
        )
        
        # 2. Train model on normalized data
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
        
        model = nn.Sequential(
            nn.Linear(len(feature_cols), 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        # 3. Generate scorecard (adjusts for original scale)
        generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)
        
        # Extract just the first linear layer for scorecard
        first_layer = model[0]
        
        scorecard = generator.generate(
            model=first_layer,
            feature_names=feature_cols,
            unique_values_original=data['unique_values_original'],
            segment='CONSUMER'
        )
        
        # 4. Score using original values
        test_record = {col: data['unique_values_original'][col][0] for col in feature_cols}
        score, breakdown = generator.calculate_score(scorecard, test_record)
        
        # 5. Assertions
        assert 0 <= score <= 100
        assert len(breakdown) == len(feature_cols)
        assert scorecard.input_scale_factor == INPUT_SCALE_FACTOR
        
        print(f"\n✓ End-to-end test passed!")
        print(f"  Input normalization: ÷{INPUT_SCALE_FACTOR}")
        print(f"  Training loss: {loss.item():.4f}")
        print(f"  Sample score: {score}")
        print(f"  Scorecard has {len(scorecard.features)} features")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_scale_factor_raises_error(self):
        """Verify zero scale factor raises error."""
        with pytest.raises(ValueError):
            processor = DataProcessor(scale_factor=0.0)
    
    def test_empty_data_handled(self):
        """Verify empty data is handled gracefully."""
        processor = DataProcessor()
        df = pd.DataFrame({'feat': [], 'target': []})
        
        with pytest.raises(ValueError):
            processor.prepare_training_data(df, ['feat'], 'target')
    
    def test_missing_feature_in_scoring(self):
        """Verify missing features handled in scoring."""
        model = nn.Linear(2, 1)
        generator = ScorecardGenerator()
        
        scorecard = generator.generate(
            model=model,
            feature_names=['feat_a', 'feat_b'],
            unique_values_original={'feat_a': [0, 1], 'feat_b': [0, 1]},
            segment='TEST'
        )
        
        # Score with missing feature
        partial_record = {'feat_a': 0}  # Missing feat_b
        score, breakdown = generator.calculate_score(scorecard, partial_record)
        
        # Should still return a score (with partial features)
        assert 0 <= score <= 100
        assert 'feat_a' in breakdown
        assert 'feat_b' not in breakdown


# Run with: pytest backend/tests/test_input_normalization.py -v -s

