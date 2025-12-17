"""
Comprehensive Tests for Pydantic Schemas

Tests cover:
- Valid data creates models correctly
- Validation errors for invalid data
- Nested models work properly
- JSON serialization round-trips
- Default values are applied correctly
"""

import json
import pytest
from pydantic import ValidationError

from app.models.schemas import (
    SegmentStats,
    FeatureBin,
    FeatureSummary,
    UploadResponse,
    NeuralNetworkConfig,
    RegularizationConfig,
    LossConfig,
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingRequest,
    TrainingProgress,
    EpochMetrics,
    TrainingHistory,
    ModelMetrics,
    ScorecardBinPoints,
    FeatureScorecard,
    Scorecard,
    ScorecardResults,
    ScoreRequest,
    ScoreBreakdown,
    ScoreResponse,
)


# ============================================================================
# TEST VALID DATA CREATES MODELS
# ============================================================================

class TestValidData:
    """Test that valid data creates models correctly."""
    
    def test_segment_stats_valid(self):
        """Test SegmentStats with valid data."""
        stats = SegmentStats(
            segment="CONSUMER",
            count=10000,
            bad_count=500,
            bad_rate=0.05
        )
        assert stats.segment == "CONSUMER"
        assert stats.count == 10000
        assert stats.bad_count == 500
        assert stats.bad_rate == 0.05
    
    def test_feature_bin_valid(self):
        """Test FeatureBin with valid data."""
        bin = FeatureBin(
            bin_label="Poor (500-580)",
            woe_value=-0.523
        )
        assert bin.bin_label == "Poor (500-580)"
        assert bin.woe_value == -0.523
    
    def test_feature_summary_valid(self):
        """Test FeatureSummary with valid data."""
        bins = [
            FeatureBin(bin_label="Very Poor (<500)", woe_value=-0.823),
            FeatureBin(bin_label="Poor (500-580)", woe_value=-0.523),
        ]
        summary = FeatureSummary(
            name="credit_score",
            num_bins=2,
            bins=bins,
            min_woe=-0.823,
            max_woe=-0.523,
            mean_woe=-0.673,
            target_correlation=0.45
        )
        assert summary.name == "credit_score"
        assert summary.num_bins == 2
        assert len(summary.bins) == 2
        assert summary.min_woe == -0.823
        assert summary.max_woe == -0.523
        assert summary.mean_woe == -0.673
        assert summary.target_correlation == 0.45
    
    def test_upload_response_valid(self):
        """Test UploadResponse with valid data."""
        segment_stats = [
            SegmentStats(segment="CONSUMER", count=10000, bad_count=500, bad_rate=0.05)
        ]
        feature_summary = [
            FeatureSummary(
                name="credit_score",
                num_bins=2,
                bins=[FeatureBin(bin_label="Poor", woe_value=-0.5)],
                min_woe=-0.5,
                max_woe=-0.5,
                mean_woe=-0.5,
                target_correlation=0.45
            )
        ]
        response = UploadResponse(
            file_id="550e8400-e29b-41d4-a716-446655440000",
            filename="portfolio_data.xlsx",
            num_records=50000,
            num_features=15,
            segments=["CONSUMER", "SME"],
            segment_stats=segment_stats,
            feature_summary=feature_summary,
            target_distribution={"good_count": 47500, "bad_count": 2500, "bad_rate": 0.05}
        )
        assert response.file_id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.filename == "portfolio_data.xlsx"
        assert response.num_records == 50000
        assert response.num_features == 15
        assert len(response.segments) == 2
        assert len(response.segment_stats) == 1
        assert len(response.feature_summary) == 1
    
    def test_neural_network_config_valid(self):
        """Test NeuralNetworkConfig with valid data."""
        config = NeuralNetworkConfig(
            model_type="neural_network",
            hidden_layers=[32, 16],
            activation="relu",
            dropout_rate=0.2,
            use_batch_norm=True
        )
        assert config.model_type == "neural_network"
        assert config.hidden_layers == [32, 16]
        assert config.activation == "relu"
        assert config.dropout_rate == 0.2
        assert config.use_batch_norm is True
    
    def test_regularization_config_valid(self):
        """Test RegularizationConfig with valid data."""
        config = RegularizationConfig(
            l1_lambda=0.0,
            l2_lambda=0.01,
            gradient_clip_norm=1.0
        )
        assert config.l1_lambda == 0.0
        assert config.l2_lambda == 0.01
        assert config.gradient_clip_norm == 1.0
    
    def test_loss_config_valid(self):
        """Test LossConfig with valid data."""
        config = LossConfig(
            loss_type="combined",
            loss_alpha=0.3,
            auc_gamma=2.0
        )
        assert config.loss_type == "combined"
        assert config.loss_alpha == 0.3
        assert config.auc_gamma == 2.0
    
    def test_training_config_valid(self):
        """Test TrainingConfig with valid data."""
        config = TrainingConfig(
            segment="ALL",
            test_size=0.30,
            selected_features=["credit_score", "debt_to_income"],
            network=NeuralNetworkConfig(hidden_layers=[32, 16]),
            regularization=RegularizationConfig(l2_lambda=0.01),
            loss=LossConfig(loss_type="combined"),
            learning_rate=0.001,
            batch_size=256,
            epochs=100,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=15),
            use_class_weights=True
        )
        assert config.segment == "ALL"
        assert config.test_size == 0.30
        assert config.selected_features == ["credit_score", "debt_to_income"]
        assert config.network.hidden_layers == [32, 16]
        assert config.regularization.l2_lambda == 0.01
        assert config.loss.loss_type == "combined"
        assert config.learning_rate == 0.001
        assert config.batch_size == 256
        assert config.epochs == 100
    
    def test_training_request_valid(self):
        """Test TrainingRequest with valid data."""
        training_config = TrainingConfig(
            segment="ALL",
            test_size=0.30
        )
        request = TrainingRequest(
            file_id="550e8400-e29b-41d4-a716-446655440000",
            config=training_config
        )
        assert request.file_id == "550e8400-e29b-41d4-a716-446655440000"
        assert request.config.segment == "ALL"
        assert request.config.test_size == 0.30
    
    def test_training_progress_valid(self):
        """Test TrainingProgress with valid data."""
        progress = TrainingProgress(
            job_id="job-12345",
            status="training",
            current_epoch=45,
            total_epochs=100,
            current_metrics={"train_ar": 0.65, "test_ar": 0.62},
            message="Training in progress..."
        )
        assert progress.job_id == "job-12345"
        assert progress.status == "training"
        assert progress.current_epoch == 45
        assert progress.total_epochs == 100
        assert progress.current_metrics["train_ar"] == 0.65
        assert progress.message == "Training in progress..."
    
    def test_epoch_metrics_valid(self):
        """Test EpochMetrics with valid data."""
        metrics = EpochMetrics(
            epoch=50,
            train_loss=0.523,
            test_loss=0.545,
            train_auc=0.825,
            test_auc=0.810,
            train_ar=0.65,
            test_ar=0.62,
            train_ks=0.45,
            test_ks=0.42,
            learning_rate=0.001
        )
        assert metrics.epoch == 50
        assert metrics.train_loss == 0.523
        assert metrics.test_auc == 0.810
        assert metrics.train_ar == 0.65
        assert metrics.learning_rate == 0.001
    
    def test_model_metrics_valid(self):
        """Test ModelMetrics with valid data."""
        metrics = ModelMetrics(
            auc_roc=0.810,
            gini_ar=0.62,
            ks_statistic=0.42,
            log_loss=0.545,
            brier_score=0.125,
            accuracy=0.85,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
            ks_decile=1,
            cumulative_bad_rate_top_decile=0.25
        )
        assert metrics.auc_roc == 0.810
        assert metrics.gini_ar == 0.62
        assert metrics.ks_statistic == 0.42
        assert metrics.accuracy == 0.85
    
    def test_scorecard_valid(self):
        """Test Scorecard with valid data."""
        bins = [
            ScorecardBinPoints(bin_label="Very Poor (<500)", woe_value=-0.823, points=10),
            ScorecardBinPoints(bin_label="Poor (500-580)", woe_value=-0.523, points=25),
        ]
        features = [
            FeatureScorecard(
                feature="credit_score",
                weight=0.45,
                bins=bins
            )
        ]
        scorecard = Scorecard(
            segment="CONSUMER",
            base_points=50,
            features=features,
            score_range=(0, 100),
            total_min_score=0,
            total_max_score=100
        )
        assert scorecard.segment == "CONSUMER"
        assert scorecard.base_points == 50
        assert len(scorecard.features) == 1
        assert scorecard.score_range == (0, 100)
        assert scorecard.total_min_score == 0
        assert scorecard.total_max_score == 100
    
    def test_score_request_valid(self):
        """Test ScoreRequest with valid data."""
        request = ScoreRequest(
            records=[
                {"credit_score": 0.623, "debt_to_income": -0.234},
                {"credit_score": -0.523, "debt_to_income": -0.823}
            ]
        )
        assert len(request.records) == 2
        assert request.records[0]["credit_score"] == 0.623
    
    def test_score_breakdown_valid(self):
        """Test ScoreBreakdown with valid data."""
        breakdown = ScoreBreakdown(
            base_points=50,
            feature_points={"credit_score": 25, "debt_to_income": 15},
            total_score=75,
            probability=0.15
        )
        assert breakdown.base_points == 50
        assert breakdown.feature_points["credit_score"] == 25
        assert breakdown.total_score == 75
        assert breakdown.probability == 0.15
    
    def test_score_response_valid(self):
        """Test ScoreResponse with valid data."""
        breakdowns = [
            ScoreBreakdown(
                base_points=50,
                feature_points={"credit_score": 25},
                total_score=75,
                probability=0.15
            )
        ]
        response = ScoreResponse(scores=breakdowns)
        assert len(response.scores) == 1
        assert response.scores[0].total_score == 75


# ============================================================================
# TEST VALIDATION ERRORS
# ============================================================================

class TestValidationErrors:
    """Test that validation errors are raised for invalid data."""
    
    def test_neural_network_config_invalid_activation(self):
        """Test NeuralNetworkConfig with invalid activation."""
        with pytest.raises(ValidationError) as exc_info:
            NeuralNetworkConfig(activation="invalid_activation")
        errors = exc_info.value.errors()
        assert any("Activation must be one of" in str(err.get("msg", "")) for err in errors)
    
    def test_training_config_test_size_too_large(self):
        """Test TrainingConfig with test_size > 0.5."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(test_size=0.6)
        errors = exc_info.value.errors()
        # Check for either Field constraint or validator error
        assert any(
            "less than or equal to 0.5" in str(err.get("msg", "")) or
            "Test size must be between 0.1 and 0.5" in str(err.get("msg", ""))
            for err in errors
        )
    
    def test_loss_config_alpha_too_large(self):
        """Test LossConfig with alpha > 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            LossConfig(loss_alpha=1.5)
        errors = exc_info.value.errors()
        # Check for either Field constraint or validator error
        assert any(
            "less than or equal to 1" in str(err.get("msg", "")) or
            "less than or equal to 1.0" in str(err.get("msg", "")) or
            "Alpha must be between 0 and 1" in str(err.get("msg", ""))
            for err in errors
        )
    
    def test_neural_network_config_negative_neurons(self):
        """Test NeuralNetworkConfig with negative neurons in hidden_layers."""
        with pytest.raises(ValidationError) as exc_info:
            NeuralNetworkConfig(hidden_layers=[32, -16])
        errors = exc_info.value.errors()
        assert any("Neurons per layer must be positive" in str(err.get("msg", "")) for err in errors)
    
    def test_segment_stats_negative_count(self):
        """Test SegmentStats with negative count."""
        with pytest.raises(ValidationError) as exc_info:
            SegmentStats(
                segment="CONSUMER",
                count=-100,
                bad_count=500,
                bad_rate=0.05
            )
        errors = exc_info.value.errors()
        assert any("greater than or equal to 0" in str(err.get("msg", "")) for err in errors)
    
    def test_feature_summary_invalid_num_bins(self):
        """Test FeatureSummary with invalid num_bins."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureSummary(
                name="credit_score",
                num_bins=1,  # Less than 2
                bins=[FeatureBin(bin_label="Poor", woe_value=-0.5)],
                min_woe=-0.5,
                max_woe=-0.5,
                mean_woe=-0.5,
                target_correlation=0.45
            )
        errors = exc_info.value.errors()
        assert any("greater than or equal to 2" in str(err.get("msg", "")) for err in errors)
    
    def test_training_config_test_size_too_small(self):
        """Test TrainingConfig with test_size < 0.1."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(test_size=0.05)
        errors = exc_info.value.errors()
        assert any(
            "greater than or equal to 0.1" in str(err.get("msg", "")) or
            "Test size must be between 0.1 and 0.5" in str(err.get("msg", ""))
            for err in errors
        )
    
    def test_loss_config_alpha_negative(self):
        """Test LossConfig with negative alpha."""
        with pytest.raises(ValidationError) as exc_info:
            LossConfig(loss_alpha=-0.1)
        errors = exc_info.value.errors()
        assert any(
            "greater than or equal to 0" in str(err.get("msg", "")) or
            "greater than or equal to 0.0" in str(err.get("msg", "")) or
            "Alpha must be between 0 and 1" in str(err.get("msg", ""))
            for err in errors
        )
    
    def test_neural_network_config_invalid_model_type(self):
        """Test NeuralNetworkConfig with invalid model_type."""
        with pytest.raises(ValidationError) as exc_info:
            NeuralNetworkConfig(model_type="invalid_type")
        errors = exc_info.value.errors()
        assert any("Model type must be one of" in str(err.get("msg", "")) for err in errors)
    
    def test_loss_config_invalid_loss_type(self):
        """Test LossConfig with invalid loss_type."""
        with pytest.raises(ValidationError) as exc_info:
            LossConfig(loss_type="invalid_loss")
        errors = exc_info.value.errors()
        assert any("Loss type must be one of" in str(err.get("msg", "")) for err in errors)
    
    def test_loss_config_invalid_auc_loss_type(self):
        """Test LossConfig with invalid auc_loss_type."""
        with pytest.raises(ValidationError) as exc_info:
            LossConfig(auc_loss_type="invalid_auc_type")
        errors = exc_info.value.errors()
        assert any("AUC loss type must be one of" in str(err.get("msg", "")) for err in errors)
    
    def test_scorecard_invalid_score_range(self):
        """Test Scorecard with invalid score_range."""
        with pytest.raises(ValidationError) as exc_info:
            Scorecard(
                segment="CONSUMER",
                base_points=50,
                features=[],
                score_range=(0, 200),  # Invalid range
                total_min_score=0,
                total_max_score=100
            )
        errors = exc_info.value.errors()
        assert any("Score range must be (0, 100)" in str(err.get("msg", "")) for err in errors)
    
    def test_scorecard_zero_neurons(self):
        """Test NeuralNetworkConfig with zero neurons."""
        with pytest.raises(ValidationError) as exc_info:
            NeuralNetworkConfig(hidden_layers=[32, 0])
        errors = exc_info.value.errors()
        assert any("Neurons per layer must be positive" in str(err.get("msg", "")) for err in errors)


# ============================================================================
# TEST NESTED MODELS
# ============================================================================

class TestNestedModels:
    """Test nested models work correctly."""
    
    def test_training_request_nested_configs(self):
        """Test TrainingRequest with nested configs."""
        network_config = NeuralNetworkConfig(
            model_type="neural_network",
            hidden_layers=[64, 32, 16],
            activation="tanh",
            dropout_rate=0.3,
            use_batch_norm=False
        )
        regularization_config = RegularizationConfig(
            l1_lambda=0.001,
            l2_lambda=0.02,
            gradient_clip_norm=2.0
        )
        loss_config = LossConfig(
            loss_type="combined",
            loss_alpha=0.2,
            auc_gamma=3.0
        )
        training_config = TrainingConfig(
            segment="CONSUMER",
            test_size=0.25,
            selected_features=["credit_score", "debt_to_income", "employment_years"],
            network=network_config,
            regularization=regularization_config,
            loss=loss_config,
            learning_rate=0.005,
            batch_size=512,
            epochs=200,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=20),
            use_class_weights=False
        )
        request = TrainingRequest(
            file_id="test-file-id",
            config=training_config
        )
        
        # Verify all nested values are accessible
        assert request.config.segment == "CONSUMER"
        assert request.config.network.model_type == "neural_network"
        assert request.config.network.hidden_layers == [64, 32, 16]
        assert request.config.network.activation == "tanh"
        assert request.config.regularization.l1_lambda == 0.001
        assert request.config.regularization.l2_lambda == 0.02
        assert request.config.loss.loss_type == "combined"
        assert request.config.loss.loss_alpha == 0.2
        assert request.config.learning_rate == 0.005
        assert request.config.batch_size == 512
    
    def test_upload_response_nested_models(self):
        """Test UploadResponse with nested SegmentStats and FeatureSummary."""
        segment_stats = [
            SegmentStats(segment="CONSUMER", count=30000, bad_count=1500, bad_rate=0.05),
            SegmentStats(segment="SME", count=20000, bad_count=1000, bad_rate=0.05)
        ]
        feature_summary = [
            FeatureSummary(
                name="credit_score",
                num_bins=3,
                bins=[
                    FeatureBin(bin_label="Poor", woe_value=-0.5),
                    FeatureBin(bin_label="Fair", woe_value=0.0),
                    FeatureBin(bin_label="Good", woe_value=0.5)
                ],
                min_woe=-0.5,
                max_woe=0.5,
                mean_woe=0.0,
                target_correlation=0.45
            )
        ]
        response = UploadResponse(
            file_id="test-id",
            filename="test.xlsx",
            num_records=50000,
            num_features=1,
            segments=["CONSUMER", "SME"],
            segment_stats=segment_stats,
            feature_summary=feature_summary,
            target_distribution={"good_count": 47500, "bad_count": 2500, "bad_rate": 0.05}
        )
        
        # Verify nested models
        assert len(response.segment_stats) == 2
        assert response.segment_stats[0].segment == "CONSUMER"
        assert response.segment_stats[1].segment == "SME"
        assert len(response.feature_summary) == 1
        assert response.feature_summary[0].name == "credit_score"
        assert len(response.feature_summary[0].bins) == 3
    
    def test_scorecard_nested_features(self):
        """Test Scorecard with nested FeatureScorecard and ScorecardBinPoints."""
        bins = [
            ScorecardBinPoints(bin_label="Very Poor", woe_value=-0.8, points=10),
            ScorecardBinPoints(bin_label="Poor", woe_value=-0.5, points=25),
            ScorecardBinPoints(bin_label="Good", woe_value=0.2, points=50),
            ScorecardBinPoints(bin_label="Excellent", woe_value=0.6, points=75)
        ]
        features = [
            FeatureScorecard(
                feature="credit_score",
                weight=0.45,
                bins=bins[:2]
            ),
            FeatureScorecard(
                feature="debt_to_income",
                weight=0.35,
                bins=bins[2:]
            )
        ]
        scorecard = Scorecard(
            segment="CONSUMER",
            base_points=50,
            features=features,
            score_range=(0, 100),
            total_min_score=0,
            total_max_score=100
        )
        
        # Verify nested structure
        assert len(scorecard.features) == 2
        assert scorecard.features[0].feature == "credit_score"
        assert len(scorecard.features[0].bins) == 2
        assert scorecard.features[0].bins[0].bin_label == "Very Poor"
        assert scorecard.features[1].feature == "debt_to_income"
        assert len(scorecard.features[1].bins) == 2


# ============================================================================
# TEST JSON SERIALIZATION
# ============================================================================

class TestJSONSerialization:
    """Test JSON serialization round-trips."""
    
    def test_segment_stats_json_roundtrip(self):
        """Test SegmentStats JSON serialization."""
        original = SegmentStats(
            segment="CONSUMER",
            count=10000,
            bad_count=500,
            bad_rate=0.05
        )
        json_str = original.model_dump_json()
        parsed = SegmentStats.model_validate_json(json_str)
        assert parsed.segment == original.segment
        assert parsed.count == original.count
        assert parsed.bad_count == original.bad_count
        assert parsed.bad_rate == original.bad_rate
    
    def test_neural_network_config_json_roundtrip(self):
        """Test NeuralNetworkConfig JSON serialization."""
        original = NeuralNetworkConfig(
            model_type="neural_network",
            hidden_layers=[32, 16],
            activation="relu",
            dropout_rate=0.2,
            use_batch_norm=True
        )
        json_str = original.model_dump_json()
        parsed = NeuralNetworkConfig.model_validate_json(json_str)
        assert parsed.model_type == original.model_type
        assert parsed.hidden_layers == original.hidden_layers
        assert parsed.activation == original.activation
        assert parsed.dropout_rate == original.dropout_rate
        assert parsed.use_batch_norm == original.use_batch_norm
    
    def test_training_config_json_roundtrip(self):
        """Test TrainingConfig JSON serialization with nested models."""
        original = TrainingConfig(
            segment="ALL",
            test_size=0.30,
            selected_features=["credit_score"],
            network=NeuralNetworkConfig(hidden_layers=[64, 32]),
            regularization=RegularizationConfig(l2_lambda=0.02),
            loss=LossConfig(loss_type="combined", loss_alpha=0.3),
            learning_rate=0.001,
            batch_size=256,
            epochs=100
        )
        json_str = original.model_dump_json()
        parsed = TrainingConfig.model_validate_json(json_str)
        assert parsed.segment == original.segment
        assert parsed.test_size == original.test_size
        assert parsed.selected_features == original.selected_features
        assert parsed.network.hidden_layers == original.network.hidden_layers
        assert parsed.regularization.l2_lambda == original.regularization.l2_lambda
        assert parsed.loss.loss_type == original.loss.loss_type
        assert parsed.learning_rate == original.learning_rate
    
    def test_training_request_json_roundtrip(self):
        """Test TrainingRequest JSON serialization."""
        config = TrainingConfig(segment="ALL", test_size=0.30)
        original = TrainingRequest(
            file_id="test-id",
            config=config
        )
        json_str = original.model_dump_json()
        parsed = TrainingRequest.model_validate_json(json_str)
        assert parsed.file_id == original.file_id
        assert parsed.config.segment == original.config.segment
        assert parsed.config.test_size == original.config.test_size
    
    def test_scorecard_json_roundtrip(self):
        """Test Scorecard JSON serialization with nested models."""
        bins = [
            ScorecardBinPoints(bin_label="Poor", woe_value=-0.5, points=25),
            ScorecardBinPoints(bin_label="Good", woe_value=0.2, points=50)
        ]
        features = [
            FeatureScorecard(feature="credit_score", weight=0.45, bins=bins)
        ]
        original = Scorecard(
            segment="CONSUMER",
            base_points=50,
            features=features,
            score_range=(0, 100),
            total_min_score=0,
            total_max_score=100
        )
        json_str = original.model_dump_json()
        parsed = Scorecard.model_validate_json(json_str)
        assert parsed.segment == original.segment
        assert parsed.base_points == original.base_points
        assert len(parsed.features) == len(original.features)
        assert parsed.features[0].feature == original.features[0].feature
        assert parsed.score_range == original.score_range
    
    def test_model_metrics_json_roundtrip(self):
        """Test ModelMetrics JSON serialization."""
        original = ModelMetrics(
            auc_roc=0.810,
            gini_ar=0.62,
            ks_statistic=0.42,
            log_loss=0.545,
            brier_score=0.125,
            accuracy=0.85,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
            ks_decile=1,
            cumulative_bad_rate_top_decile=0.25
        )
        json_str = original.model_dump_json()
        parsed = ModelMetrics.model_validate_json(json_str)
        assert parsed.auc_roc == original.auc_roc
        assert parsed.gini_ar == original.gini_ar
        assert parsed.ks_statistic == original.ks_statistic
        assert parsed.accuracy == original.accuracy


# ============================================================================
# TEST DEFAULTS
# ============================================================================

class TestDefaults:
    """Test that defaults are applied correctly."""
    
    def test_neural_network_config_defaults(self):
        """Test NeuralNetworkConfig defaults."""
        config = NeuralNetworkConfig()
        assert config.model_type == "neural_network"
        assert config.hidden_layers == [32, 16]
        assert config.activation == "relu"
        assert config.dropout_rate == 0.2
        assert config.use_batch_norm is True
    
    def test_regularization_config_defaults(self):
        """Test RegularizationConfig defaults."""
        config = RegularizationConfig()
        assert config.l1_lambda == 0.0
        assert config.l2_lambda == 0.01
        assert config.gradient_clip_norm == 1.0
    
    def test_loss_config_defaults(self):
        """Test LossConfig defaults."""
        config = LossConfig()
        assert config.loss_type == "combined"
        assert config.loss_alpha == 0.3
        assert config.auc_gamma == 2.0
        assert config.auc_loss_type == "pairwise"
        assert config.margin == 0.0
    
    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()
        assert config.segment == "ALL"
        assert config.test_size == 0.30
        assert config.selected_features is None
        assert isinstance(config.network, NeuralNetworkConfig)
        assert config.network.hidden_layers == [32, 16]
        assert isinstance(config.regularization, RegularizationConfig)
        assert config.regularization.l2_lambda == 0.01
        assert isinstance(config.loss, LossConfig)
        assert config.loss.loss_type == "combined"
        assert config.learning_rate == 0.001
        assert config.batch_size == 256
        assert config.epochs == 100
        assert config.early_stopping.enabled is False  # Default is disabled
        assert config.early_stopping.patience == 10  # Default value
        assert config.use_class_weights is True
    
    def test_training_progress_optional_fields(self):
        """Test TrainingProgress with optional fields as None."""
        progress = TrainingProgress(
            job_id="job-123",
            status="queued",
            current_epoch=0,
            total_epochs=100
        )
        assert progress.job_id == "job-123"
        assert progress.status == "queued"
        assert progress.current_epoch == 0
        assert progress.total_epochs == 100
        assert progress.current_metrics is None
        assert progress.message is None
    
    def test_training_config_partial_override(self):
        """Test TrainingConfig with partial overrides."""
        # Only override some fields, rest should use defaults
        config = TrainingConfig(
            segment="CONSUMER",
            test_size=0.25
        )
        assert config.segment == "CONSUMER"
        assert config.test_size == 0.25
        # Check defaults still apply
        assert config.selected_features is None
        assert config.network.hidden_layers == [32, 16]
        assert config.learning_rate == 0.001
        assert config.batch_size == 256

