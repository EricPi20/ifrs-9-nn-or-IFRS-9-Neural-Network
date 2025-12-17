"""
Tests for Configuration Module

Comprehensive tests for the Settings class including:
- Default value loading
- Environment variable overrides
- Directory creation
- Validation of invalid values
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestDefaultValues:
    """Test that default values load correctly."""
    
    def test_default_upload_dir(self):
        """Test UPLOAD_DIR default value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings()
                assert settings.UPLOAD_DIR == Path("./data/uploads")
    
    def test_default_model_dir(self):
        """Test MODEL_DIR default value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings()
                assert settings.MODEL_DIR == Path("./data/models")
    
    def test_default_max_upload_size(self):
        """Test MAX_UPLOAD_SIZE_MB default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.MAX_UPLOAD_SIZE_MB == 100
            assert settings.max_upload_size_bytes == 100 * 1024 * 1024
    
    def test_default_allowed_extensions(self):
        """Test ALLOWED_EXTENSIONS default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.ALLOWED_EXTENSIONS == [".csv"]
    
    def test_default_target_column(self):
        """Test DEFAULT_TARGET_COLUMN default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_TARGET_COLUMN == "target"
    
    def test_default_id_column(self):
        """Test DEFAULT_ID_COLUMN default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_ID_COLUMN == "account_id"
    
    def test_default_segment_column(self):
        """Test DEFAULT_SEGMENT_COLUMN default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_SEGMENT_COLUMN == "segment"
    
    def test_default_test_size(self):
        """Test DEFAULT_TEST_SIZE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_TEST_SIZE == 0.30
    
    def test_default_random_state(self):
        """Test RANDOM_STATE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.RANDOM_STATE == 42
    
    def test_default_hidden_layers(self):
        """Test DEFAULT_HIDDEN_LAYERS default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_HIDDEN_LAYERS == [32, 16]
    
    def test_default_activation(self):
        """Test DEFAULT_ACTIVATION default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_ACTIVATION == "relu"
    
    def test_default_dropout_rate(self):
        """Test DEFAULT_DROPOUT_RATE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_DROPOUT_RATE == 0.2
    
    def test_default_learning_rate(self):
        """Test DEFAULT_LEARNING_RATE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_LEARNING_RATE == 0.001
    
    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_BATCH_SIZE == 256
    
    def test_default_epochs(self):
        """Test DEFAULT_EPOCHS default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_EPOCHS == 100
    
    def test_default_early_stopping_patience(self):
        """Test DEFAULT_EARLY_STOPPING_PATIENCE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_EARLY_STOPPING_PATIENCE == 15
    
    def test_default_l1_lambda(self):
        """Test DEFAULT_L1_LAMBDA default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_L1_LAMBDA == 0.0
    
    def test_default_l2_lambda(self):
        """Test DEFAULT_L2_LAMBDA default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_L2_LAMBDA == 0.01
    
    def test_default_use_batch_norm(self):
        """Test DEFAULT_USE_BATCH_NORM default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_USE_BATCH_NORM is True
    
    def test_default_loss_type(self):
        """Test DEFAULT_LOSS_TYPE default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_LOSS_TYPE == "combined"
    
    def test_default_loss_alpha(self):
        """Test DEFAULT_LOSS_ALPHA default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_LOSS_ALPHA == 0.3
    
    def test_default_auc_gamma(self):
        """Test DEFAULT_AUC_GAMMA default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.DEFAULT_AUC_GAMMA == 2.0
    
    def test_default_score_min(self):
        """Test SCORE_MIN default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.SCORE_MIN == 0
    
    def test_default_score_max(self):
        """Test SCORE_MAX default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.SCORE_MAX == 100
    
    def test_default_api_prefix(self):
        """Test API_PREFIX default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.API_PREFIX == "/api"
    
    def test_default_cors_origins(self):
        """Test CORS_ORIGINS default value."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.CORS_ORIGINS == ["http://localhost:5173"]


class TestEnvironmentVariableOverride:
    """Test that environment variables override defaults."""
    
    def test_override_upload_dir(self):
        """Test UPLOAD_DIR override from environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "custom_uploads")
            with patch.dict(os.environ, {"UPLOAD_DIR": custom_dir}, clear=False):
                settings = Settings()
                assert str(settings.UPLOAD_DIR) == custom_dir
    
    def test_override_model_dir(self):
        """Test MODEL_DIR override from environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "custom_models")
            with patch.dict(os.environ, {"MODEL_DIR": custom_dir}, clear=False):
                settings = Settings()
                assert str(settings.MODEL_DIR) == custom_dir
    
    def test_override_max_upload_size(self):
        """Test MAX_UPLOAD_SIZE_MB override from environment."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_MB": "200"}, clear=False):
            settings = Settings()
            assert settings.MAX_UPLOAD_SIZE_MB == 200
            assert settings.max_upload_size_bytes == 200 * 1024 * 1024
    
    def test_override_allowed_extensions_string(self):
        """Test ALLOWED_EXTENSIONS override with JSON array."""
        import json
        with patch.dict(os.environ, {"ALLOWED_EXTENSIONS": json.dumps([".csv", ".xlsx", ".parquet"])}, clear=False):
            settings = Settings()
            assert settings.ALLOWED_EXTENSIONS == [".csv", ".xlsx", ".parquet"]
    
    def test_allowed_extensions_validator_comma_separated(self):
        """Test that ALLOWED_EXTENSIONS validator handles comma-separated strings when passed directly."""
        # Test validator directly (bypassing pydantic_settings JSON parsing)
        from app.config import Settings
        settings = Settings.model_validate({"ALLOWED_EXTENSIONS": ".csv,.xlsx,.parquet"})
        assert settings.ALLOWED_EXTENSIONS == [".csv", ".xlsx", ".parquet"]
    
    def test_override_target_column(self):
        """Test DEFAULT_TARGET_COLUMN override from environment."""
        with patch.dict(os.environ, {"DEFAULT_TARGET_COLUMN": "default_flag"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_TARGET_COLUMN == "default_flag"
    
    def test_override_test_size(self):
        """Test DEFAULT_TEST_SIZE override from environment."""
        with patch.dict(os.environ, {"DEFAULT_TEST_SIZE": "0.25"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_TEST_SIZE == 0.25
    
    def test_override_random_state(self):
        """Test RANDOM_STATE override from environment."""
        with patch.dict(os.environ, {"RANDOM_STATE": "123"}, clear=False):
            settings = Settings()
            assert settings.RANDOM_STATE == 123
    
    def test_override_hidden_layers_string(self):
        """Test DEFAULT_HIDDEN_LAYERS override with JSON array."""
        import json
        with patch.dict(os.environ, {"DEFAULT_HIDDEN_LAYERS": json.dumps([64, 32, 16])}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_HIDDEN_LAYERS == [64, 32, 16]
    
    def test_hidden_layers_validator_comma_separated(self):
        """Test that DEFAULT_HIDDEN_LAYERS validator handles comma-separated strings when passed directly."""
        # Test validator directly (bypassing pydantic_settings JSON parsing)
        from app.config import Settings
        settings = Settings.model_validate({"DEFAULT_HIDDEN_LAYERS": "64,32,16"})
        assert settings.DEFAULT_HIDDEN_LAYERS == [64, 32, 16]
    
    def test_override_activation(self):
        """Test DEFAULT_ACTIVATION override from environment."""
        with patch.dict(os.environ, {"DEFAULT_ACTIVATION": "tanh"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_ACTIVATION == "tanh"
    
    def test_override_dropout_rate(self):
        """Test DEFAULT_DROPOUT_RATE override from environment."""
        with patch.dict(os.environ, {"DEFAULT_DROPOUT_RATE": "0.5"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_DROPOUT_RATE == 0.5
    
    def test_override_learning_rate(self):
        """Test DEFAULT_LEARNING_RATE override from environment."""
        with patch.dict(os.environ, {"DEFAULT_LEARNING_RATE": "0.01"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_LEARNING_RATE == 0.01
    
    def test_override_batch_size(self):
        """Test DEFAULT_BATCH_SIZE override from environment."""
        with patch.dict(os.environ, {"DEFAULT_BATCH_SIZE": "512"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_BATCH_SIZE == 512
    
    def test_override_epochs(self):
        """Test DEFAULT_EPOCHS override from environment."""
        with patch.dict(os.environ, {"DEFAULT_EPOCHS": "200"}, clear=False):
            settings = Settings()
            assert settings.DEFAULT_EPOCHS == 200
    
    def test_override_cors_origins_string(self):
        """Test CORS_ORIGINS override with JSON array."""
        import json
        with patch.dict(os.environ, {"CORS_ORIGINS": json.dumps(["http://localhost:3000", "http://localhost:8080"])}, clear=False):
            settings = Settings()
            assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:8080"]
    
    def test_cors_origins_validator_comma_separated(self):
        """Test that CORS_ORIGINS validator handles comma-separated strings when passed directly."""
        # Test validator directly (bypassing pydantic_settings JSON parsing)
        from app.config import Settings
        settings = Settings.model_validate({"CORS_ORIGINS": "http://localhost:3000,http://localhost:8080"})
        assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:8080"]


class TestDirectoryCreation:
    """Test that directories are created automatically."""
    
    def test_upload_dir_created(self):
        """Test that UPLOAD_DIR is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = os.path.join(tmpdir, "test_uploads")
            with patch.dict(os.environ, {"UPLOAD_DIR": upload_dir}, clear=False):
                settings = Settings()
                assert Path(upload_dir).exists()
                assert Path(upload_dir).is_dir()
    
    def test_model_dir_created(self):
        """Test that MODEL_DIR is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "test_models")
            with patch.dict(os.environ, {"MODEL_DIR": model_dir}, clear=False):
                settings = Settings()
                assert Path(model_dir).exists()
                assert Path(model_dir).is_dir()
    
    def test_nested_directories_created(self):
        """Test that nested directory paths are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "level1", "level2", "uploads")
            with patch.dict(os.environ, {"UPLOAD_DIR": nested_dir}, clear=False):
                settings = Settings()
                assert Path(nested_dir).exists()
                assert Path(nested_dir).is_dir()


class TestInvalidValues:
    """Test that invalid values raise ValidationError."""
    
    def test_invalid_test_size_too_large(self):
        """Test that DEFAULT_TEST_SIZE > 1.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_TEST_SIZE": "1.5"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_test_size_negative(self):
        """Test that DEFAULT_TEST_SIZE < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_TEST_SIZE": "-0.1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_dropout_rate_too_large(self):
        """Test that DEFAULT_DROPOUT_RATE > 1.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_DROPOUT_RATE": "1.5"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_dropout_rate_negative(self):
        """Test that DEFAULT_DROPOUT_RATE < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_DROPOUT_RATE": "-0.1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_learning_rate_zero(self):
        """Test that DEFAULT_LEARNING_RATE = 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_LEARNING_RATE": "0.0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_learning_rate_negative(self):
        """Test that DEFAULT_LEARNING_RATE < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_LEARNING_RATE": "-0.001"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_batch_size_zero(self):
        """Test that DEFAULT_BATCH_SIZE = 0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_BATCH_SIZE": "0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_batch_size_negative(self):
        """Test that DEFAULT_BATCH_SIZE < 0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_BATCH_SIZE": "-1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_epochs_zero(self):
        """Test that DEFAULT_EPOCHS = 0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_EPOCHS": "0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_epochs_negative(self):
        """Test that DEFAULT_EPOCHS < 0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_EPOCHS": "-1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_early_stopping_patience_negative(self):
        """Test that DEFAULT_EARLY_STOPPING_PATIENCE < 0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_EARLY_STOPPING_PATIENCE": "-1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_l1_lambda_negative(self):
        """Test that DEFAULT_L1_LAMBDA < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_L1_LAMBDA": "-0.1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_l2_lambda_negative(self):
        """Test that DEFAULT_L2_LAMBDA < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_L2_LAMBDA": "-0.1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_loss_alpha_too_large(self):
        """Test that DEFAULT_LOSS_ALPHA > 1.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_LOSS_ALPHA": "1.5"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_loss_alpha_negative(self):
        """Test that DEFAULT_LOSS_ALPHA < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_LOSS_ALPHA": "-0.1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_auc_gamma_zero(self):
        """Test that DEFAULT_AUC_GAMMA = 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_AUC_GAMMA": "0.0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_auc_gamma_negative(self):
        """Test that DEFAULT_AUC_GAMMA < 0.0 raises ValidationError."""
        with patch.dict(os.environ, {"DEFAULT_AUC_GAMMA": "-1.0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_score_max_zero(self):
        """Test that SCORE_MAX = 0 raises ValidationError."""
        with patch.dict(os.environ, {"SCORE_MAX": "0"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_score_max_negative(self):
        """Test that SCORE_MAX < 0 raises ValidationError."""
        with patch.dict(os.environ, {"SCORE_MAX": "-1"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_hidden_layers_string(self):
        """Test that invalid DEFAULT_HIDDEN_LAYERS string raises ValidationError."""
        # Test validator directly since pydantic_settings will fail JSON parsing first
        from app.config import Settings
        with pytest.raises(ValidationError):
            Settings.model_validate({"DEFAULT_HIDDEN_LAYERS": "not,a,number"})
    
    def test_invalid_max_upload_size_type(self):
        """Test that non-integer MAX_UPLOAD_SIZE_MB raises ValidationError."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_MB": "not_a_number"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()

