"""
Pydantic Request/Response Models

This module defines all Pydantic schemas for API request and response validation.
All models include proper validation, field descriptions, and JSON schema generation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


# ============================================================================
# SEGMENT SCHEMAS
# ============================================================================

class SegmentStats(BaseModel):
    """Statistics for a single portfolio segment."""
    
    segment: str = Field(
        ...,
        description="Segment name (e.g., 'CONSUMER', 'SME')"
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of records in this segment"
    )
    bad_count: int = Field(
        ...,
        ge=0,
        description="Number of bad (target=1) records"
    )
    bad_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of bads (0.0 to 1.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment": "CONSUMER",
                "count": 10000,
                "bad_count": 500,
                "bad_rate": 0.05
            }
        }


# ============================================================================
# FEATURE SCHEMAS
# ============================================================================

class FeatureBin(BaseModel):
    """A single bin within a feature."""
    
    bin_label: str = Field(
        ...,
        description="Human-readable label (e.g., 'Poor (500-580)')"
    )
    woe_value: float = Field(
        ...,
        description="WoE value for this bin"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "bin_label": "Poor (500-580)",
                "woe_value": -0.523
            }
        }


class FeatureSummary(BaseModel):
    """Summary statistics for a single WoE feature."""
    
    name: str = Field(
        ...,
        description="Feature name"
    )
    num_bins: int = Field(
        ...,
        ge=2,
        le=6,
        description="Number of unique bins (2-6)"
    )
    bins: List[FeatureBin] = Field(
        ...,
        description="List of bins with labels and WoE values"
    )
    min_woe: float = Field(
        ...,
        description="Minimum WoE value"
    )
    max_woe: float = Field(
        ...,
        description="Maximum WoE value"
    )
    mean_woe: float = Field(
        ...,
        description="Mean WoE value"
    )
    target_correlation: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Correlation with target variable"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "credit_score",
                "num_bins": 4,
                "bins": [
                    {"bin_label": "Very Poor (<500)", "woe_value": -0.823},
                    {"bin_label": "Poor (500-580)", "woe_value": -0.523},
                    {"bin_label": "Good (580-650)", "woe_value": 0.123},
                    {"bin_label": "Excellent (>650)", "woe_value": 0.623}
                ],
                "min_woe": -0.823,
                "max_woe": 0.623,
                "mean_woe": -0.150,
                "target_correlation": 0.45
            }
        }


# ============================================================================
# UPLOAD SCHEMAS
# ============================================================================

class UploadResponse(BaseModel):
    """Response after successful file upload."""
    
    file_id: str = Field(
        ...,
        description="UUID for the uploaded file"
    )
    filename: str = Field(
        ...,
        description="Original filename"
    )
    num_records: int = Field(
        ...,
        ge=0,
        description="Total number of records"
    )
    num_features: int = Field(
        ...,
        ge=0,
        description="Number of WoE features"
    )
    segments: List[str] = Field(
        ...,
        description="List of unique segments"
    )
    segment_stats: List[SegmentStats] = Field(
        ...,
        description="Statistics for each segment"
    )
    feature_summary: List[FeatureSummary] = Field(
        ...,
        description="Summary statistics for each feature"
    )
    target_distribution: Dict[str, Any] = Field(
        ...,
        description="Target distribution (good_count, bad_count, bad_rate)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "portfolio_data.xlsx",
                "num_records": 50000,
                "num_features": 15,
                "segments": ["CONSUMER", "SME", "CORPORATE"],
                "segment_stats": [
                    {
                        "segment": "CONSUMER",
                        "count": 30000,
                        "bad_count": 1500,
                        "bad_rate": 0.05
                    }
                ],
                "feature_summary": [],
                "target_distribution": {
                    "good_count": 47500,
                    "bad_count": 2500,
                    "bad_rate": 0.05
                }
            }
        }


# ============================================================================
# TRAINING CONFIGURATION SCHEMAS
# ============================================================================

class NeuralNetworkConfig(BaseModel):
    """Neural network architecture configuration."""
    
    model_type: str = Field(
        default='neural_network',
        description="Model type: 'linear' or 'neural_network'"
    )
    hidden_layers: List[int] = Field(
        default=[32, 16],
        description="Neurons per layer (empty = linear model)"
    )
    activation: str = Field(
        default='relu',
        description="Activation function"
    )
    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Dropout probability"
    )
    use_batch_norm: bool = Field(
        default=True,
        description="Use batch normalization"
    )
    skip_connection: bool = Field(
        default=False,
        description="Skip connection from input to output layer (residual learning)"
    )
    
    @field_validator('hidden_layers')
    @classmethod
    def validate_layers(cls, v):
        """Validate each layer has positive neurons."""
        if any(n <= 0 for n in v):
            raise ValueError('Neurons per layer must be positive')
        return v
    
    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v):
        """Validate activation function is supported."""
        allowed = ['relu', 'leaky_relu', 'elu', 'selu', 'tanh']
        if v not in allowed:
            raise ValueError(f'Activation must be one of {allowed}')
        return v
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        """Validate model type."""
        allowed = ['linear', 'neural_network']
        if v not in allowed:
            raise ValueError(f'Model type must be one of {allowed}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "neural_network",
                "hidden_layers": [32, 16],
                "activation": "relu",
                "dropout_rate": 0.2,
                "use_batch_norm": True,
                "skip_connection": False
            }
        }


class RegularizationConfig(BaseModel):
    """Regularization settings."""
    
    l1_lambda: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 (Lasso) penalty coefficient"
    )
    l2_lambda: float = Field(
        default=0.01,
        ge=0.0,
        description="L2 (Ridge) penalty coefficient"
    )
    gradient_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping threshold"
    )
    
    @field_validator('l1_lambda', 'l2_lambda')
    @classmethod
    def validate_non_negative(cls, v):
        """Validate regularization is non-negative."""
        if v < 0:
            raise ValueError('Regularization must be non-negative')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "l1_lambda": 0.0,
                "l2_lambda": 0.01,
                "gradient_clip_norm": 1.0
            }
        }


class LossConfig(BaseModel):
    """Loss function configuration for AR optimization."""
    
    loss_type: str = Field(
        default='combined',
        description="Loss type: 'bce', 'pairwise_auc', 'soft_auc', 'wmw', or 'combined'"
    )
    loss_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="BCE weight in combined loss (lower = more AR focus)"
    )
    auc_gamma: float = Field(
        default=2.0,
        gt=0.0,
        description="Sharpness parameter for soft AUC"
    )
    auc_loss_type: str = Field(
        default='pairwise',
        description="AUC surrogate for combined loss: 'pairwise', 'soft', or 'wmw'"
    )
    margin: float = Field(
        default=0.0,
        ge=0.0,
        description="Margin for pairwise/WMW losses (enforces stricter separation between positive and negative samples)"
    )
    
    @field_validator('loss_type')
    @classmethod
    def validate_loss_type(cls, v):
        """Validate loss type is supported."""
        allowed = ['bce', 'pairwise_auc', 'soft_auc', 'wmw', 'combined']
        if v not in allowed:
            raise ValueError(f'Loss type must be one of {allowed}')
        return v
    
    @field_validator('auc_loss_type')
    @classmethod
    def validate_auc_loss_type(cls, v):
        """Validate AUC loss type is supported."""
        allowed = ['pairwise', 'soft', 'wmw']
        if v not in allowed:
            raise ValueError(f'AUC loss type must be one of {allowed}')
        return v
    
    @field_validator('loss_alpha')
    @classmethod
    def validate_alpha(cls, v):
        """Validate alpha is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Alpha must be between 0 and 1')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "loss_type": "combined",
                "loss_alpha": 0.3,
                "auc_gamma": 2.0,
                "auc_loss_type": "pairwise",
                "margin": 0.0
            }
        }


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    
    enabled: bool = Field(
        default=False,
        description="Whether early stopping is enabled"
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Number of epochs to wait for improvement before stopping"
    )
    min_delta: float = Field(
        default=0.001,
        ge=0.0,
        description="Minimum change to qualify as an improvement"
    )
    monitor: str = Field(
        default='test_ar',
        description="Metric to monitor for early stopping (default: 'test_ar')"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001,
                "monitor": "test_ar"
            }
        }


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    
    # Data
    segment: str = Field(
        default='ALL',
        description="Segment to train on ('ALL' for all segments)"
    )
    test_size: float = Field(
        default=0.30,
        ge=0.1,
        le=0.5,
        description="Test set proportion (TRAIN/TEST only, no validation set)"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        le=999999,
        description="Random seed for reproducibility (affects train/test split, weight initialization, shuffling)"
    )
    selected_features: Optional[List[str]] = Field(
        default=None,
        description="Selected features (None = use all features)"
    )
    
    # Neural Network
    network: NeuralNetworkConfig = Field(
        default_factory=NeuralNetworkConfig,
        description="Neural network architecture configuration"
    )
    
    # Regularization
    regularization: RegularizationConfig = Field(
        default_factory=RegularizationConfig,
        description="Regularization settings"
    )
    
    # Loss
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description="Loss function configuration"
    )
    
    # Optimizer
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        le=0.1,
        description="Learning rate"
    )
    batch_size: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="Training batch size"
    )
    epochs: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description="Early stopping configuration"
    )
    
    # Class weights
    use_class_weights: bool = Field(
        default=True,
        description="Auto-balance for imbalanced data using class weights"
    )
    
    @field_validator('test_size')
    @classmethod
    def validate_test_size(cls, v):
        """Validate test size is between 0.1 and 0.5."""
        if not 0.1 <= v <= 0.5:
            raise ValueError('Test size must be between 0.1 and 0.5')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment": "ALL",
                "test_size": 0.30,
                "random_seed": 42,
                "selected_features": None,
                "network": {
                    "model_type": "neural_network",
                    "hidden_layers": [32, 16],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "use_batch_norm": True
                },
                "regularization": {
                    "l1_lambda": 0.0,
                    "l2_lambda": 0.01,
                    "gradient_clip_norm": 1.0
                },
                "loss": {
                    "loss_type": "combined",
                    "loss_alpha": 0.3,
                    "auc_gamma": 2.0,
                    "auc_loss_type": "pairwise",
                    "margin": 0.0
                },
                "learning_rate": 0.001,
                "batch_size": 256,
                "epochs": 100,
                "early_stopping": {
                    "enabled": False,
                    "patience": 10,
                    "min_delta": 0.001,
                    "monitor": "test_ar"
                },
                "use_class_weights": True
            }
        }


class TrainingRequest(BaseModel):
    """Request to start training."""
    
    file_id: str = Field(
        ...,
        description="UUID of the uploaded file to train on"
    )
    config: TrainingConfig = Field(
        ...,
        description="Training configuration"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "config": {
                    "segment": "ALL",
                    "test_size": 0.30,
                    "selected_features": None,
                    "network": {
                        "model_type": "neural_network",
                        "hidden_layers": [32, 16],
                        "activation": "relu",
                        "dropout_rate": 0.2,
                        "use_batch_norm": True
                    },
                    "regularization": {
                        "l1_lambda": 0.0,
                        "l2_lambda": 0.01,
                        "gradient_clip_norm": 1.0
                    },
                    "loss": {
                        "loss_type": "combined",
                        "loss_alpha": 0.3,
                        "auc_gamma": 2.0,
                        "auc_loss_type": "pairwise",
                        "margin": 0.0
                    },
                    "learning_rate": 0.001,
                    "batch_size": 256,
                    "epochs": 100,
                    "early_stopping": {
                        "enabled": False,
                        "patience": 10,
                        "min_delta": 0.001,
                        "monitor": "test_ar"
                    },
                    "use_class_weights": True
                }
            }
        }


# ============================================================================
# TRAINING RESPONSE SCHEMAS
# ============================================================================

class TrainingResponse(BaseModel):
    """Response after starting a training job."""
    
    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    status: str = Field(
        ...,
        description="Job status: 'pending', 'training', 'completed', or 'failed'"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    created_at: datetime = Field(
        ...,
        description="ISO format timestamp of creation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-12345",
                "status": "pending",
                "message": "Training job started",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class TrainingProgress(BaseModel):
    """Current training progress."""
    
    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    status: str = Field(
        ...,
        description="Job status: 'queued', 'training', 'completed', or 'failed'"
    )
    current_epoch: int = Field(
        ...,
        ge=0,
        description="Current training epoch"
    )
    total_epochs: int = Field(
        ...,
        ge=1,
        description="Total number of epochs"
    )
    current_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Latest metrics (AR, AUC, KS, loss)"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message or error description"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-12345",
                "status": "training",
                "current_epoch": 45,
                "total_epochs": 100,
                "current_metrics": {
                    "train_ar": 0.65,
                    "test_ar": 0.62,
                    "train_auc": 0.825,
                    "test_auc": 0.810,
                    "train_ks": 0.45,
                    "test_ks": 0.42,
                    "loss": 0.523
                },
                "message": "Training in progress..."
            }
        }


class EpochMetrics(BaseModel):
    """Metrics for a single training epoch."""
    
    epoch: int = Field(
        ...,
        ge=0,
        description="Epoch number"
    )
    train_loss: float = Field(
        ...,
        ge=0.0,
        description="Training loss"
    )
    test_loss: float = Field(
        ...,
        ge=0.0,
        description="Test loss"
    )
    train_auc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Training AUC-ROC"
    )
    test_auc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Test AUC-ROC"
    )
    train_ar: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Training Accuracy Ratio (Gini = 2*AUC - 1)"
    )
    test_ar: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Test Accuracy Ratio (Gini = 2*AUC - 1)"
    )
    train_ks: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Training Kolmogorov-Smirnov statistic"
    )
    test_ks: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Test Kolmogorov-Smirnov statistic"
    )
    learning_rate: float = Field(
        ...,
        gt=0.0,
        description="Learning rate at this epoch"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "epoch": 50,
                "train_loss": 0.523,
                "test_loss": 0.545,
                "train_auc": 0.825,
                "test_auc": 0.810,
                "train_ar": 0.65,
                "test_ar": 0.62,
                "train_ks": 0.45,
                "test_ks": 0.42,
                "learning_rate": 0.001
            }
        }


class TrainingHistory(BaseModel):
    """Complete training history for documentation."""
    
    epochs: List[EpochMetrics] = Field(
        ...,
        description="Metrics for each epoch"
    )
    best_epoch: int = Field(
        ...,
        ge=0,
        description="Epoch with best test performance"
    )
    total_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total training duration in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "epochs": [],
                "best_epoch": 85,
                "total_duration_seconds": 1234.56
            }
        }


# ============================================================================
# MODEL METRICS SCHEMAS
# ============================================================================

class ModelMetrics(BaseModel):
    """Final model performance metrics on test set."""
    
    auc_roc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Area Under ROC Curve"
    )
    gini_ar: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Gini coefficient / Accuracy Ratio (= 2*AUC - 1)"
    )
    ks_statistic: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Kolmogorov-Smirnov statistic"
    )
    log_loss: float = Field(
        ...,
        ge=0.0,
        description="Logarithmic loss"
    )
    brier_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Brier score (mean squared error of probabilities)"
    )
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification accuracy"
    )
    precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision score"
    )
    recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recall score"
    )
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F1 score"
    )
    ks_decile: int = Field(
        ...,
        ge=1,
        le=10,
        description="Decile where KS statistic occurs"
    )
    cumulative_bad_rate_top_decile: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cumulative bad rate in top decile"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "auc_roc": 0.810,
                "gini_ar": 0.62,
                "ks_statistic": 0.42,
                "log_loss": 0.545,
                "brier_score": 0.125,
                "accuracy": 0.85,
                "precision": 0.72,
                "recall": 0.68,
                "f1_score": 0.70,
                "ks_decile": 1,
                "cumulative_bad_rate_top_decile": 0.25
            }
        }


# ============================================================================
# SCORECARD SCHEMAS
# ============================================================================

class ScorecardBinPoints(BaseModel):
    """Points for a single bin in the scorecard."""
    
    bin_label: str = Field(
        ...,
        description="Human-readable bin label (e.g., 'Very Poor (<500)')"
    )
    woe_value: float = Field(
        ...,
        description="WoE value for this bin"
    )
    points: int = Field(
        ...,
        description="Scorecard points for this bin"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "bin_label": "Very Poor (<500)",
                "woe_value": -0.823,
                "points": 10
            }
        }


class FeatureScorecard(BaseModel):
    """Scorecard for a single feature."""
    
    feature: str = Field(
        ...,
        description="Feature name"
    )
    weight: float = Field(
        ...,
        description="Neural network weight for this feature"
    )
    bins: List[ScorecardBinPoints] = Field(
        ...,
        description="Points per bin"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature": "credit_score",
                "weight": 0.45,
                "bins": [
                    {"bin_label": "Very Poor (<500)", "woe_value": -0.823, "points": 10},
                    {"bin_label": "Poor (500-580)", "woe_value": -0.523, "points": 25},
                    {"bin_label": "Good (580-650)", "woe_value": 0.123, "points": 50},
                    {"bin_label": "Excellent (>650)", "woe_value": 0.623, "points": 75}
                ]
            }
        }


class Scorecard(BaseModel):
    """Complete scorecard output."""
    
    segment: str = Field(
        ...,
        description="Portfolio segment"
    )
    base_points: int = Field(
        ...,
        description="Base points (intercept)"
    )
    features: List[FeatureScorecard] = Field(
        ...,
        description="Scorecard for each feature"
    )
    score_range: Tuple[int, int] = Field(
        ...,
        description="Score range tuple (min, max) - scale is 0-100 where 100 = best"
    )
    total_min_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Minimum possible score"
    )
    total_max_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Maximum possible score"
    )
    
    @field_validator('score_range')
    @classmethod
    def validate_score_range(cls, v):
        """Validate score range is (0, 100)."""
        if v != (0, 100):
            raise ValueError('Score range must be (0, 100) where 100 = best (lowest risk)')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment": "CONSUMER",
                "base_points": 50,
                "features": [
                    {
                        "feature": "credit_score",
                        "weight": 0.45,
                        "bins": [
                            {"bin_label": "Very Poor (<500)", "woe_value": -0.823, "points": 10},
                            {"bin_label": "Poor (500-580)", "woe_value": -0.523, "points": 25}
                        ]
                    }
                ],
                "score_range": [0, 100],
                "total_min_score": 0,
                "total_max_score": 100
            }
        }


# ============================================================================
# RESULTS SCHEMAS
# ============================================================================

class ScorecardResults(BaseModel):
    """Complete results after training."""
    
    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    segment: str = Field(
        ...,
        description="Portfolio segment"
    )
    status: str = Field(
        ...,
        description="Job status"
    )
    created_at: str = Field(
        ...,
        description="ISO format timestamp of creation"
    )
    config: TrainingConfig = Field(
        ...,
        description="Training configuration used"
    )
    metrics: ModelMetrics = Field(
        ...,
        description="Final model performance metrics"
    )
    scorecard: Scorecard = Field(
        ...,
        description="Complete scorecard"
    )
    training_history: TrainingHistory = Field(
        ...,
        description="Complete training history"
    )
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Feature importance scores"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-12345",
                "segment": "CONSUMER",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "config": {},
                "metrics": {},
                "scorecard": {},
                "training_history": {},
                "feature_importance": {
                    "credit_score": 0.45,
                    "debt_to_income": 0.32,
                    "employment_years": 0.23
                }
            }
        }


# ============================================================================
# SCORING SCHEMAS
# ============================================================================

class ScoreRequest(BaseModel):
    """Request to score one or more records."""
    
    records: List[Dict[str, float]] = Field(
        ...,
        min_length=1,
        description="List of records, each as {feature: woe_value}"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "records": [
                    {
                        "credit_score": 0.623,
                        "debt_to_income": -0.234,
                        "employment_years": 0.456
                    },
                    {
                        "credit_score": -0.523,
                        "debt_to_income": -0.823,
                        "employment_years": -0.123
                    }
                ]
            }
        }


class ScoreBreakdown(BaseModel):
    """Point breakdown for a single record."""
    
    base_points: int = Field(
        ...,
        description="Base points (intercept)"
    )
    feature_points: Dict[str, int] = Field(
        ...,
        description="Points contribution per feature {feature: points}"
    )
    total_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Total scorecard score (0-100, 100 = best)"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of default (PD)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "base_points": 50,
                "feature_points": {
                    "credit_score": 25,
                    "debt_to_income": 15,
                    "employment_years": 10
                },
                "total_score": 75,
                "probability": 0.15
            }
        }


class ScoreResponse(BaseModel):
    """Response with scores and breakdowns."""
    
    scores: List[ScoreBreakdown] = Field(
        ...,
        description="Score breakdown for each input record"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "scores": [
                    {
                        "base_points": 50,
                        "feature_points": {
                            "credit_score": 25,
                            "debt_to_income": 15,
                            "employment_years": 10
                        },
                        "total_score": 75,
                        "probability": 0.15
                    }
                ]
            }
        }


# ============================================================================
# RESULTS SCHEMAS (for results router)
# ============================================================================

class ModelResults(BaseModel):
    """Model evaluation results."""
    
    job_id: str = Field(..., description="Job identifier")
    segment: Optional[str] = Field(None, description="Segment name")
    train_ar: float = Field(..., description="Training Accuracy Ratio")
    test_ar: float = Field(..., description="Test Accuracy Ratio")
    train_auc: float = Field(..., description="Training AUC-ROC")
    test_auc: float = Field(..., description="Test AUC-ROC")
    train_ks: float = Field(..., description="Training KS statistic")
    test_ks: float = Field(..., description="Test KS statistic")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance")
    training_time: float = Field(..., description="Training time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


class ScorecardResponse(BaseModel):
    """Scorecard response."""
    
    job_id: str = Field(..., description="Job identifier")
    segment: Optional[str] = Field(None, description="Segment name")
    scorecard_data: Dict[str, Any] = Field(..., description="Scorecard data")
    score_range: Dict[str, int] = Field(..., description="Score range")
    created_at: datetime = Field(..., description="Creation timestamp")
