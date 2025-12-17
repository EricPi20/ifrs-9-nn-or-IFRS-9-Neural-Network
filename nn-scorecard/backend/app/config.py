"""
Configuration and Settings

This module contains comprehensive application configuration using Pydantic Settings
for environment variable management. All configurable parameters for the application
are defined here with sensible defaults.
"""

from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ============================================================================
    # FILE STORAGE SETTINGS
    # ============================================================================
    
    UPLOAD_DIR: Path = Field(
        default=Path("./data/uploads"),
        description="Path for uploaded CSV files"
    )
    
    MODEL_DIR: Path = Field(
        default=Path("./data/models"),
        description="Path for saved model checkpoints"
    )
    
    EXPORT_DIR: Path = Field(
        default=Path("./data/exports"),
        description="Path for exported reports and files"
    )
    
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=500,
        description="Maximum file size in megabytes"
    )
    
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=[".csv"],
        description="List of allowed file extensions"
    )
    
    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string or return list as-is."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",") if ext.strip()]
        return v
    
    # ============================================================================
    # DATA PROCESSING SETTINGS
    # ============================================================================
    
    DEFAULT_TARGET_COLUMN: str = Field(
        default="target",
        description="Name of target column for model training"
    )
    
    DEFAULT_ID_COLUMN: str = Field(
        default="account_id",
        description="Name of ID column for account identification"
    )
    
    DEFAULT_SEGMENT_COLUMN: str = Field(
        default="segment",
        description="Name of segment column for data segmentation"
    )
    
    DEFAULT_TEST_SIZE: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for test split (0.0 to 1.0)"
    )
    
    RANDOM_STATE: int = Field(
        default=42,
        description="Random seed for reproducibility across data splits and training"
    )
    
    # ============================================================================
    # NEURAL NETWORK DEFAULTS
    # ============================================================================
    
    DEFAULT_HIDDEN_LAYERS: List[int] = Field(
        default=[32, 16],
        description="Default neural network architecture (list of hidden layer sizes)"
    )
    
    @field_validator("DEFAULT_HIDDEN_LAYERS", mode="before")
    @classmethod
    def parse_hidden_layers(cls, v: Union[str, List[int]]) -> List[int]:
        """Parse comma-separated string of integers or return list as-is."""
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v
    
    DEFAULT_ACTIVATION: str = Field(
        default="relu",
        description="Default activation function for hidden layers"
    )
    
    DEFAULT_DROPOUT_RATE: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Default dropout rate for regularization (0.0 to 1.0)"
    )
    
    DEFAULT_LEARNING_RATE: float = Field(
        default=0.001,
        gt=0.0,
        description="Default learning rate for optimizer"
    )
    
    DEFAULT_BATCH_SIZE: int = Field(
        default=256,
        gt=0,
        description="Default batch size for training"
    )
    
    DEFAULT_EPOCHS: int = Field(
        default=100,
        gt=0,
        description="Default maximum number of training epochs"
    )
    
    DEFAULT_EARLY_STOPPING_PATIENCE: int = Field(
        default=15,
        ge=0,
        description="Default patience for early stopping (epochs to wait before stopping)"
    )
    
    # ============================================================================
    # REGULARIZATION DEFAULTS
    # ============================================================================
    
    DEFAULT_L1_LAMBDA: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 regularization strength (Lasso regularization)"
    )
    
    DEFAULT_L2_LAMBDA: float = Field(
        default=0.01,
        ge=0.0,
        description="L2 regularization strength (Ridge regularization)"
    )
    
    DEFAULT_USE_BATCH_NORM: bool = Field(
        default=True,
        description="Whether to use batch normalization by default"
    )
    
    # ============================================================================
    # LOSS FUNCTION DEFAULTS
    # ============================================================================
    
    DEFAULT_LOSS_TYPE: str = Field(
        default="combined",
        description="Default loss function type (e.g., 'bce', 'focal', 'combined')"
    )
    
    DEFAULT_LOSS_ALPHA: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="BCE weight in combined loss function (0.0 to 1.0)"
    )
    
    DEFAULT_AUC_GAMMA: float = Field(
        default=2.0,
        gt=0.0,
        description="Gamma parameter for soft AUC loss function"
    )
    
    # ============================================================================
    # SCORECARD SCALING
    # ============================================================================
    
    SCORE_MIN: int = Field(
        default=0,
        description="Minimum score in the scorecard scale"
    )
    
    SCORE_MAX: int = Field(
        default=100,
        gt=0,
        description="Maximum score in the scorecard scale"
    )
    
    # ============================================================================
    # API SETTINGS
    # ============================================================================
    
    API_PREFIX: str = Field(
        default="/api",
        description="API route prefix for all endpoints"
    )
    
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173"],
        description="Allowed CORS origins for cross-origin requests"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string or return list as-is."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @model_validator(mode="after")
    def create_directories(self) -> "Settings":
        """
        Create necessary directories if they don't exist.
        
        This validator runs after all fields are validated and ensures
        that UPLOAD_DIR and MODEL_DIR directories are created.
        """
        # Convert to Path if they're strings (from env vars)
        upload_dir = Path(self.UPLOAD_DIR)
        model_dir = Path(self.MODEL_DIR)
        export_dir = Path(self.EXPORT_DIR)
        
        # Create directories if they don't exist
        upload_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the fields to be Path objects
        self.UPLOAD_DIR = upload_dir
        self.MODEL_DIR = model_dir
        self.EXPORT_DIR = export_dir
        
        return self
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MAX_UPLOAD_SIZE_MB to bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# Global settings instance for import throughout the app
settings = Settings()
