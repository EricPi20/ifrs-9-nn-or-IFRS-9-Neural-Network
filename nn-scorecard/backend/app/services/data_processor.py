"""
Data Validation and Processing for Pre-binned WoE Credit Risk Data

This module handles CSV validation, segment analysis, feature analysis,
and data preparation for credit risk datasets with pre-binned WoE values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from ..core.constants import INPUT_SCALE_FACTOR


class SegmentedDataProcessor:
    """
    Processor for pre-binned WoE credit risk data with segment support.
    
    This class handles:
    1. CSV validation (format, columns, data types)
    2. Segment analysis (counts, bad rates per segment)
    3. Feature analysis (WoE statistics, correlations)
    4. Data preparation (train/test split, feature extraction)
    5. Bin mapping (optional: load bin labels for WoE values)
    
    Attributes:
        target_col: Name of binary target column (default: 'target')
        id_col: Name of unique identifier column (default: 'account_id')
        segment_col: Name of segment column (default: 'segment')
        feature_prefix: Prefix for WoE feature columns (default: 'woe_')
    """
    
    def __init__(
        self,
        target_col: str = 'target',
        id_col: str = 'account_id',
        segment_col: str = 'segment',
        feature_prefix: str = 'woe_',
        scale_factor: float = INPUT_SCALE_FACTOR
    ):
        """
        Initialize the SegmentedDataProcessor.
        
        Args:
            target_col: Name of binary target column (default: 'target')
            id_col: Name of unique identifier column (default: 'account_id')
            segment_col: Name of segment column (default: 'segment')
            feature_prefix: Prefix for WoE feature columns (default: 'woe_')
            scale_factor: Factor to divide inputs by for normalization (default: 50.0)
        """
        self.target_col = target_col
        self.id_col = id_col
        self.segment_col = segment_col
        self.feature_prefix = feature_prefix
        self.scale_factor = scale_factor
        
        # Validate scale_factor
        if scale_factor == 0.0:
            raise ValueError("scale_factor cannot be zero")
        
        # State variables
        self._feature_names: List[str] = []
        self._segments: List[str] = []
        self._bin_mapping: Optional[Dict[str, List[Dict]]] = None
    
    # === VALIDATION METHODS ===
    
    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Validate uploaded CSV data.
        
        Checks:
        1. Required columns exist (target, id)
        2. Target column is binary (0 and 1 only)
        3. ID column has unique values
        4. WoE feature columns are numeric
        5. No entirely null columns
        6. WoE values are in reasonable range (-5 to +5)
        
        Args:
            df: Pandas DataFrame to validate
            
        Returns:
            List of error/warning messages (empty if valid)
        """
        errors = []
        
        # Check target column
        if self.target_col not in df.columns:
            errors.append(f"Missing required column: '{self.target_col}'")
        else:
            unique_targets = df[self.target_col].dropna().unique()
            if not set(unique_targets).issubset({0, 1}):
                errors.append(f"Target column must be binary (0/1). Found: {list(unique_targets)}")
        
        # Check ID column
        if self.id_col not in df.columns:
            errors.append(f"Missing required column: '{self.id_col}'")
        elif df[self.id_col].duplicated().any():
            n_dups = df[self.id_col].duplicated().sum()
            errors.append(f"ID column has {n_dups} duplicate values")
        
        # Check WoE features
        feature_cols = self._get_feature_columns(df)
        if len(feature_cols) == 0:
            errors.append(f"No WoE feature columns found (looking for '{self.feature_prefix}*' prefix)")
        
        for col in feature_cols:
            # Check numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Feature '{col}' must be numeric")
                continue
            
            # Check for all nulls
            if df[col].isna().all():
                errors.append(f"Feature '{col}' is entirely null")
                continue
            
            # Warn about extreme WoE values
            col_min, col_max = df[col].min(), df[col].max()
            if col_min < -5 or col_max > 5:
                errors.append(
                    f"WARNING: Feature '{col}' has extreme WoE values "
                    f"(range: [{col_min:.2f}, {col_max:.2f}]). "
                    f"Typical WoE range is -3 to +3."
                )
        
        return errors
    
    # === SEGMENT METHODS ===
    
    def get_segments(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of unique segments in data.
        
        Returns ['ALL'] if no segment column exists.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of unique segment names, or ['ALL'] if no segment column
        """
        if self.segment_col not in df.columns:
            return ['ALL']
        
        segments = sorted(df[self.segment_col].dropna().unique().tolist())
        self._segments = segments
        return segments
    
    def get_segment_stats(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate statistics for each segment.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of dicts with keys: segment, count, bad_count, bad_rate
        """
        if self.segment_col not in df.columns:
            return [{
                'segment': 'ALL',
                'count': len(df),
                'bad_count': int(df[self.target_col].sum()),
                'bad_rate': float(df[self.target_col].mean())
            }]
        
        stats = []
        for segment in self.get_segments(df):
            seg_df = df[df[self.segment_col] == segment]
            stats.append({
                'segment': segment,
                'count': len(seg_df),
                'bad_count': int(seg_df[self.target_col].sum()),
                'bad_rate': float(seg_df[self.target_col].mean())
            })
        
        return stats
    
    def filter_segment(self, df: pd.DataFrame, segment: str) -> pd.DataFrame:
        """
        Filter DataFrame to specific segment.
        
        Args:
            df: Pandas DataFrame
            segment: Segment name to filter to, or 'ALL' for no filtering
            
        Returns:
            Filtered DataFrame copy
        """
        if segment == 'ALL' or self.segment_col not in df.columns:
            return df.copy()
        return df[df[self.segment_col] == segment].copy()
    
    # === FEATURE ANALYSIS METHODS ===
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of WoE feature columns.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of feature column names
        """
        exclude = [self.id_col, self.segment_col, self.target_col]
        
        # If prefix is set, filter by prefix
        if self.feature_prefix:
            return [c for c in df.columns 
                    if c.startswith(self.feature_prefix) and c not in exclude]
        
        # Otherwise, all numeric columns except excluded
        return [c for c in df.columns 
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    def analyze_features(
        self, 
        df: pd.DataFrame, 
        segment: str = 'ALL'
    ) -> List[Dict]:
        """
        Analyze WoE features for a specific segment.
        
        For each feature, calculates:
        - Number of unique bins (WoE values)
        - List of unique WoE values
        - Min, max, mean WoE
        - Correlation with target
        
        Args:
            df: Pandas DataFrame
            segment: Segment to analyze (default: 'ALL')
            
        Returns:
            List of feature analysis dicts, sorted by absolute correlation
            (most predictive first). Each dict contains:
            - name: Feature name
            - num_bins: Number of unique WoE values
            - bins: List of dicts with 'bin_label' and 'woe_value'
            - min_woe: Minimum WoE value
            - max_woe: Maximum WoE value
            - mean_woe: Mean WoE value
            - target_correlation: Correlation with target variable
        """
        seg_df = self.filter_segment(df, segment)
        feature_cols = self._get_feature_columns(seg_df)
        self._feature_names = feature_cols
        
        analysis = []
        for col in feature_cols:
            unique_woe = sorted(seg_df[col].dropna().unique())
            
            # Get bin labels from mapping if available
            bins = []
            for woe in unique_woe:
                bin_label = self._get_bin_label(col, woe)
                bins.append({
                    'bin_label': bin_label,
                    'woe_value': float(woe)
                })
            
            corr = seg_df[col].corr(seg_df[self.target_col])
            
            analysis.append({
                'name': col,
                'num_bins': len(unique_woe),
                'bins': bins,
                'min_woe': float(seg_df[col].min()),
                'max_woe': float(seg_df[col].max()),
                'mean_woe': float(seg_df[col].mean()),
                'target_correlation': float(corr) if not pd.isna(corr) else 0.0
            })
        
        # Sort by absolute correlation
        analysis.sort(key=lambda x: abs(x['target_correlation']), reverse=True)
        
        return analysis
    
    def get_unique_woe_values(
        self, 
        df: pd.DataFrame, 
        segment: str = 'ALL'
    ) -> Dict[str, List[float]]:
        """
        Get unique WoE values for each feature.
        
        These represent the pre-defined bins from the binning process.
        
        Args:
            df: Pandas DataFrame
            segment: Segment to analyze (default: 'ALL')
            
        Returns:
            Dict mapping feature name to sorted list of WoE values
        """
        seg_df = self.filter_segment(df, segment)
        feature_cols = self._get_feature_columns(seg_df)
        
        return {
            col: sorted(seg_df[col].dropna().unique().tolist())
            for col in feature_cols
        }
    
    # === BIN MAPPING METHODS ===
    
    def load_bin_mapping(self, mapping_df: pd.DataFrame) -> None:
        """
        Load bin labels from a mapping DataFrame.
        
        Expected columns: feature, bin_label, woe_value
        
        Args:
            mapping_df: DataFrame with columns 'feature', 'bin_label', 'woe_value'
        """
        self._bin_mapping = {}
        
        for feature in mapping_df['feature'].unique():
            feature_rows = mapping_df[mapping_df['feature'] == feature]
            self._bin_mapping[feature] = [
                {
                    'bin_label': row['bin_label'],
                    'woe_value': float(row['woe_value'])
                }
                for _, row in feature_rows.iterrows()
            ]
    
    def _get_bin_label(self, feature: str, woe_value: float) -> str:
        """
        Get bin label for a WoE value, or generate a default.
        
        Args:
            feature: Feature name
            woe_value: WoE value to get label for
            
        Returns:
            Bin label string
        """
        if self._bin_mapping and feature in self._bin_mapping:
            for bin_info in self._bin_mapping[feature]:
                if abs(bin_info['woe_value'] - woe_value) < 0.001:
                    return bin_info['bin_label']
        
        # Default label based on WoE value
        if woe_value > 0.5:
            return f'Low Risk (WoE: {woe_value:.2f})'
        elif woe_value > 0:
            return f'Below Avg Risk (WoE: {woe_value:.2f})'
        elif woe_value > -0.5:
            return f'Above Avg Risk (WoE: {woe_value:.2f})'
        else:
            return f'High Risk (WoE: {woe_value:.2f})'
    
    # === DATA PREPARATION METHODS ===
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'target',
        segment: str = None,
        segment_col: str = 'segment',
        scale_factor: float = None
    ) -> Dict:
        """
        Prepare data for neural network training.
        
        IMPORTANT: Normalizes input values by dividing by scale_factor.
        Original CSV values are (standardized log odds × -50).
        Normalized values are in approximately [-3, +3] range.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            segment: Optional segment filter
            segment_col: Segment column name
            scale_factor: Factor to divide inputs by (default: uses instance scale_factor)
            
        Returns:
            Dict with X, y, feature_names, and metadata
        """
        # Use instance scale_factor if not provided
        if scale_factor is None:
            scale_factor = self.scale_factor
        
        # Filter by segment if specified
        if segment and segment != 'ALL' and segment_col in df.columns:
            df = df[df[segment_col] == segment].copy()
        
        # Extract features and target
        X_original = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        
        # Store original unique values per feature (for scorecard display)
        unique_values_original = {}
        for i, col in enumerate(feature_cols):
            unique_values_original[col] = sorted(df[col].dropna().unique().tolist())
        
        # === CRITICAL: Normalize inputs for neural network ===
        # Original values: ~[-150, +150] (std log odds × -50)
        # Normalized values: ~[-3, +3] (suitable for NN)
        X_normalized = X_original / scale_factor
        
        # Store feature statistics for later use
        unique_values_normalized = {
            col: [v / scale_factor for v in vals]
            for col, vals in unique_values_original.items()
        }
        
        return {
            'X': X_normalized,           # Normalized for training
            'X_original': X_original,    # Original for reference
            'y': y,
            'feature_names': feature_cols,
            'unique_values_original': unique_values_original,
            'unique_values_normalized': unique_values_normalized,
            'scale_factor': scale_factor,
            'num_records': len(y),
            'bad_rate': float(y.mean())
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        segment: str = 'ALL',
        selected_features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training.
        
        Args:
            df: Full DataFrame
            segment: Segment to filter to (default: 'ALL')
            selected_features: Features to use (None = all available features)
        
        Returns:
            Tuple of:
            - X: Feature matrix (n_samples, n_features) as float32
            - y: Target vector (n_samples,) as float32
            - feature_names: List of feature names used
        """
        seg_df = self.filter_segment(df, segment)
        
        # Get features
        all_features = self._get_feature_columns(seg_df)
        if selected_features:
            feature_names = [f for f in selected_features if f in all_features]
        else:
            feature_names = all_features
        
        self._feature_names = feature_names
        
        X = seg_df[feature_names].values.astype(np.float32)
        y = seg_df[self.target_col].values.astype(np.float32)
        
        return X, y, feature_names
    
    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.30,
        random_state: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train and test sets (stratified).
        
        NOTE: We use train/test split ONLY, no validation set.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set (default: 0.30)
            random_state: Random seed for reproducibility (default: 42)
        
        Returns:
            Dict with 'train' and 'test' keys, each containing (X, y) tuple
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
    
    # === PROPERTIES ===
    
    @property
    def feature_names(self) -> List[str]:
        """
        Get current feature names.
        
        Returns:
            List of feature names from last analysis/preparation
        """
        return self._feature_names
    
    @property
    def segments(self) -> List[str]:
        """
        Get available segments.
        
        Returns:
            List of segment names
        """
        return self._segments
    
    # === COMPATIBILITY METHODS (for existing API) ===
    
    def validate_and_extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Validate CSV file and extract metadata.
        
        Compatibility method for existing API endpoints.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with keys: rows, columns, segments
        """
        df = pd.read_csv(file_path)
        
        # Validate
        errors = self.validate(df)
        if errors:
            raise ValueError(f"Validation failed: {', '.join(errors)}")
        
        # Extract metadata
        segments = self.get_segments(df)
        
        return {
            'rows': len(df),
            'columns': list(df.columns),
            'segments': segments
        }
    
    def load_data(
        self, 
        file_path: str, 
        segment: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file and prepare for training.
        
        Compatibility method for existing API endpoints.
        
        Args:
            file_path: Path to CSV file
            segment: Segment to filter to (None = 'ALL')
            
        Returns:
            Tuple of (X DataFrame, y Series)
        """
        df = pd.read_csv(file_path)
        
        # Validate
        errors = self.validate(df)
        if errors:
            raise ValueError(f"Validation failed: {', '.join(errors)}")
        
        # Filter segment
        segment_filter = segment if segment else 'ALL'
        seg_df = self.filter_segment(df, segment_filter)
        
        # Get features
        feature_cols = self._get_feature_columns(seg_df)
        self._feature_names = feature_cols
        
        X = seg_df[feature_cols]
        y = seg_df[self.target_col]
        
        return X, y


# Backward compatibility alias
DataProcessor = SegmentedDataProcessor