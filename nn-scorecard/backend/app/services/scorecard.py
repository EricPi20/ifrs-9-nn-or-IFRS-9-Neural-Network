"""
Scorecard Conversion

This module converts trained neural network models into interpretable
scorecards with a 0-100 scale, where 100 represents the lowest risk (best)
and 0 represents the highest risk (worst).

CRITICAL: 
- Model was trained on NORMALIZED inputs (original values ÷ INPUT_SCALE_FACTOR)
- Original CSV values are: standardized log odds × (-50), range ~[-150, +150]
- Normalized values for training: ~[-3, +3] (better for NN stability)
- Scorecard displays ORIGINAL CSV values (not normalized)
- Weights are adjusted: weight_for_original = weight_normalized / scale_factor

The scorecard calculation is:
- Points_for_bin = Weight_For_Original × Original_Input_Value
- Total_Score = Σ(Points_for_all_features)
- Clamp to [0, 100]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from ..core.constants import INPUT_SCALE_FACTOR, SCORE_MIN, SCORE_MAX

logger = logging.getLogger(__name__)


@dataclass
class BinScore:
    """Score for a single bin of a feature."""
    bin_index: int
    input_value: float      # The standardized log odds × -50 value
    bin_label: str          # Human-readable label (if provided)
    raw_points: float       # Weight × Input_Value (before scaling)
    scaled_points: int      # Final points after scaling to 0-100 range
    count_train: int = 0
    count_test: int = 0
    bad_rate_train: float = 0.0
    bad_rate_test: float = 0.0


@dataclass
class FeatureScore:
    """Complete score info for one feature."""
    feature_name: str
    weight: float                    # Weight adjusted for original scale
    weight_normalized: float         # Weight from model (trained on normalized)
    importance_rank: int
    bins: List[BinScore]
    min_points: int         # Minimum possible points for this feature
    max_points: int         # Maximum possible points for this feature


@dataclass
class Scorecard:
    """Complete scorecard."""
    segment: str
    model_type: str
    score_min: int = 0
    score_max: int = 100
    
    # Scaling factors (for documentation)
    raw_min: float = 0.0    # Min possible raw score (before scaling)
    raw_max: float = 0.0    # Max possible raw score (before scaling)
    scale_factor: float = 1.0        # For converting raw to 0-100
    offset: float = 0.0
    input_scale_factor: float = 50.0  # The normalization factor used
    
    features: List[FeatureScore] = field(default_factory=list)
    
    # Computed properties
    min_possible_score: int = 0
    max_possible_score: int = 100


class ScorecardGenerator:
    """
    Generate scorecard from trained neural network.
    
    IMPORTANT: The model was trained on normalized inputs (÷ scale_factor).
    The scorecard must display ORIGINAL CSV values (× -50) with correctly
    scaled points.
    
    Weight Adjustment:
    - Model learned: output = weight_normalized × input_normalized
    - Where: input_normalized = input_original / scale_factor
    - So: output = weight_normalized × (input_original / scale_factor)
    - Equivalent to: output = (weight_normalized / scale_factor) × input_original
    - Therefore: weight_for_original = weight_normalized / scale_factor
    
    Input: Features where values = standardized log odds × (-50)
    Output: Scorecard with points per bin, total in [0, 100]
    
    Score 100 = Best (lowest risk)
    Score 0 = Worst (highest risk)
    """
    
    def __init__(
        self, 
        score_min: int = SCORE_MIN, 
        score_max: int = SCORE_MAX,
        scale_factor: float = INPUT_SCALE_FACTOR
    ):
        self.score_min = score_min
        self.score_max = score_max
        self.scale_factor = scale_factor
    
    def generate(
        self,
        model: nn.Module,
        feature_names: List[str],
        unique_values_original: Dict[str, List[float]],  # Original CSV values (× -50)
        segment: str,
        X_train_original: np.ndarray = None,  # Original CSV values
        y_train: np.ndarray = None,
        X_test_original: np.ndarray = None,   # Original CSV values
        y_test: np.ndarray = None,
        bin_labels: Dict[str, Dict[float, str]] = None
    ) -> Scorecard:
        """
        Generate scorecard from trained model.
        
        CRITICAL: 
        - Model was trained on NORMALIZED inputs (÷ scale_factor)
        - Scorecard displays ORIGINAL CSV values (× -50)
        - Weights must be adjusted accordingly
        
        Args:
            model: Trained neural network (trained on normalized data)
            feature_names: List of feature names
            unique_values_original: Dict of feature -> unique ORIGINAL values
            segment: Segment name
            X_train_original: Original (unnormalized) training features
            y_train: Training targets
            X_test_original: Original (unnormalized) test features
            y_test: Test targets
            bin_labels: Optional bin label mapping
            
        Returns:
            Scorecard with points calculated for ORIGINAL values
        """
        # Extract weights from model (trained on normalized inputs)
        weights_normalized = self._extract_weights(model, len(feature_names))
        
        # === CRITICAL: Adjust weights for original scale ===
        # Model sees: normalized_input = original_input / scale_factor
        # Model computes: weight_normalized × normalized_input
        # Which equals: weight_normalized × (original_input / scale_factor)
        # For scorecard with original values, we need:
        # weight_for_original = weight_normalized / scale_factor
        weights_for_original = weights_normalized / self.scale_factor
        
        logger.info(f"Weights (normalized): {weights_normalized}")
        logger.info(f"Weights (for original scale): {weights_for_original}")
        logger.info(f"Scale factor: {self.scale_factor}")
        
        # Calculate raw score range using ORIGINAL values
        raw_min = 0.0
        raw_max = 0.0
        
        for i, feat in enumerate(feature_names):
            w = weights_for_original[i]
            values = unique_values_original.get(feat, [0.0])
            contributions = [w * v for v in values]  # v is original (× -50)
            
            raw_min += min(contributions)
            raw_max += max(contributions)
        
        logger.info(f"Raw score range: [{raw_min:.2f}, {raw_max:.2f}]")
        
        # Calculate scaling to map raw scores to [0, 100]
        raw_range = raw_max - raw_min
        if raw_range == 0:
            raw_range = 1.0
        
        score_scale_factor = (self.score_max - self.score_min) / raw_range
        score_offset = self.score_min - (raw_min * score_scale_factor)
        
        logger.info(f"Score scale factor: {score_scale_factor:.6f}")
        logger.info(f"Score offset: {score_offset:.2f}")
        
        # Calculate importance ranking
        weight_abs = np.abs(weights_for_original)
        importance_order = np.argsort(-weight_abs)
        importance_ranks = np.empty_like(importance_order)
        importance_ranks[importance_order] = np.arange(len(weights_for_original)) + 1
        
        # Generate feature scorecards
        feature_scores = []
        
        for i, feat in enumerate(feature_names):
            w = weights_for_original[i]
            values = unique_values_original.get(feat, [0.0])  # Original CSV values
            
            bins = []
            for j, val in enumerate(values):
                # Points using ORIGINAL value and ADJUSTED weight
                raw_points = w * val
                scaled_points = int(round(raw_points * score_scale_factor))
                
                # Bin label
                label = f"Bin {j+1}"
                if bin_labels and feat in bin_labels and val in bin_labels[feat]:
                    label = bin_labels[feat][val]
                else:
                    label = f"Bin {j+1} (value: {val:.1f})"
                
                # Calculate bin statistics using ORIGINAL data
                count_train, bad_rate_train = 0, 0.0
                count_test, bad_rate_test = 0, 0.0
                
                if X_train_original is not None and y_train is not None:
                    mask = np.abs(X_train_original[:, i] - val) < 0.01
                    count_train = int(mask.sum())
                    if count_train > 0:
                        bad_rate_train = float(y_train[mask].mean())
                
                if X_test_original is not None and y_test is not None:
                    mask = np.abs(X_test_original[:, i] - val) < 0.01
                    count_test = int(mask.sum())
                    if count_test > 0:
                        bad_rate_test = float(y_test[mask].mean())
                
                bins.append(BinScore(
                    bin_index=j,
                    input_value=val,            # ORIGINAL CSV value (× -50)
                    bin_label=label,
                    raw_points=raw_points,
                    scaled_points=scaled_points,
                    count_train=count_train,
                    count_test=count_test,
                    bad_rate_train=bad_rate_train,
                    bad_rate_test=bad_rate_test
                ))
            
            bins.sort(key=lambda b: b.input_value)
            bin_points = [b.scaled_points for b in bins]
            
            feature_scores.append(FeatureScore(
                feature_name=feat,
                weight=float(w),                # Adjusted weight for original scale
                weight_normalized=float(weights_normalized[i]),  # Add this field
                importance_rank=int(importance_ranks[i]),
                bins=bins,
                min_points=min(bin_points) if bin_points else 0,
                max_points=max(bin_points) if bin_points else 0
            ))
        
        feature_scores.sort(key=lambda f: f.importance_rank)
        
        # Calculate actual min/max possible scores
        min_possible = sum(f.min_points for f in feature_scores)
        max_possible = sum(f.max_points for f in feature_scores)
        
        min_possible_final = max(self.score_min, int(round(min_possible + score_offset)))
        max_possible_final = min(self.score_max, int(round(max_possible + score_offset)))
        
        return Scorecard(
            segment=segment,
            model_type=self._get_model_type(model),
            score_min=self.score_min,
            score_max=self.score_max,
            raw_min=raw_min,
            raw_max=raw_max,
            scale_factor=score_scale_factor,
            offset=score_offset,
            input_scale_factor=self.scale_factor,  # Store for documentation
            features=feature_scores,
            min_possible_score=min_possible_final,
            max_possible_score=max_possible_final
        )
    
    def _extract_weights(self, model: nn.Module, n_features: int) -> np.ndarray:
        """
        Extract feature weights from model.
        
        For models with skip connection: Use get_feature_weights() method
        For linear model: Use linear layer weights directly
        For neural network: Use first layer weights (input importance)
        """
        # Check if model has get_feature_weights method (for skip connection models)
        if hasattr(model, 'get_feature_weights'):
            try:
                weights = model.get_feature_weights()
                if isinstance(weights, torch.Tensor):
                    return weights.cpu().numpy()
                return weights
            except Exception as e:
                logger.warning(f"Could not use get_feature_weights: {e}, falling back to standard extraction")
        
        # Try to find the first linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_matrix = module.weight.data.cpu().numpy()
                
                if weight_matrix.shape[1] == n_features:
                    # This is the input layer
                    if weight_matrix.shape[0] == 1:
                        # Linear model: single output
                        return weight_matrix.flatten()
                    else:
                        # Neural network: average across hidden neurons
                        # This gives input feature importance
                        return weight_matrix.mean(axis=0)
        
        raise ValueError("Could not extract weights from model")
    
    def _get_model_type(self, model: nn.Module) -> str:
        """Determine model type."""
        layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(layers) == 1:
            return 'linear'
        return 'neural_network'
    
    def calculate_score(
        self,
        scorecard: Scorecard,
        input_values: Dict[str, float]  # ORIGINAL CSV values (× -50)
    ) -> Tuple[int, Dict[str, int]]:
        """
        Calculate score for a single record.
        
        Args:
            scorecard: Generated scorecard
            input_values: Dict of feature_name -> ORIGINAL CSV value (× -50)
            
        Returns:
            Tuple of (total_score, breakdown_by_feature)
        """
        total_score = 0.0
        breakdown = {}
        matched_features = 0
        
        for fs in scorecard.features:
            val = input_values.get(fs.feature_name)
            if val is None:
                # Feature not provided - skip (contributes 0)
                continue
            
            # Find matching bin - try exact match first, then tolerance
            matched_bin = None
            min_diff = float('inf')
            closest_bin = None
            
            for b in fs.bins:
                diff = abs(b.input_value - val)
                if diff < 0.01:  # Exact match with tolerance
                    matched_bin = b
                    break
                if diff < min_diff:
                    min_diff = diff
                    closest_bin = b
            
            if matched_bin:
                # scaled_points are already the final points to sum
                points = matched_bin.scaled_points
                breakdown[fs.feature_name] = int(round(points))
                total_score += points
                matched_features += 1
            elif closest_bin and min_diff < 1.0:  # Close enough match (within 1.0)
                # Use closest bin if very close
                points = closest_bin.scaled_points
                breakdown[fs.feature_name] = int(round(points))
                total_score += points
                matched_features += 1
            else:
                # Value not in bins - interpolate between bins
                if len(fs.bins) > 0:
                    # Find the two bins this value falls between
                    sorted_bins = sorted(fs.bins, key=lambda b: b.input_value)
                    
                    # Check if value is below minimum
                    if val < sorted_bins[0].input_value:
                        points = sorted_bins[0].scaled_points
                    # Check if value is above maximum
                    elif val > sorted_bins[-1].input_value:
                        points = sorted_bins[-1].scaled_points
                    else:
                        # Find the two bins to interpolate between
                        for i in range(len(sorted_bins) - 1):
                            if sorted_bins[i].input_value <= val <= sorted_bins[i + 1].input_value:
                                # Linear interpolation
                                bin1 = sorted_bins[i]
                                bin2 = sorted_bins[i + 1]
                                if bin2.input_value != bin1.input_value:
                                    ratio = (val - bin1.input_value) / (bin2.input_value - bin1.input_value)
                                    points = bin1.scaled_points + ratio * (bin2.scaled_points - bin1.scaled_points)
                                else:
                                    points = bin1.scaled_points
                                break
                        else:
                            # Fallback to first bin
                            points = sorted_bins[0].scaled_points
                else:
                    # No bins available - use feature's min points
                    points = fs.min_points
                
                breakdown[fs.feature_name] = int(round(points))
                total_score += points
                matched_features += 1
        
        # Clamp to valid range [0, 100]
        # scaled_points are already in the 0-100 range, so just round and clamp
        total_score = int(round(total_score))
        total_score = max(self.score_min, min(self.score_max, total_score))
        
        # Warn if no features matched
        if matched_features == 0:
            logger.warning(f"No features matched for scoring. Input features: {list(input_values.keys())}, Scorecard features: {[f.feature_name for f in scorecard.features]}")
        
        return total_score, breakdown
    
    def batch_score(
        self,
        scorecard: Scorecard,
        records: List[Dict[str, float]]
    ) -> List[Tuple[int, Dict[str, int]]]:
        """Score multiple records."""
        return [self.calculate_score(scorecard, r) for r in records]


# Alias for backward compatibility
ScorecardConverter = ScorecardGenerator
