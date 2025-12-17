"""
Custom PyTorch Loss Functions for AR (Accuracy Ratio / Gini) Optimization

This module provides differentiable surrogates for AUC optimization, which is
essential for maximizing Accuracy Ratio (AR) / Gini coefficient in credit risk models.

Background:
AR = Gini = 2 × AUC - 1

The challenge is that AUC is based on ranking (counting concordant pairs),
which involves indicator functions that are non-differentiable. We need
differentiable surrogates to optimize AR with gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Optional


class PairwiseAUCLoss(nn.Module):
    """
    Pairwise ranking loss that approximates AUC optimization.
    
    For each pair of (positive sample, negative sample):
    - We want score(positive) > score(negative)
    - Loss = log(1 + exp(-(s_pos - s_neg + margin)))
    
    This is the RankNet-style pairwise logistic loss.
    It's the most stable and commonly used approach.
    
    Mathematical formulation:
    L = (1 / |P| * |N|) * Σ_{i in P} Σ_{j in N} log(1 + exp(-(s_i - s_j + margin)))
    
    where:
    - P = set of positive samples (bad=1)
    - N = set of negative samples (good=0)
    - s_i, s_j = predicted scores
    - margin = minimum desired score difference
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Initialize Pairwise AUC Loss.
        
        Args:
            margin: Minimum desired score difference between positive and negative samples.
                   Higher margin enforces stronger separation.
        """
        super(PairwiseAUCLoss, self).__init__()
        self.margin = margin
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate pairwise AUC loss.
        
        Args:
            y_pred: Predicted scores/probabilities, shape (batch_size,) or (batch_size, 1)
            y_true: True binary labels (1 for bad/default, 0 for good), shape (batch_size,)
        
        Returns:
            Mean pairwise loss value
        """
        # Flatten predictions to 1D
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # Separate positive (bad=1) and negative (good=0) samples
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)
        
        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]
        
        # Handle edge cases
        n_pos = pos_scores.shape[0]
        n_neg = neg_scores.shape[0]
        
        if n_pos == 0 or n_neg == 0:
            # No pairs to compare, return zero loss
            # Use y_pred.sum() * 0.0 to maintain gradient flow
            return y_pred.sum() * 0.0
        
        # Create all pairs: (pos_score, neg_score)
        # Shape: (n_pos, n_neg)
        pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)
        
        # Pairwise differences: s_pos - s_neg
        # Shape: (n_pos, n_neg)
        score_diff = pos_expanded - neg_expanded
        
        # RankNet loss: log(1 + exp(-(s_pos - s_neg + margin)))
        # This encourages s_pos > s_neg + margin
        loss = F.softplus(-(score_diff + self.margin))
        
        # Return mean over all pairs
        return loss.mean()


class SoftAUCLoss(nn.Module):
    """
    Soft AUC approximation using sigmoid function.
    
    True AUC = (1/n_pos*n_neg) * Σ Σ I(s_pos > s_neg)
    
    We approximate the indicator I with sigmoid:
    Soft AUC ≈ (1/n_pos*n_neg) * Σ Σ σ(γ * (s_pos - s_neg))
    
    gamma controls sharpness:
    - Higher gamma = closer to true AUC but harder to optimize
    - Lower gamma = smoother but less accurate
    
    Mathematical formulation:
    Soft_AUC = (1 / |P| * |N|) * Σ_{i in P} Σ_{j in N} σ(γ * (s_i - s_j))
    
    Loss = 1 - Soft_AUC (since we minimize loss, we want to maximize AUC)
    
    where:
    - σ(x) = sigmoid(x) = 1 / (1 + exp(-x))
    - γ = sharpness parameter
    """
    
    def __init__(self, gamma: float = 2.0):
        """
        Initialize Soft AUC Loss.
        
        Args:
            gamma: Sharpness parameter for sigmoid approximation.
                  Higher values make the approximation closer to the true indicator function.
                  Typical range: 1.0 to 10.0
        """
        super(SoftAUCLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate soft AUC loss.
        
        Args:
            y_pred: Predicted scores/probabilities, shape (batch_size,) or (batch_size, 1)
            y_true: True binary labels (1 for bad/default, 0 for good), shape (batch_size,)
        
        Returns:
            Loss value (1 - soft_auc)
        """
        # Flatten predictions to 1D
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # Separate positive (bad=1) and negative (good=0) samples
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)
        
        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]
        
        # Handle edge cases
        n_pos = pos_scores.shape[0]
        n_neg = neg_scores.shape[0]
        
        if n_pos == 0 or n_neg == 0:
            # No pairs to compare, return zero loss
            # Use y_pred.sum() * 0.0 to maintain gradient flow
            return y_pred.sum() * 0.0
        
        # Create all pairs: (pos_score, neg_score)
        # Shape: (n_pos, n_neg)
        pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)
        
        # Pairwise differences: s_pos - s_neg
        # Shape: (n_pos, n_neg)
        score_diff = pos_expanded - neg_expanded
        
        # Apply sigmoid with gamma scaling: σ(γ * (s_pos - s_neg))
        # This approximates I(s_pos > s_neg)
        soft_indicators = torch.sigmoid(self.gamma * score_diff)
        
        # Soft AUC = mean of soft indicators
        soft_auc = soft_indicators.mean()
        
        # Return 1 - soft_auc (since we minimize loss)
        return 1.0 - soft_auc


class WMWLoss(nn.Module):
    """
    Wilcoxon-Mann-Whitney (WMW) loss with polynomial penalty.
    
    Instead of I(s_pos > s_neg), we penalize violations:
    Loss = max(0, s_neg - s_pos + margin)^p
    
    p controls the penalty shape:
    - p=1: linear penalty (hinge loss)
    - p=2: quadratic (default, smooth gradients)
    - p>2: stronger penalty for large violations
    
    Mathematical formulation:
    L = (1 / |P| * |N|) * Σ_{i in P} Σ_{j in N} max(0, s_j - s_i + margin)^p
    
    where:
    - P = set of positive samples (bad=1)
    - N = set of negative samples (good=0)
    - margin = minimum desired score difference
    - p = penalty power
    """
    
    def __init__(self, margin: float = 0.0, p: float = 2.0):
        """
        Initialize WMW Loss.
        
        Args:
            margin: Minimum desired score difference between positive and negative samples.
            p: Power for penalty function. p=2 gives smooth quadratic penalty.
        """
        super(WMWLoss, self).__init__()
        self.margin = margin
        self.p = p
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate WMW loss.
        
        Args:
            y_pred: Predicted scores/probabilities, shape (batch_size,) or (batch_size, 1)
            y_true: True binary labels (1 for bad/default, 0 for good), shape (batch_size,)
        
        Returns:
            Mean WMW loss value
        """
        # Flatten predictions to 1D
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # Separate positive (bad=1) and negative (good=0) samples
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)
        
        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]
        
        # Handle edge cases
        n_pos = pos_scores.shape[0]
        n_neg = neg_scores.shape[0]
        
        if n_pos == 0 or n_neg == 0:
            # No pairs to compare, return zero loss
            # Use y_pred.sum() * 0.0 to maintain gradient flow
            return y_pred.sum() * 0.0
        
        # Create all pairs: (pos_score, neg_score)
        # Shape: (n_pos, n_neg)
        pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)
        
        # Violation: s_neg - s_pos + margin
        # We want s_pos > s_neg + margin, so violations are when s_neg >= s_pos - margin
        # Shape: (n_pos, n_neg)
        violations = neg_expanded - pos_expanded + self.margin
        
        # Penalty: max(0, violation)^p
        # Only penalize when violation > 0 (i.e., when ranking is wrong)
        penalty = torch.clamp(violations, min=0.0) ** self.p
        
        # Return mean over all pairs
        return penalty.mean()


class CombinedLoss(nn.Module):
    """
    Combines BCE loss (for calibration) with AUC surrogate (for discrimination).
    
    Total Loss = α * BCE + (1-α) * AUC_Loss
    
    Rationale:
    - BCE ensures probabilities are well-calibrated for PD estimation
    - AUC surrogate maximizes discrimination (AR/Gini)
    - α = 0.3 gives 70% weight to AR optimization (recommended)
    
    This hybrid approach balances:
    1. Probability calibration (BCE)
    2. Ranking quality / discrimination (AUC surrogate)
    """
    
    def __init__(
        self, 
        alpha: float = 0.3,
        auc_loss_type: str = 'pairwise',  # 'pairwise', 'soft', 'wmw'
        gamma: float = 2.0,
        margin: float = 0.0
    ):
        """
        Initialize Combined Loss.
        
        Args:
            alpha: Weight for BCE component (0.0 to 1.0).
                  Lower alpha = more focus on AR optimization.
                  Recommended: 0.3 (70% weight to AR)
            auc_loss_type: Type of AUC surrogate to use.
                          Options: 'pairwise', 'soft', 'wmw'
            gamma: Sharpness parameter for soft AUC (only used if auc_loss_type='soft')
            margin: Margin parameter for pairwise/WMW losses
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        
        # Initialize BCE loss
        self.bce_loss = nn.BCELoss()
        
        # Initialize AUC surrogate loss
        if auc_loss_type == 'pairwise':
            self.auc_loss = PairwiseAUCLoss(margin=margin)
        elif auc_loss_type == 'soft':
            self.auc_loss = SoftAUCLoss(gamma=gamma)
        elif auc_loss_type == 'wmw':
            self.auc_loss = WMWLoss(margin=margin, p=2.0)
        else:
            raise ValueError(f"Unknown AUC loss type: {auc_loss_type}. "
                           f"Must be one of: 'pairwise', 'soft', 'wmw'")
        
        self.auc_loss_type = auc_loss_type
    
    def forward(
        self, 
        y_pred: Tensor, 
        y_true: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate combined loss.
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size,) or (batch_size, 1)
            y_true: True binary labels (1 for bad/default, 0 for good), shape (batch_size,)
        
        Returns:
            Tuple of (total_loss, breakdown_dict)
            breakdown_dict contains: 'bce', 'auc', 'total'
        """
        # Ensure predictions are in [0, 1] range for BCE
        y_pred_clamped = torch.clamp(y_pred.squeeze(), min=0.0, max=1.0)
        y_true_flat = y_true.squeeze()
        
        # Calculate BCE loss
        bce = self.bce_loss(y_pred_clamped, y_true_flat)
        
        # Calculate AUC surrogate loss
        auc = self.auc_loss(y_pred, y_true)
        
        # Combined loss
        total = self.alpha * bce + (1.0 - self.alpha) * auc
        
        # Create breakdown dictionary
        breakdown = {
            'bce': bce.item(),
            'auc': auc.item(),
            'total': total.item()
        }
        
        return total, breakdown


class EfficientPairSampler:
    """
    For large datasets, computing all pairs is O(n_pos * n_neg).
    This sampler randomly selects a subset of pairs for efficiency.
    
    Usage:
        sampler = EfficientPairSampler(max_pairs=10000)
        pos_scores, neg_scores = sampler.sample(y_pred, y_true)
        # Then compute loss on sampled pairs only
    """
    
    def __init__(self, max_pairs: int = 10000):
        """
        Initialize Efficient Pair Sampler.
        
        Args:
            max_pairs: Maximum number of pairs to sample.
                     If n_pos * n_neg <= max_pairs, all pairs are used.
        """
        self.max_pairs = max_pairs
    
    def sample(
        self, 
        y_pred: Tensor, 
        y_true: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample pairs of (positive, negative) scores.
        
        Args:
            y_pred: Predicted scores, shape (batch_size,) or (batch_size, 1)
            y_true: True binary labels, shape (batch_size,)
        
        Returns:
            Tuple of (pos_scores, neg_scores) where each is a tensor of sampled scores.
            The number of pairs is min(n_pos * n_neg, max_pairs).
        """
        # Flatten predictions to 1D
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # Separate positive and negative samples
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)
        
        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]
        
        n_pos = pos_scores.shape[0]
        n_neg = neg_scores.shape[0]
        
        # Handle edge cases
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([], device=y_pred.device), torch.tensor([], device=y_pred.device)
        
        total_pairs = n_pos * n_neg
        
        if total_pairs <= self.max_pairs:
            # Use all pairs
            pos_expanded = pos_scores.unsqueeze(1).expand(n_pos, n_neg)  # (n_pos, n_neg)
            neg_expanded = neg_scores.unsqueeze(0).expand(n_pos, n_neg)  # (n_pos, n_neg)
            
            pos_sampled = pos_expanded.flatten()
            neg_sampled = neg_expanded.flatten()
        else:
            # Sample random pairs
            # Generate random indices
            pos_indices = torch.randint(0, n_pos, (self.max_pairs,), device=y_pred.device)
            neg_indices = torch.randint(0, n_neg, (self.max_pairs,), device=y_pred.device)
            
            pos_sampled = pos_scores[pos_indices]
            neg_sampled = neg_scores[neg_indices]
        
        return pos_sampled, neg_sampled


# Backward compatibility: Keep old ARLoss and RankLoss classes
class ARLoss(nn.Module):
    """
    Legacy AR loss function (kept for backward compatibility).
    
    This is a simplified version. For better performance, use CombinedLoss.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize AR loss.
        
        Args:
            alpha: Weight for AR component vs standard BCE
        """
        super(ARLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        # Use pairwise AUC as AR surrogate
        self.ar_loss = PairwiseAUCLoss(margin=0.0)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate AR-optimized loss.
        
        Args:
            predictions: Model predictions (probabilities)
            targets: True binary labels
            
        Returns:
            Combined loss value
        """
        # Standard binary cross-entropy
        bce = self.bce_loss(predictions.squeeze(), targets.squeeze())
        
        # AR surrogate using pairwise AUC
        ar_component = self.ar_loss(predictions, targets)
        
        return bce + self.alpha * ar_component


class RankLoss(nn.Module):
    """
    Legacy rank-based loss function (kept for backward compatibility).
    
    This is equivalent to PairwiseAUCLoss. Use PairwiseAUCLoss for new code.
    """
    
    def __init__(self, margin: float = 0.0):
        """Initialize rank loss."""
        super(RankLoss, self).__init__()
        self.pairwise_loss = PairwiseAUCLoss(margin=margin)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate rank-based loss.
        
        Args:
            predictions: Model predictions
            targets: True binary labels
            
        Returns:
            Rank loss value
        """
        return self.pairwise_loss(predictions, targets)


def create_loss_function(config) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: LossConfig object (from schemas.py) or dict with loss configuration.
                Expected fields:
                - loss_type: 'bce', 'pairwise_auc', 'soft_auc', 'wmw', or 'combined'
                - loss_alpha: BCE weight for combined loss (default: 0.3)
                - auc_gamma: Gamma for soft AUC (default: 2.0)
    
    Returns:
        PyTorch loss module
    
    Examples:
        >>> from app.models.schemas import LossConfig
        >>> config = LossConfig(loss_type='combined', loss_alpha=0.3)
        >>> loss_fn = create_loss_function(config)
        
        >>> # Or with dict
        >>> loss_fn = create_loss_function({'loss_type': 'pairwise_auc'})
    """
    # Handle both dict and object configs
    if isinstance(config, dict):
        loss_type = config.get('loss_type', 'bce')
        loss_alpha = config.get('loss_alpha', 0.3)
        auc_gamma = config.get('auc_gamma', 2.0)
        margin = config.get('margin', 0.0)
    else:
        # Assume it's a Pydantic model or object with attributes
        loss_type = getattr(config, 'loss_type', 'bce')
        loss_alpha = getattr(config, 'loss_alpha', 0.3)
        auc_gamma = getattr(config, 'auc_gamma', 2.0)
        margin = getattr(config, 'margin', 0.0)
    
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'pairwise_auc':
        return PairwiseAUCLoss(margin=margin)
    elif loss_type == 'soft_auc':
        return SoftAUCLoss(gamma=auc_gamma)
    elif loss_type == 'wmw':
        return WMWLoss(margin=margin, p=2.0)
    elif loss_type == 'combined':
        # Determine AUC surrogate type from config if available
        auc_loss_type = 'pairwise'  # default
        if isinstance(config, dict):
            auc_loss_type = config.get('auc_loss_type', 'pairwise')
        else:
            auc_loss_type = getattr(config, 'auc_loss_type', 'pairwise')
        
        return CombinedLoss(
            alpha=loss_alpha,
            auc_loss_type=auc_loss_type,
            gamma=auc_gamma,
            margin=margin
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                       f"Must be one of: 'bce', 'pairwise_auc', 'soft_auc', 'wmw', 'combined'")
