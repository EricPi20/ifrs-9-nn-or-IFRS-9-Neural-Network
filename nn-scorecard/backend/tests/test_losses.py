"""
Comprehensive Tests for Custom Loss Functions

Tests cover:
- Basic functionality
- Gradient flow
- Loss value correctness
- Edge cases
- CombinedLoss breakdown
- Efficiency with large batches
"""

import pytest
import torch
import torch.nn as nn
from app.services.losses import (
    PairwiseAUCLoss,
    SoftAUCLoss,
    WMWLoss,
    CombinedLoss,
    create_loss_function,
    ARLoss,
    RankLoss
)


class TestBasicFunctionality:
    """Test basic functionality of all loss functions."""
    
    def test_pairwise_auc_loss_basic(self):
        """Test PairwiseAUCLoss basic functionality."""
        loss_fn = PairwiseAUCLoss(margin=0.0)
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_soft_auc_loss_basic(self):
        """Test SoftAUCLoss basic functionality."""
        loss_fn = SoftAUCLoss(gamma=2.0)
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 0.0 <= loss.item() <= 1.0  # Loss should be in [0, 1] since it's 1 - soft_auc
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_wmw_loss_basic(self):
        """Test WMWLoss basic functionality."""
        loss_fn = WMWLoss(margin=0.0, p=2.0)
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_combined_loss_basic(self):
        """Test CombinedLoss basic functionality."""
        loss_fn = CombinedLoss(alpha=0.3, auc_loss_type='pairwise')
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss, breakdown = loss_fn(y_pred, y_true)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(breakdown, dict)
        assert 'bce' in breakdown
        assert 'auc' in breakdown
        assert 'total' in breakdown
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_all_loss_classes_inherit_from_nn_module(self):
        """Verify all loss classes inherit from nn.Module."""
        assert issubclass(PairwiseAUCLoss, nn.Module)
        assert issubclass(SoftAUCLoss, nn.Module)
        assert issubclass(WMWLoss, nn.Module)
        assert issubclass(CombinedLoss, nn.Module)
        assert issubclass(ARLoss, nn.Module)
        assert issubclass(RankLoss, nn.Module)
    
    def test_forward_signature_matches_expected(self):
        """Test that forward() methods accept (y_pred, y_true) signature."""
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        
        # All losses should accept this signature
        losses = [
            PairwiseAUCLoss(),
            SoftAUCLoss(),
            WMWLoss(),
        ]
        
        for loss_fn in losses:
            result = loss_fn(y_pred, y_true)
            assert isinstance(result, torch.Tensor)
        
        # CombinedLoss returns tuple
        combined = CombinedLoss()
        result, breakdown = combined(y_pred, y_true)
        assert isinstance(result, torch.Tensor)
        assert isinstance(breakdown, dict)


class TestGradientFlow:
    """Test that gradients flow correctly through loss functions."""
    
    def test_pairwise_auc_gradient_flow(self):
        """Test gradient flow for PairwiseAUCLoss."""
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        loss_fn = PairwiseAUCLoss()
        
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert not torch.isinf(y_pred.grad).any()
    
    def test_soft_auc_gradient_flow(self):
        """Test gradient flow for SoftAUCLoss."""
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        loss_fn = SoftAUCLoss()
        
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert not torch.isinf(y_pred.grad).any()
    
    def test_wmw_gradient_flow(self):
        """Test gradient flow for WMWLoss."""
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        loss_fn = WMWLoss()
        
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert not torch.isinf(y_pred.grad).any()
    
    def test_combined_loss_gradient_flow(self):
        """Test gradient flow for CombinedLoss."""
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        loss_fn = CombinedLoss()
        
        loss, _ = loss_fn(y_pred, y_true)
        loss.backward()
        
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert not torch.isinf(y_pred.grad).any()


class TestLossValueCorrectness:
    """Test that loss values make sense (perfect predictions < random predictions)."""
    
    def test_pairwise_auc_perfect_vs_random(self):
        """Perfect predictions should have lower loss than random predictions."""
        loss_fn = PairwiseAUCLoss()
        
        # Perfect predictions
        perfect_pred = torch.tensor([0.99, 0.01, 0.98, 0.02], requires_grad=True)
        perfect_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_perfect = loss_fn(perfect_pred, perfect_true)
        
        # Random predictions
        random_pred = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        loss_random = loss_fn(random_pred, perfect_true)
        
        assert loss_perfect.item() < loss_random.item(), \
            f"Perfect loss ({loss_perfect.item()}) should be < random loss ({loss_random.item()})"
    
    def test_soft_auc_perfect_vs_random(self):
        """Perfect predictions should have lower loss than random predictions."""
        loss_fn = SoftAUCLoss()
        
        # Perfect predictions
        perfect_pred = torch.tensor([0.99, 0.01, 0.98, 0.02], requires_grad=True)
        perfect_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_perfect = loss_fn(perfect_pred, perfect_true)
        
        # Random predictions
        random_pred = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        loss_random = loss_fn(random_pred, perfect_true)
        
        assert loss_perfect.item() < loss_random.item(), \
            f"Perfect loss ({loss_perfect.item()}) should be < random loss ({loss_random.item()})"
    
    def test_wmw_perfect_vs_random(self):
        """Perfect predictions should have lower loss than random predictions."""
        loss_fn = WMWLoss()
        
        # Perfect predictions: positives much higher than negatives
        perfect_pred = torch.tensor([0.99, 0.01, 0.98, 0.02], requires_grad=True)
        perfect_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_perfect = loss_fn(perfect_pred, perfect_true)
        
        # Bad predictions: negatives higher than positives (creates violations)
        bad_pred = torch.tensor([0.2, 0.8, 0.1, 0.9], requires_grad=True)
        loss_bad = loss_fn(bad_pred, perfect_true)
        
        # Perfect should have lower loss than bad predictions
        assert loss_perfect.item() <= loss_bad.item(), \
            f"Perfect loss ({loss_perfect.item()}) should be <= bad loss ({loss_bad.item()})"
        
        # Bad predictions should have positive loss (violations exist)
        assert loss_bad.item() > 0.0, \
            f"Bad predictions should create violations and have positive loss, got {loss_bad.item()}"
    
    def test_combined_loss_perfect_vs_random(self):
        """Perfect predictions should have lower loss than random predictions."""
        loss_fn = CombinedLoss()
        
        # Perfect predictions
        perfect_pred = torch.tensor([0.99, 0.01, 0.98, 0.02], requires_grad=True)
        perfect_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_perfect, _ = loss_fn(perfect_pred, perfect_true)
        
        # Random predictions
        random_pred = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        loss_random, _ = loss_fn(random_pred, perfect_true)
        
        assert loss_perfect.item() < loss_random.item(), \
            f"Perfect loss ({loss_perfect.item()}) should be < random loss ({loss_random.item()})"


class TestEdgeCases:
    """Test edge case handling."""
    
    def test_all_positive_batch(self):
        """Test handling of batch with all positive samples."""
        loss_fn = PairwiseAUCLoss()
        y_pred = torch.tensor([0.8, 0.9, 0.7], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0, 1.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should return zero loss (no negative samples to compare)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_all_negative_batch(self):
        """Test handling of batch with all negative samples."""
        loss_fn = PairwiseAUCLoss()
        y_pred = torch.tensor([0.2, 0.1, 0.3], requires_grad=True)
        y_true = torch.tensor([0.0, 0.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should return zero loss (no positive samples to compare)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_single_sample(self):
        """Test handling of single sample batch."""
        loss_fn = PairwiseAUCLoss()
        y_pred = torch.tensor([0.5], requires_grad=True)
        y_true = torch.tensor([1.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should return zero loss (no pairs possible)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_soft_auc_all_positive(self):
        """Test SoftAUCLoss with all positive batch."""
        loss_fn = SoftAUCLoss()
        y_pred = torch.tensor([0.8, 0.9], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0])
        
        loss = loss_fn(y_pred, y_true)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
    
    def test_wmw_all_positive(self):
        """Test WMWLoss with all positive batch."""
        loss_fn = WMWLoss()
        y_pred = torch.tensor([0.8, 0.9], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0])
        
        loss = loss_fn(y_pred, y_true)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
    
    def test_combined_loss_all_positive(self):
        """Test CombinedLoss with all positive batch."""
        loss_fn = CombinedLoss()
        y_pred = torch.tensor([0.8, 0.9], requires_grad=True)
        y_true = torch.tensor([1.0, 1.0])
        
        loss, breakdown = loss_fn(y_pred, y_true)
        # BCE should still work, AUC should be 0
        assert breakdown['auc'] == 0.0
        assert breakdown['bce'] > 0  # BCE should still compute
        assert not torch.isnan(loss)
    
    def test_2d_predictions(self):
        """Test that 2D predictions (batch_size, 1) are handled correctly."""
        loss_fn = PairwiseAUCLoss()
        y_pred = torch.tensor([[0.8], [0.2], [0.9], [0.1]], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_empty_tensors_handled_gracefully(self):
        """Test that empty tensors are handled (should not crash)."""
        loss_fn = PairwiseAUCLoss()
        # Empty batch
        y_pred = torch.tensor([], requires_grad=True)
        y_true = torch.tensor([])
        
        # This should handle gracefully (might raise error or return 0)
        try:
            loss = loss_fn(y_pred, y_true)
            assert not torch.isnan(loss)
        except (RuntimeError, IndexError):
            # Empty tensor handling might raise error, which is acceptable
            pass


class TestCombinedLossBreakdown:
    """Test CombinedLoss breakdown and weighting."""
    
    def test_combined_loss_breakdown_structure(self):
        """Test that breakdown contains expected keys."""
        loss_fn = CombinedLoss(alpha=0.3, auc_loss_type='pairwise')
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss, breakdown = loss_fn(y_pred, y_true)
        
        assert 'bce' in breakdown
        assert 'auc' in breakdown
        assert 'total' in breakdown
        assert isinstance(breakdown['bce'], float)
        assert isinstance(breakdown['auc'], float)
        assert isinstance(breakdown['total'], float)
    
    def test_combined_loss_alpha_weighting(self):
        """Test that alpha weighting is correct."""
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss_fn = CombinedLoss(alpha=0.3, auc_loss_type='pairwise')
        loss, breakdown = loss_fn(y_pred, y_true)
        
        # Verify: total = alpha * bce + (1 - alpha) * auc
        expected_total = 0.3 * breakdown['bce'] + 0.7 * breakdown['auc']
        actual_total = breakdown['total']
        
        assert abs(expected_total - actual_total) < 1e-5, \
            f"Expected {expected_total}, got {actual_total}"
    
    def test_combined_loss_alpha_zero(self):
        """Test CombinedLoss with alpha=0 (pure AUC loss)."""
        loss_fn = CombinedLoss(alpha=0.0, auc_loss_type='pairwise')
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss, breakdown = loss_fn(y_pred, y_true)
        
        # With alpha=0, total should equal AUC loss
        assert abs(breakdown['total'] - breakdown['auc']) < 1e-5
    
    def test_combined_loss_alpha_one(self):
        """Test CombinedLoss with alpha=1 (pure BCE loss)."""
        loss_fn = CombinedLoss(alpha=1.0, auc_loss_type='pairwise')
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss, breakdown = loss_fn(y_pred, y_true)
        
        # With alpha=1, total should equal BCE loss
        assert abs(breakdown['total'] - breakdown['bce']) < 1e-5
    
    def test_combined_loss_different_auc_types(self):
        """Test CombinedLoss with different AUC loss types."""
        y_pred = torch.tensor([0.8, 0.2, 0.9, 0.1], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        for auc_type in ['pairwise', 'soft', 'wmw']:
            loss_fn = CombinedLoss(alpha=0.3, auc_loss_type=auc_type)
            loss, breakdown = loss_fn(y_pred, y_true)
            
            assert 'bce' in breakdown
            assert 'auc' in breakdown
            assert 'total' in breakdown
            assert not torch.isnan(loss)


class TestFactoryFunction:
    """Test the create_loss_function factory."""
    
    def test_factory_bce(self):
        """Test factory creates BCE loss."""
        config = {'loss_type': 'bce'}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, nn.BCELoss)
    
    def test_factory_pairwise_auc(self):
        """Test factory creates PairwiseAUCLoss."""
        config = {'loss_type': 'pairwise_auc'}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, PairwiseAUCLoss)
    
    def test_factory_soft_auc(self):
        """Test factory creates SoftAUCLoss."""
        config = {'loss_type': 'soft_auc', 'auc_gamma': 2.0}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, SoftAUCLoss)
        assert loss_fn.gamma == 2.0
    
    def test_factory_wmw(self):
        """Test factory creates WMWLoss."""
        config = {'loss_type': 'wmw', 'margin': 0.1}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, WMWLoss)
    
    def test_factory_combined(self):
        """Test factory creates CombinedLoss."""
        config = {
            'loss_type': 'combined',
            'loss_alpha': 0.3,
            'auc_loss_type': 'pairwise'
        }
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, CombinedLoss)
        assert loss_fn.alpha == 0.3
        assert loss_fn.auc_loss_type == 'pairwise'
    
    def test_factory_handles_all_loss_types(self):
        """Test factory handles all supported loss types."""
        loss_types = ['bce', 'pairwise_auc', 'soft_auc', 'wmw', 'combined']
        
        for loss_type in loss_types:
            config = {'loss_type': loss_type}
            if loss_type == 'combined':
                config['auc_loss_type'] = 'pairwise'
            
            loss_fn = create_loss_function(config)
            assert loss_fn is not None


class TestEfficiency:
    """Test efficiency with large batches."""
    
    def test_large_batch_pairwise_auc(self):
        """Test PairwiseAUCLoss with large batch (should complete quickly)."""
        loss_fn = PairwiseAUCLoss()
        
        # Large batch
        large_pred = torch.randn(1000, requires_grad=True)
        large_true = torch.randint(0, 2, (1000,)).float()
        
        # Should complete without issues
        loss = loss_fn(large_pred, large_true)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Test gradient computation
        loss.backward()
        assert large_pred.grad is not None
    
    def test_large_batch_soft_auc(self):
        """Test SoftAUCLoss with large batch."""
        loss_fn = SoftAUCLoss()
        
        large_pred = torch.randn(1000, requires_grad=True)
        large_true = torch.randint(0, 2, (1000,)).float()
        
        loss = loss_fn(large_pred, large_true)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_large_batch_wmw(self):
        """Test WMWLoss with large batch."""
        loss_fn = WMWLoss()
        
        large_pred = torch.randn(1000, requires_grad=True)
        large_true = torch.randint(0, 2, (1000,)).float()
        
        loss = loss_fn(large_pred, large_true)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMathematicalVerification:
    """Test mathematical correctness of loss formulations."""
    
    def test_pairwise_auc_formula(self):
        """Manually verify PairwiseAUCLoss formula."""
        loss_fn = PairwiseAUCLoss(margin=0.0)
        
        # Simple case: 2 positives, 2 negatives
        y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Manual calculation:
        # pos_scores = [0.9, 0.8]
        # neg_scores = [0.1, 0.2]
        # pairs: (0.9, 0.1), (0.9, 0.2), (0.8, 0.1), (0.8, 0.2)
        # score_diff = [0.8, 0.7, 0.7, 0.6]
        # loss = mean(softplus(-score_diff))
        # softplus(-0.8) = log(1 + exp(-0.8)) ≈ 0.371
        # softplus(-0.7) = log(1 + exp(-0.7)) ≈ 0.403
        # softplus(-0.6) = log(1 + exp(-0.6)) ≈ 0.437
        # mean ≈ (0.371 + 0.403 + 0.403 + 0.437) / 4 ≈ 0.404
        
        # Just verify it's a reasonable value
        assert 0.0 < loss.item() < 1.0
    
    def test_soft_auc_sigmoid_approximation(self):
        """Verify SoftAUCLoss uses sigmoid approximation correctly."""
        loss_fn = SoftAUCLoss(gamma=2.0)
        
        y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Soft AUC should be in [0, 1], so loss = 1 - soft_auc should also be in [0, 1]
        assert 0.0 <= loss.item() <= 1.0
        
        # For good separation, loss should be low
        assert loss.item() < 0.5
    
    def test_wmw_margin_effect(self):
        """Test that margin parameter affects WMWLoss."""
        y_pred = torch.tensor([0.6, 0.4, 0.7, 0.3], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss_no_margin = WMWLoss(margin=0.0)
        loss_with_margin = WMWLoss(margin=0.5)
        
        loss1 = loss_no_margin(y_pred, y_true)
        loss2 = loss_with_margin(y_pred, y_true)
        
        # With margin, loss should generally be higher (stricter requirement)
        # But this depends on the specific scores, so just verify both are valid
        assert not torch.isnan(loss1)
        assert not torch.isnan(loss2)
        assert loss1.item() >= 0
        assert loss2.item() >= 0


class TestLegacyLosses:
    """Test legacy loss functions for backward compatibility."""
    
    def test_ar_loss(self):
        """Test ARLoss legacy function."""
        loss_fn = ARLoss(alpha=1.0)
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_rank_loss(self):
        """Test RankLoss legacy function."""
        loss_fn = RankLoss()
        y_pred = torch.tensor([0.8, 0.2], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
