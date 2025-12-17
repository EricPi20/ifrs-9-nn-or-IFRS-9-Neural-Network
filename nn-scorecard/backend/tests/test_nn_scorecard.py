"""
Tests for Neural Network Models

Comprehensive unit tests for LinearScorecardNN and ScorecardNN classes.
"""

import pytest
import torch
import numpy as np
from app.services.nn_scorecard import LinearScorecardNN, ScorecardNN, create_model
from app.models.schemas import NeuralNetworkConfig


# ============================================================================
# LinearScorecardNN Tests
# ============================================================================

def test_linear_scorecard_nn_forward():
    """Test LinearScorecardNN forward pass."""
    model = LinearScorecardNN(input_dim=10)
    x = torch.randn(32, 10)
    
    # Test forward pass
    y = model(x)
    assert y.shape == (32, 1)
    assert (y >= 0).all() and (y <= 1).all()  # Probabilities
    
    # Test with different batch sizes
    x2 = torch.randn(1, 10)
    y2 = model(x2)
    assert y2.shape == (1, 1)
    
    x3 = torch.randn(100, 10)
    y3 = model(x3)
    assert y3.shape == (100, 1)


def test_linear_scorecard_nn_coefficients():
    """Test coefficient extraction from LinearScorecardNN."""
    model = LinearScorecardNN(input_dim=10)
    x = torch.randn(32, 10)
    
    # Test coefficient extraction
    weights, bias = model.get_coefficients()
    assert weights.shape == (10,)
    assert isinstance(bias, float)
    assert isinstance(weights, np.ndarray)
    
    # Test that coefficients match model parameters
    model_weights = model.linear.weight.data.cpu().numpy().flatten()
    model_bias = model.linear.bias.data.cpu().item()
    np.testing.assert_array_almost_equal(weights, model_weights)
    assert abs(bias - model_bias) < 1e-6


def test_linear_scorecard_nn_log_odds():
    """Test log odds extraction."""
    model = LinearScorecardNN(input_dim=10)
    x = torch.randn(32, 10)
    
    log_odds = model.get_log_odds(x)
    probabilities = model(x)
    
    # Verify log_odds is before sigmoid
    assert log_odds.shape == (32, 1)
    # Manually apply sigmoid to log_odds and compare
    manual_probs = torch.sigmoid(log_odds)
    torch.testing.assert_close(probabilities, manual_probs)


# ============================================================================
# ScorecardNN Architecture Tests
# ============================================================================

def test_scorecard_nn_single_hidden_layer():
    """Test ScorecardNN with single hidden layer."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32])
    x = torch.randn(32, 10)
    
    y = model(x)
    assert y.shape == (32, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_scorecard_nn_multiple_hidden_layers():
    """Test ScorecardNN with multiple hidden layers."""
    model = ScorecardNN(input_dim=10, hidden_layers=[64, 32, 16])
    x = torch.randn(32, 10)
    
    y = model(x)
    assert y.shape == (32, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_scorecard_nn_empty_hidden_layers():
    """Test ScorecardNN with empty hidden layers (linear model)."""
    model = ScorecardNN(input_dim=10, hidden_layers=[])
    x = torch.randn(32, 10)
    
    y = model(x)
    assert y.shape == (32, 1)
    assert (y >= 0).all() and (y <= 1).all()
    
    # Should behave like a linear model
    assert len(model.hidden_layers) == 0


def test_scorecard_nn_configurable_neurons():
    """Test configurable neurons per layer."""
    configs = [
        [128],              # Single layer, 128 neurons
        [64, 64],           # Two equal layers
        [128, 64, 32, 16],  # Four decreasing layers
        [16, 32, 64],       # Increasing layers
    ]
    
    for hidden_layers in configs:
        model = ScorecardNN(input_dim=10, hidden_layers=hidden_layers)
        assert model.hidden_layers == hidden_layers
        summary = model.get_architecture_summary()
        assert summary['hidden_layers'] == hidden_layers
        
        # Test forward pass works
        x = torch.randn(32, 10)
        y = model(x)
        assert y.shape == (32, 1)


# ============================================================================
# Activation Function Tests
# ============================================================================

def test_all_activation_functions():
    """Test all supported activation functions."""
    activations = ['relu', 'leaky_relu', 'elu', 'selu', 'tanh']
    
    for activation in activations:
        model = ScorecardNN(
            input_dim=10,
            hidden_layers=[32],
            activation=activation
        )
        x = torch.randn(32, 10)
        y = model(x)
        assert y.shape == (32, 1)
        assert (y >= 0).all() and (y <= 1).all()
        
        # Verify activation is set correctly
        assert model.activation_name == activation
        summary = model.get_architecture_summary()
        assert summary['activation'] == activation


def test_invalid_activation_function():
    """Test that invalid activation function raises error."""
    with pytest.raises(ValueError, match="not supported"):
        ScorecardNN(input_dim=10, hidden_layers=[32], activation='invalid_activation')


# ============================================================================
# Regularization Tests
# ============================================================================

def test_l1_regularization():
    """Test L1 regularization calculation."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
    
    l1 = model.get_l1_regularization()
    assert isinstance(l1, torch.Tensor)
    assert l1.shape == ()  # Scalar
    assert l1.item() > 0  # Should have some weight magnitude
    
    # L1 should be sum of absolute values
    manual_l1 = sum(torch.abs(p).sum() for p in model.parameters())
    torch.testing.assert_close(l1, manual_l1)


def test_l2_regularization():
    """Test L2 regularization calculation."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
    
    l2 = model.get_l2_regularization()
    assert isinstance(l2, torch.Tensor)
    assert l2.shape == ()  # Scalar
    assert l2.item() > 0  # Should have some weight magnitude
    
    # L2 should be sum of squared values
    manual_l2 = sum((p ** 2).sum() for p in model.parameters())
    torch.testing.assert_close(l2, manual_l2)


def test_regularization_with_linear_model():
    """Test regularization with linear model."""
    model = LinearScorecardNN(input_dim=10)
    
    # Linear model doesn't have regularization methods, but we can test
    # that it has parameters
    params = list(model.parameters())
    assert len(params) == 2  # weight and bias


# ============================================================================
# Gradient Flow Tests
# ============================================================================

def test_gradient_flow():
    """Test that gradients flow through the network."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
    x = torch.randn(32, 10, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
    
    # Check input gradients
    assert x.grad is not None


def test_gradient_flow_linear():
    """Test gradient flow for linear model."""
    model = LinearScorecardNN(input_dim=10)
    x = torch.randn(32, 10, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_model_neural_network():
    """Test create_model factory for neural network."""
    config = NeuralNetworkConfig(
        model_type='neural_network',
        hidden_layers=[64, 32],
        activation='relu',
        dropout_rate=0.3
    )
    model = create_model(input_dim=10, config=config)
    assert isinstance(model, ScorecardNN)
    assert model.hidden_layers == [64, 32]
    assert model.activation_name == 'relu'
    assert model.dropout_rate == 0.3


def test_create_model_linear():
    """Test create_model factory for linear model."""
    config_linear = NeuralNetworkConfig(model_type='linear')
    model_linear = create_model(input_dim=10, config=config_linear)
    assert isinstance(model_linear, LinearScorecardNN)


def test_create_model_empty_hidden_layers():
    """Test create_model with empty hidden layers creates linear model."""
    config = NeuralNetworkConfig(
        model_type='neural_network',
        hidden_layers=[]
    )
    model = create_model(input_dim=10, config=config)
    assert isinstance(model, LinearScorecardNN)


# ============================================================================
# Architecture Summary Tests
# ============================================================================

def test_architecture_summary():
    """Test architecture summary generation."""
    model = ScorecardNN(
        input_dim=15,
        hidden_layers=[64, 32],
        activation='relu',
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    summary = model.get_architecture_summary()
    assert summary['input_dim'] == 15
    assert summary['hidden_layers'] == [64, 32]
    assert summary['activation'] == 'relu'
    assert summary['dropout_rate'] == 0.2
    assert summary['use_batch_norm'] == True
    assert 'total_parameters' in summary
    assert 'trainable_parameters' in summary
    assert summary['total_parameters'] > 0
    assert summary['trainable_parameters'] == summary['total_parameters']


# ============================================================================
# Weight Extraction Tests
# ============================================================================

def test_get_first_layer_weights():
    """Test first layer weight extraction."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
    weights = model.get_first_layer_weights()
    
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (32, 10)  # (hidden_dim, input_dim)


def test_get_first_layer_weights_linear():
    """Test first layer weights for linear model."""
    model = ScorecardNN(input_dim=10, hidden_layers=[])
    weights = model.get_first_layer_weights()
    
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (1, 10)  # (output_dim, input_dim)


# ============================================================================
# Device Handling Tests
# ============================================================================

def test_device_handling():
    """Test model works on CPU."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
    x = torch.randn(32, 10)
    
    y = model(x)
    assert y.device.type == 'cpu'


def test_device_handling_linear():
    """Test linear model works on CPU."""
    model = LinearScorecardNN(input_dim=10)
    x = torch.randn(32, 10)
    
    y = model(x)
    assert y.device.type == 'cpu'


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_different_input_dimensions():
    """Test models with different input dimensions."""
    for input_dim in [1, 5, 10, 50, 100]:
        model = ScorecardNN(input_dim=input_dim, hidden_layers=[32])
        x = torch.randn(16, input_dim)
        y = model(x)
        assert y.shape == (16, 1)


def test_dropout_zero():
    """Test model with dropout_rate=0.0."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32], dropout_rate=0.0)
    x = torch.randn(32, 10)
    y = model(x)
    assert y.shape == (32, 1)


def test_no_batch_norm():
    """Test model without batch normalization."""
    model = ScorecardNN(
        input_dim=10,
        hidden_layers=[32],
        use_batch_norm=False
    )
    x = torch.randn(32, 10)
    y = model(x)
    assert y.shape == (32, 1)
    assert model.use_batch_norm == False


def test_batch_norm_with_single_sample():
    """Test batch norm with single sample (should still work in eval mode)."""
    model = ScorecardNN(input_dim=10, hidden_layers=[32], use_batch_norm=True)
    model.eval()  # Important for batch norm with single sample
    
    x = torch.randn(1, 10)
    y = model(x)
    assert y.shape == (1, 1)
