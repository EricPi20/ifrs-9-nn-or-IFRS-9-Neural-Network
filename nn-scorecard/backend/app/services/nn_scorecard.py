"""
Neural Network Model Classes for Credit Scorecard Development

This module defines fully configurable PyTorch neural network architectures
for credit scorecard modeling with support for:
- Configurable hidden layers (0 for linear model)
- Multiple activation functions
- Dropout and batch normalization
- L1 and L2 regularization
- Weight extraction for scorecard conversion
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import numpy as np


class LinearScorecardNN(nn.Module):
    """
    Single-layer linear model (equivalent to logistic regression).
    
    This is the most interpretable model where:
    log_odds = Σ(weight_i * WoE_i) + bias
    
    Use this when:
    - Maximum interpretability is required
    - Features are already well-engineered (WoE transformed)
    - No feature interactions are expected
    
    Attributes:
        input_dim: Number of input features
        feature_names: Optional list of feature names for interpretability
    
    Example:
        >>> model = LinearScorecardNN(input_dim=15)
        >>> x = torch.randn(100, 15)  # 100 samples, 15 features
        >>> probabilities = model(x)  # Shape: (100, 1)
        >>> log_odds = model.get_log_odds(x)  # Shape: (100, 1)
        >>> weights, bias = model.get_coefficients()
        >>> print(f"Weights shape: {weights.shape}, Bias: {bias}")
        Weights shape: (15,), Bias: 0.123
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize linear scorecard model.
        
        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.feature_names: List[str] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probability.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        logits = self.linear(x)
        return self.sigmoid(logits)
    
    def get_log_odds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw log odds (before sigmoid).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Log odds tensor of shape (batch_size, 1)
        """
        return self.linear(x)
    
    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """
        Extract model coefficients for scorecard.
        
        Returns:
            weights: Array of shape (input_dim,) with feature weights
            bias: Scalar bias term
            
        Example:
            >>> model = LinearScorecardNN(input_dim=5)
            >>> weights, bias = model.get_coefficients()
            >>> print(f"Feature weights: {weights}")
            >>> print(f"Bias term: {bias}")
        """
        weights = self.linear.weight.data.cpu().numpy().flatten()
        bias = self.linear.bias.data.cpu().item()
        return weights, bias


class ScorecardNN(nn.Module):
    """
    Configurable neural network for credit scorecard development.
    
    This network supports:
    - Configurable hidden layers: [64, 32] means 2 layers with 64 and 32 neurons
    - Multiple activation functions: relu, leaky_relu, elu, selu, tanh
    - Dropout for regularization
    - Batch normalization for training stability
    - Optional skip connection from input to output (residual learning)
    
    Architecture (without skip connection):
        Input (n_features)
        -> [Hidden Layer 1 -> BatchNorm -> Activation -> Dropout] (optional)
        -> [Hidden Layer 2 -> BatchNorm -> Activation -> Dropout] (optional)
        -> ... (more hidden layers as configured)
        -> Output Layer (1 neuron) -> Sigmoid
    
    Architecture (with skip connection):
        Input (n_features)
        -> [Hidden Layers -> Output Layer] (logits)
        -> ⊕ (add skip connection: W_skip @ Input)
        -> Sigmoid
    
    When hidden_layers=[] (empty list), this becomes equivalent to 
    LinearScorecardNN.
    
    Attributes:
        input_dim: Number of input features
        hidden_layers: List of neurons per hidden layer (e.g., [64, 32, 16])
        activation: Activation function name
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
        skip_connection: Whether to use skip connection from input to output
        feature_names: List of feature names for interpretability
    
    Example:
        >>> # Simple 2-layer network
        >>> model = ScorecardNN(
        ...     input_dim=15,
        ...     hidden_layers=[32, 16],
        ...     activation='relu',
        ...     dropout_rate=0.2,
        ...     use_batch_norm=True
        ... )
        >>> x = torch.randn(100, 15)
        >>> probabilities = model(x)  # Shape: (100, 1)
        >>> 
        >>> # Network with skip connection
        >>> skip_model = ScorecardNN(
        ...     input_dim=15,
        ...     hidden_layers=[32, 16],
        ...     activation='relu',
        ...     skip_connection=True
        ... )
        >>> 
        >>> # Deep network
        >>> deep_model = ScorecardNN(
        ...     input_dim=20,
        ...     hidden_layers=[128, 64, 32, 16],
        ...     activation='leaky_relu',
        ...     dropout_rate=0.3
        ... )
        >>> 
        >>> # Linear model (no hidden layers)
        >>> linear_model = ScorecardNN(
        ...     input_dim=15,
        ...     hidden_layers=[]
        ... )
    """
    
    # Mapping of activation names to PyTorch modules
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        skip_connection: bool = False
    ):
        """
        Initialize configurable neural network.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons per hidden layer. Empty list [] 
                          creates a linear model.
            activation: Activation function name. Must be one of:
                       'relu', 'leaky_relu', 'elu', 'selu', 'tanh', 'sigmoid'
            dropout_rate: Dropout probability (0.0 to 1.0). Set to 0.0 to disable.
            use_batch_norm: Whether to use batch normalization after each 
                           hidden layer
            skip_connection: If True, adds direct connection from input to output.
                            Creates residual learning: output = sigmoid(hidden + W_skip @ input)
            
        Raises:
            ValueError: If activation function is not supported
        """
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.skip_connection = skip_connection
        self.feature_names: List[str] = []
        
        # Validate activation
        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from: {list(self.ACTIVATIONS.keys())}"
            )
        
        activation_class = self.ACTIVATIONS[activation]
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Handle case where hidden_layers is empty (linear model)
        if len(hidden_layers) == 0:
            layers.append(nn.Linear(prev_dim, 1))
            # Sigmoid will be added in forward() if skip_connection, otherwise here
            if not skip_connection:
                layers.append(nn.Sigmoid())
        else:
            for i, hidden_dim in enumerate(hidden_layers):
                # Linear layer
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                # Batch normalization (before activation)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                # Activation
                layers.append(activation_class())
                
                # Dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
            
            # Output layer from hidden layers
            layers.append(nn.Linear(prev_dim, 1))
            # Sigmoid will be added in forward() if skip_connection, otherwise here
            if not skip_connection:
                layers.append(nn.Sigmoid())
        
        self.hidden_network = nn.Sequential(*layers)
        
        # Skip connection layer (input directly to output)
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, 1, bias=False)
            print(f"[MODEL] Skip connection enabled: input({input_dim}) -> output")
        else:
            self.skip_layer = None
        
        # Final activation (always use this for skip connection, or as fallback)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
        print(f"[MODEL] Created ScorecardNN:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Hidden layers: {hidden_layers}")
        print(f"  - Activation: {activation}")
        print(f"  - Dropout: {dropout_rate}")
        print(f"  - Skip connection: {skip_connection}")
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probability.
        
        If skip_connection=True:
            output = sigmoid(hidden_output + W_skip @ input)
        Else:
            output = sigmoid(hidden_output) or hidden_network(x) if sigmoid already included
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        # Pass through hidden layers (this gives logits, not probabilities)
        hidden_logits = self.hidden_network(x)
        
        # Add skip connection if enabled
        if self.skip_connection and self.skip_layer is not None:
            skip_out = self.skip_layer(x)
            output_logits = hidden_logits + skip_out
            return self.sigmoid(output_logits)
        else:
            # For models without skip connection, hidden_network already includes sigmoid
            return hidden_logits

    def get_first_layer_weights(self) -> np.ndarray:
        """
        Get weights of the first layer.
        
        For interpretation, we use first-layer weights as an approximation
        of feature importance (assuming roughly linear relationship through
        the network).
        
        Returns:
            Array of shape (hidden_dim, input_dim) if hidden layers exist,
            or (1, input_dim) for linear model
            
        Example:
            >>> model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
            >>> weights = model.get_first_layer_weights()
            >>> print(f"First layer weights shape: {weights.shape}")
            First layer weights shape: (32, 10)
        """
        # Get the first Linear layer
        first_linear = None
        for module in self.hidden_network.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        
        if first_linear is not None:
            return first_linear.weight.data.cpu().numpy()
        return np.zeros((1, self.input_dim))
    
    def get_feature_weights(self) -> torch.Tensor:
        """
        Get effective feature weights for scorecard.
        
        For skip connection model, combines hidden layer influence with skip weights.
        For regular model, extracts first layer weights.
        
        Returns:
            Tensor of shape (input_dim,) with feature importance weights
        """
        if self.skip_connection and self.skip_layer is not None:
            # Skip connection weights directly show feature importance
            skip_weights = self.skip_layer.weight.data.squeeze()
            
            # Also get influence through hidden layers (approximate)
            first_hidden = None
            for layer in self.hidden_network:
                if isinstance(layer, nn.Linear):
                    first_hidden = layer
                    break
            
            if first_hidden is not None:
                # Combine skip weights with first layer influence
                # Skip weights dominate, but hidden layers add some contribution
                hidden_weights = first_hidden.weight.data.abs().mean(dim=0)
                combined = skip_weights + 0.1 * hidden_weights  # Skip weights dominate
                return combined
            
            return skip_weights
        else:
            # Regular model: use first layer weights
            for layer in self.hidden_network:
                if isinstance(layer, nn.Linear):
                    return layer.weight.data.abs().mean(dim=0) if layer.weight.data.shape[0] > 1 else layer.weight.data.squeeze()
            
            return torch.zeros(self.input_dim)
    
    def get_l1_regularization(self) -> torch.Tensor:
        """
        Calculate L1 regularization term (sum of absolute weights).
        
        L1 encourages sparsity - some weights become exactly zero.
        Useful for feature selection.
        
        Returns:
            Scalar tensor with L1 regularization value
            
        Example:
            >>> model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
            >>> l1_reg = model.get_l1_regularization()
            >>> print(f"L1 regularization: {l1_reg.item()}")
        """
        l1_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l1_reg += torch.abs(param).sum()
        return l1_reg
    
    def get_l2_regularization(self) -> torch.Tensor:
        """
        Calculate L2 regularization term (sum of squared weights).
        
        L2 encourages small weights but doesn't force them to zero.
        Useful for preventing overfitting.
        
        Returns:
            Scalar tensor with L2 regularization value
            
        Example:
            >>> model = ScorecardNN(input_dim=10, hidden_layers=[32, 16])
            >>> l2_reg = model.get_l2_regularization()
            >>> print(f"L2 regularization: {l2_reg.item()}")
        """
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg += (param ** 2).sum()
        return l2_reg
    
    def get_architecture_summary(self) -> Dict:
        """
        Get summary of network architecture.
        
        Returns:
            Dictionary with architecture details including:
            - input_dim: Number of input features
            - hidden_layers: List of hidden layer sizes
            - activation: Activation function name
            - dropout_rate: Dropout probability
            - use_batch_norm: Whether batch norm is used
            - total_parameters: Total number of parameters
            - trainable_parameters: Number of trainable parameters
            
        Example:
            >>> model = ScorecardNN(
            ...     input_dim=15,
            ...     hidden_layers=[64, 32],
            ...     activation='relu',
            ...     dropout_rate=0.2
            ... )
            >>> summary = model.get_architecture_summary()
            >>> print(f"Total parameters: {summary['total_parameters']}")
        """
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_model(
    input_dim: int,
    config: 'NeuralNetworkConfig'
) -> nn.Module:
    """
    Factory function to create model from configuration.
    
    This function creates either a LinearScorecardNN or ScorecardNN based
    on the configuration. If model_type is 'linear' or hidden_layers is empty,
    a LinearScorecardNN is created. Otherwise, a ScorecardNN is created.
    
    Args:
        input_dim: Number of input features
        config: NeuralNetworkConfig from schemas module
        
    Returns:
        LinearScorecardNN if hidden_layers is empty or model_type is 'linear',
        ScorecardNN otherwise
        
    Example:
        >>> from app.models.schemas import NeuralNetworkConfig
        >>> 
        >>> # Create linear model
        >>> linear_config = NeuralNetworkConfig(
        ...     model_type='linear',
        ...     hidden_layers=[]
        ... )
        >>> model = create_model(input_dim=15, config=linear_config)
        >>> print(type(model).__name__)
        LinearScorecardNN
        >>> 
        >>> # Create neural network
        >>> nn_config = NeuralNetworkConfig(
        ...     model_type='neural_network',
        ...     hidden_layers=[64, 32],
        ...     activation='relu',
        ...     dropout_rate=0.2
        ... )
        >>> model = create_model(input_dim=15, config=nn_config)
        >>> print(type(model).__name__)
        ScorecardNN
    """
    if config.model_type == 'linear' or len(config.hidden_layers) == 0:
        return LinearScorecardNN(input_dim)
    
    return ScorecardNN(
        input_dim=input_dim,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
        skip_connection=getattr(config, 'skip_connection', False)
    )


def get_model_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device object (cuda or cpu)
        
    Example:
        >>> device = get_model_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
        >>> 
        >>> # Move model to device
        >>> model = ScorecardNN(input_dim=10)
        >>> model = model.to(device)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
