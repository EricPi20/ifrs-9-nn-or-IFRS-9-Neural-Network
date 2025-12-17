import numpy as np
from typing import List, Dict, Tuple

class NeuralNetwork:
    """Feedforward neural network for credit scoring."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        l2_lambda: float = 0.001,
        skip_connection: bool = False,
        random_seed: int = 42,
        loss_function: str = 'bce',
        use_class_weights: bool = False,
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.skip_connection = skip_connection
        self.loss_function = loss_function
        self.use_class_weights = use_class_weights
        self.class_weights = None
        
        np.random.seed(random_seed)
        
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_layers + [1]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        if skip_connection and len(hidden_layers) > 1:
            self.skip_weight = np.random.randn(input_size, 1) * np.sqrt(2.0 / input_size) * 0.1
        
        self.training = True
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'elu':
            return np.where(x > 0, x, np.exp(x) - 1)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'selu':
            alpha, scale = 1.6732632423543772, 1.0507009873554805
            return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        return np.maximum(0, x)
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'elu':
            return np.where(x > 0, 1, np.exp(x))
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'selu':
            alpha, scale = 1.6732632423543772, 1.0507009873554805
            return scale * np.where(x > 0, 1, alpha * np.exp(x))
        return (x > 0).astype(float)
    
    def _dropout(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def set_class_weights(self, y: np.ndarray):
        """Calculate class weights for imbalanced data."""
        if self.use_class_weights:
            n_samples = len(y)
            n_pos = np.sum(y)
            n_neg = n_samples - n_pos
            # Inverse frequency weighting
            self.class_weights = {
                0: n_samples / (2 * n_neg) if n_neg > 0 else 1.0,
                1: n_samples / (2 * n_pos) if n_pos > 0 else 1.0,
            }
            print(f"[NN] Class weights: {self.class_weights}")
    
    def _compute_loss_and_gradient(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        X: np.ndarray = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss and gradient based on selected loss function.
        
        Supported loss functions:
        - bce: Binary Cross-Entropy
        - pairwise_auc: Pairwise AUC Loss
        - soft_auc: Soft AUC Loss (differentiable approximation)
        - wmw: Wilcoxon-Mann-Whitney Loss
        - combined: BCE + AUC combined loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.reshape(-1, 1)
        m = len(y_true)
        
        # Apply class weights if enabled
        if self.use_class_weights and self.class_weights:
            sample_weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
        else:
            sample_weights = np.ones_like(y_true)
        
        if self.loss_function == 'bce':
            # Binary Cross-Entropy Loss
            loss = -np.mean(sample_weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
            gradient = sample_weights * (y_pred - y_true) / m
            
        elif self.loss_function == 'pairwise_auc':
            # Pairwise AUC Loss - penalizes incorrect orderings
            pos_idx = np.where(y_true.flatten() == 1)[0]
            neg_idx = np.where(y_true.flatten() == 0)[0]
            
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                # Fall back to BCE if no pairs
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                gradient = (y_pred - y_true) / m
            else:
                # Sample pairs for efficiency
                n_pairs = min(len(pos_idx) * len(neg_idx), 1000)
                pos_samples = np.random.choice(pos_idx, n_pairs, replace=True)
                neg_samples = np.random.choice(neg_idx, n_pairs, replace=True)
                
                # Margin-based pairwise loss
                margin = 0.1
                diff = y_pred[pos_samples] - y_pred[neg_samples]
                pairwise_loss = np.maximum(0, margin - diff)
                loss = np.mean(pairwise_loss)
                
                # Gradient: push pos predictions up, neg predictions down
                gradient = np.zeros_like(y_pred)
                violated = (diff < margin).flatten()
                
                for i, (p, n) in enumerate(zip(pos_samples, neg_samples)):
                    if violated[i]:
                        gradient[p] -= 1.0 / n_pairs
                        gradient[n] += 1.0 / n_pairs
            
        elif self.loss_function == 'soft_auc':
            # Soft AUC Loss - differentiable AUC approximation
            pos_idx = np.where(y_true.flatten() == 1)[0]
            neg_idx = np.where(y_true.flatten() == 0)[0]
            
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                gradient = (y_pred - y_true) / m
            else:
                # Sigmoid approximation of indicator function
                gamma = 5.0  # Steepness parameter
                
                n_pairs = min(len(pos_idx) * len(neg_idx), 1000)
                pos_samples = np.random.choice(pos_idx, n_pairs, replace=True)
                neg_samples = np.random.choice(neg_idx, n_pairs, replace=True)
                
                diff = y_pred[pos_samples] - y_pred[neg_samples]
                sigmoid_diff = 1 / (1 + np.exp(-gamma * diff))
                
                # Loss is 1 - soft AUC
                loss = 1 - np.mean(sigmoid_diff)
                
                # Gradient
                gradient = np.zeros_like(y_pred)
                sigmoid_grad = gamma * sigmoid_diff * (1 - sigmoid_diff)
                
                for i, (p, n) in enumerate(zip(pos_samples, neg_samples)):
                    gradient[p] -= sigmoid_grad[i] / n_pairs
                    gradient[n] += sigmoid_grad[i] / n_pairs
            
        elif self.loss_function == 'wmw':
            # Wilcoxon-Mann-Whitney Loss
            pos_idx = np.where(y_true.flatten() == 1)[0]
            neg_idx = np.where(y_true.flatten() == 0)[0]
            
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                gradient = (y_pred - y_true) / m
            else:
                # WMW with exponential approximation
                gamma = 2.0
                
                n_pairs = min(len(pos_idx) * len(neg_idx), 1000)
                pos_samples = np.random.choice(pos_idx, n_pairs, replace=True)
                neg_samples = np.random.choice(neg_idx, n_pairs, replace=True)
                
                diff = y_pred[neg_samples] - y_pred[pos_samples]  # Want this negative
                exp_diff = np.exp(gamma * diff)
                
                loss = np.mean(exp_diff)
                
                gradient = np.zeros_like(y_pred)
                exp_grad = gamma * exp_diff
                
                for i, (p, n) in enumerate(zip(pos_samples, neg_samples)):
                    gradient[p] -= exp_grad[i] / n_pairs
                    gradient[n] += exp_grad[i] / n_pairs
            
        elif self.loss_function == 'combined':
            # Combined BCE + AUC Loss
            alpha = 0.5  # Weight for BCE
            
            # BCE component
            bce_loss = -np.mean(sample_weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
            bce_gradient = sample_weights * (y_pred - y_true) / m
            
            # Soft AUC component
            pos_idx = np.where(y_true.flatten() == 1)[0]
            neg_idx = np.where(y_true.flatten() == 0)[0]
            
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                gamma = 5.0
                n_pairs = min(len(pos_idx) * len(neg_idx), 500)
                pos_samples = np.random.choice(pos_idx, n_pairs, replace=True)
                neg_samples = np.random.choice(neg_idx, n_pairs, replace=True)
                
                diff = y_pred[pos_samples] - y_pred[neg_samples]
                sigmoid_diff = 1 / (1 + np.exp(-gamma * diff))
                
                auc_loss = 1 - np.mean(sigmoid_diff)
                
                auc_gradient = np.zeros_like(y_pred)
                sigmoid_grad = gamma * sigmoid_diff * (1 - sigmoid_diff)
                
                for i, (p, n) in enumerate(zip(pos_samples, neg_samples)):
                    auc_gradient[p] -= sigmoid_grad[i] / n_pairs
                    auc_gradient[n] += sigmoid_grad[i] / n_pairs
            else:
                auc_loss = 0
                auc_gradient = np.zeros_like(y_pred)
            
            loss = alpha * bce_loss + (1 - alpha) * auc_loss
            gradient = alpha * bce_gradient + (1 - alpha) * auc_gradient
            
        else:
            # Default to BCE
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            gradient = (y_pred - y_true) / m
        
        return float(loss), gradient
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List, List]:
        activations = [X]
        pre_activations = []
        current = X
        
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self._activate(z)
            a = self._dropout(a)
            activations.append(a)
            current = a
        
        z = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        
        if self.skip_connection and len(self.hidden_layers) > 1:
            z = z + X @ self.skip_weight
        
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        activations.append(output)
        
        return output, activations, pre_activations
    
    def backward(self, X, y, activations, pre_activations, learning_rate) -> float:
        """Backward pass using selected loss function."""
        m = X.shape[0]
        y = y.reshape(-1, 1)
        output = activations[-1]
        
        # Get loss and initial gradient from loss function
        loss, dz = self._compute_loss_and_gradient(y, output, X)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dw = (activations[i].T @ dz) / m + self.l2_lambda * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self._activate_derivative(pre_activations[i-1])
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
        
        if self.skip_connection and len(self.hidden_layers) > 1:
            skip_grad = (X.T @ (output - y)) / m
            self.skip_weight -= learning_rate * skip_grad
        
        # Add L2 regularization to loss
        loss += 0.5 * self.l2_lambda * sum(np.sum(w ** 2) for w in self.weights)
        
        return loss
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        output, _, _ = self.forward(X)
        self.training = True
        return output.flatten()
    
    def get_feature_importance(self) -> np.ndarray:
        importance = np.sum(np.abs(self.weights[0]), axis=1)
        return importance / np.sum(importance)


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sorted_idx = np.argsort(-y_pred)
    y_sorted = y_true[sorted_idx]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr_prev, fpr_prev, auc = 0, 0, 0
    tp, fp = 0, 0
    
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
    
    return auc


def calculate_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sorted_idx = np.argsort(-y_pred)
    y_sorted = y_true[sorted_idx]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    
    return np.max(np.abs(cum_pos - cum_neg))


def generate_roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    sorted_idx = np.argsort(-y_pred)
    y_sorted = y_true[sorted_idx]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    fpr_list, tpr_list = [0], [0]
    tp, fp = 0, 0
    step = max(1, len(y_sorted) // 100)
    
    for i in range(0, len(y_sorted), step):
        if y_sorted[i] == 1:
            tp += step
        else:
            fp += step
        
        fpr_list.append(min(100, fp / n_neg * 100))
        tpr_list.append(min(100, tp / n_pos * 100))
    
    fpr_list.append(100)
    tpr_list.append(100)
    
    return {
        'fpr': fpr_list,
        'tpr': tpr_list,
        'auc': round(calculate_auc(y_true, y_pred), 4),
    }


def generate_score_histogram(scores: np.ndarray, y_true: np.ndarray, n_bins: int = 20) -> List[Dict]:
    bin_edges = np.linspace(0, 100, n_bins + 1)
    histogram = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= low) & (scores < high) if i < n_bins - 1 else (scores >= low) & (scores <= high)
        
        total = np.sum(mask)
        bad = np.sum(y_true[mask]) if total > 0 else 0
        
        histogram.append({
            'bin': f"{int(low)}-{int(high)}",
            'total': int(total),
            'good': int(total - bad),
            'bad': int(bad),
            'bad_rate': round(bad / total * 100, 2) if total > 0 else 0,
        })
    
    return histogram


def generate_score_bands(scores: np.ndarray, y_true: np.ndarray) -> List[Dict]:
    bands = [
        (0, 20), (20, 40), (40, 60), (60, 80), (80, 100)
    ]
    
    result = []
    total_count = len(scores)
    
    for low, high in bands:
        mask = (scores >= low) & (scores <= high) if high == 100 else (scores >= low) & (scores < high)
        count = np.sum(mask)
        bad = np.sum(y_true[mask]) if count > 0 else 0
        
        result.append({
            'range': f"{low}-{high}",
            'total': int(count),
            'good': int(count - bad),
            'bad': int(bad),
            'bad_rate': round(bad / count * 100, 2) if count > 0 else 0,
            'pct_total': round(count / total_count * 100, 2) if total_count > 0 else 0,
        })
    
    return result

