"""
Training Loop and Logic

This module handles the neural network training process, including
data loading, training loop, model checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import logging

from .losses import create_loss_function, CombinedLoss
from .nn_scorecard import ScorecardNN, LinearScorecardNN
from .metrics import MetricsCalculator, CompleteMetrics

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch - ALL metrics tracked."""
    epoch: int
    train_loss: float
    test_loss: float
    train_bce_loss: Optional[float] = None
    train_auc_loss: Optional[float] = None
    test_bce_loss: Optional[float] = None
    test_auc_loss: Optional[float] = None
    train_auc: float = 0.0
    test_auc: float = 0.0
    train_ar: float = 0.0
    test_ar: float = 0.0
    train_ks: float = 0.0
    test_ks: float = 0.0
    learning_rate: float = 0.0
    epoch_time_seconds: float = 0.0


@dataclass
class TrainingHistory:
    """Complete training history for documentation."""
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_test_ar: float = 0.0
    total_training_time_seconds: float = 0.0
    early_stopping_triggered: bool = False
    early_stopping_epoch: Optional[int] = None


@dataclass
class TrainingResult:
    """Complete training result."""
    model: nn.Module
    history: TrainingHistory
    train_metrics: CompleteMetrics
    test_metrics: CompleteMetrics
    feature_names: List[str]


class EarlyStopping:
    """
    Early stopping based on TEST AR (Gini).
    Mode='max' because we want to MAXIMIZE AR.
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, epoch: int, model: nn.Module) -> bool:
        """Check if should stop. Returns True if should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save_model(model)
            return False
        
        improved = score > self.best_score + self.min_delta if self.mode == 'max' else score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self._save_model(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def _save_model(self, model: nn.Module):
        """Save model state."""
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_best_model(self, model: nn.Module):
        """Restore best model state."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


class ModelTrainer:
    """
    Neural Network Scorecard Trainer.
    
    - Train/Test split only (no validation)
    - Early stopping on TEST AR
    - Tracks all metrics per epoch
    - Supports AR optimization losses
    
    NOTE: Data in train_loader and test_loader should already be
    normalized (divided by scale_factor). This is handled by
    DataProcessor.prepare_training_data().
    """
    
    def __init__(self, device: Optional[torch.device] = None, scale_factor: float = None):
        """
        Initialize trainer.
        
        Args:
            device: PyTorch device (default: auto-detect)
            scale_factor: Input scale factor used for normalization (for documentation)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calc = MetricsCalculator()
        self.scale_factor = scale_factor  # Store for reference
        logger.info(f"Trainer initialized on device: {self.device}")
        if scale_factor:
            logger.info(f"Input scale factor: {scale_factor} (data should be normalized)")
    
    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 256,
        random_seed: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders with optional seeded shuffling."""
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        
        # Create seeded generator for reproducible shuffling
        generator = None
        if random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(random_seed)
            logger.info(f"DataLoader using seeded generator with seed: {random_seed}")
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        )
    
    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """Compute balanced class weights."""
        n = len(y_train)
        n_pos = y_train.sum()
        n_neg = n - n_pos
        return torch.FloatTensor([n / (2 * n_neg), n / (2 * n_pos)]).to(self.device)
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: 'TrainingConfig',
        feature_names: List[str],
        progress_callback: Optional[Callable] = None
    ) -> TrainingResult:
        """
        Train the model.
        
        NOTE: Data in train_loader and test_loader should already be
        normalized (divided by scale_factor). This is handled by
        DataProcessor.prepare_training_data().
        
        Args:
            model: Neural network
            train_loader: Training data (should be normalized)
            test_loader: Test data (should be normalized)
            config: All hyperparameters
            feature_names: Feature names
            progress_callback: Optional progress updates
        
        Returns:
            TrainingResult with model, history, metrics
        """
        start_time = time.time()
        model = model.to(self.device)
        
        # Loss function
        loss_fn = create_loss_function(config.loss)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.regularization.l2_lambda
        )
        
        # LR Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        
        # Early stopping on TEST AR - only if enabled
        early_stopping = None
        if config.early_stopping.enabled:
            early_stopping = EarlyStopping(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                mode='max'  # Maximize AR
            )
            logger.info(f"Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
        else:
            logger.info(f"Early stopping disabled - will train for all {config.epochs} epochs")
        
        # Track best model even if early stopping is disabled
        best_model_state = None
        best_test_ar = None
        best_epoch = 0
        
        history = TrainingHistory()
        
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            train_loss, train_breakdown = self._train_epoch(
                model, train_loader, loss_fn, optimizer, config
            )
            
            # Evaluate
            train_metrics = self._evaluate(model, train_loader)
            test_metrics = self._evaluate(model, test_loader)
            test_loss, test_breakdown = self._compute_loss(model, test_loader, loss_fn)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record epoch metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_bce_loss=train_breakdown.get('bce'),
                train_auc_loss=train_breakdown.get('auc'),
                test_bce_loss=test_breakdown.get('bce'),
                test_auc_loss=test_breakdown.get('auc'),
                train_auc=train_metrics.discrimination.auc_roc,
                test_auc=test_metrics.discrimination.auc_roc,
                train_ar=train_metrics.discrimination.gini_ar,
                test_ar=test_metrics.discrimination.gini_ar,
                train_ks=train_metrics.discrimination.ks_statistic,
                test_ks=test_metrics.discrimination.ks_statistic,
                learning_rate=current_lr,
                epoch_time_seconds=time.time() - epoch_start
            )
            history.epochs.append(epoch_metrics)
            
            # Log
            logger.info(
                f"Epoch {epoch}/{config.epochs} - "
                f"Loss: {train_loss:.4f}/{test_loss:.4f} - "
                f"AR: {train_metrics.discrimination.gini_ar:.4f}/{test_metrics.discrimination.gini_ar:.4f}"
            )
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'epoch': epoch,
                    'total_epochs': config.epochs,
                    'train_ar': train_metrics.discrimination.gini_ar,
                    'test_ar': test_metrics.discrimination.gini_ar,
                    'train_auc': train_metrics.discrimination.auc_roc,
                    'test_auc': test_metrics.discrimination.auc_roc,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_ks': train_metrics.discrimination.ks_statistic,
                    'test_ks': test_metrics.discrimination.ks_statistic,
                })
            
            # LR scheduler
            scheduler.step(test_loss)
            
            # Track best model (for both early stopping enabled and disabled)
            current_test_ar = test_metrics.discrimination.gini_ar
            if best_test_ar is None or current_test_ar > best_test_ar:
                best_test_ar = current_test_ar
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Early stopping check on TEST AR - ONLY if enabled
            if early_stopping is not None:
                if early_stopping(current_test_ar, epoch, model):
                    logger.info(f"Early stopping at epoch {epoch}. Best: {early_stopping.best_epoch}")
                    history.early_stopping_triggered = True
                    history.early_stopping_epoch = epoch
                    break
        
        # Restore best model
        if early_stopping is not None:
            early_stopping.restore_best_model(model)
            history.best_epoch = early_stopping.best_epoch
            history.best_test_ar = early_stopping.best_score or 0.0
        else:
            # Restore best model manually if early stopping was disabled
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            history.best_epoch = best_epoch
            history.best_test_ar = best_test_ar or 0.0
        
        history.total_training_time_seconds = time.time() - start_time
        
        # Final evaluation
        final_train = self._evaluate(model, train_loader)
        final_test = self._evaluate(model, test_loader)
        
        return TrainingResult(
            model=model,
            history=history,
            train_metrics=final_train,
            test_metrics=final_test,
            feature_names=feature_names
        )
    
    def _train_epoch(self, model, train_loader, loss_fn, optimizer, config):
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        breakdown = {'bce': 0.0, 'auc': 0.0}
        n_batches = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            y_pred = model(X)
            
            if isinstance(loss_fn, CombinedLoss):
                loss, bd = loss_fn(y_pred, y)
                breakdown['bce'] += bd.get('bce', 0)
                breakdown['auc'] += bd.get('auc', 0)
            else:
                # BCE loss expects matching shapes - both should be (batch_size, 1) or (batch_size,)
                # Keep both as (batch_size, 1) to match CombinedLoss interface
                loss = loss_fn(y_pred, y)
            
            # L1 regularization
            if config.regularization.l1_lambda > 0:
                l1 = sum(p.abs().sum() for p in model.parameters())
                loss = loss + config.regularization.l1_lambda * l1
            
            loss.backward()
            
            # Gradient clipping
            if config.regularization.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.regularization.gradient_clip_norm
                )
            
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches, {k: v / n_batches for k, v in breakdown.items()}
    
    def _compute_loss(self, model, loader, loss_fn):
        """Compute loss without training."""
        model.eval()
        total_loss = 0.0
        breakdown = {'bce': 0.0, 'auc': 0.0}
        n_batches = 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = model(X)
                
                if isinstance(loss_fn, CombinedLoss):
                    loss, bd = loss_fn(y_pred, y)
                    breakdown['bce'] += bd.get('bce', 0)
                    breakdown['auc'] += bd.get('auc', 0)
                else:
                    # BCE loss expects matching shapes - both should be (batch_size, 1) or (batch_size,)
                    # Keep both as (batch_size, 1) to match CombinedLoss interface
                    loss = loss_fn(y_pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches, {k: v / n_batches for k, v in breakdown.items()}
    
    def _evaluate(self, model, loader) -> CompleteMetrics:
        """Evaluate model and compute all metrics."""
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                preds = model(X).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y.numpy())
        
        y_pred = np.concatenate(all_preds).flatten()
        y_true = np.concatenate(all_targets).flatten()
        
        return self.metrics_calc.calculate_all(y_true, y_pred)
