"""
Model Evaluation Metrics

This module calculates comprehensive model evaluation metrics for IFRS9 documentation
and model validation. All metrics are calculated for both train and test sets.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class DiscriminationMetrics:
    """Metrics measuring model's ability to rank-order risk."""
    auc_roc: float
    gini_ar: float              # = 2 * AUC - 1
    ks_statistic: float
    ks_decile: int              # Decile where max KS occurs
    c_statistic: float          # Same as AUC for binary


@dataclass
class CalibrationMetrics:
    """Metrics measuring probability accuracy."""
    log_loss: float
    brier_score: float


@dataclass  
class ClassificationMetrics:
    """Classification metrics at 0.5 threshold."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float


@dataclass
class ConfusionMatrixMetrics:
    """Confusion matrix components."""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total: int


@dataclass
class CaptureRateMetrics:
    """Bad capture rates at various percentiles."""
    bad_rate_top_5pct: float
    bad_rate_top_10pct: float
    bad_rate_top_20pct: float
    bad_rate_top_30pct: float


@dataclass
class LiftMetrics:
    """Lift metrics by decile."""
    lift_top_decile: float
    lift_top_2_deciles: float
    lift_top_3_deciles: float
    cumulative_lift: List[float]


@dataclass
class CompleteMetrics:
    """All metrics combined."""
    discrimination: DiscriminationMetrics
    calibration: CalibrationMetrics
    classification: ClassificationMetrics
    confusion_matrix: ConfusionMatrixMetrics
    capture_rates: CaptureRateMetrics
    lift: LiftMetrics


class MetricsCalculator:
    """
    Calculate comprehensive model evaluation metrics.
    
    All metrics support IFRS9 documentation requirements.
    """
    
    def __init__(self, n_deciles: int = 10):
        self.n_deciles = n_deciles
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> CompleteMetrics:
        """Calculate all metrics."""
        y_pred_class = (y_pred_proba >= threshold).astype(int)
        
        return CompleteMetrics(
            discrimination=self._calc_discrimination(y_true, y_pred_proba),
            calibration=self._calc_calibration(y_true, y_pred_proba),
            classification=self._calc_classification(y_true, y_pred_class),
            confusion_matrix=self._calc_confusion_matrix(y_true, y_pred_class),
            capture_rates=self._calc_capture_rates(y_true, y_pred_proba),
            lift=self._calc_lift(y_true, y_pred_proba)
        )
    
    def _calc_discrimination(self, y_true, y_pred_proba) -> DiscriminationMetrics:
        """Calculate AUC, Gini, KS."""
        # Handle edge case: only one class present (AUC is undefined)
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # Return default values when AUC cannot be calculated
            auc = 0.5  # Undefined, use neutral value
            gini = 0.0
            ks_stat = 0.0
            ks_decile = 1
        else:
            auc = roc_auc_score(y_true, y_pred_proba)
            gini = 2 * auc - 1
            ks_stat, ks_decile = self._calc_ks_statistic(y_true, y_pred_proba)
        
        return DiscriminationMetrics(
            auc_roc=float(auc),
            gini_ar=float(gini),
            ks_statistic=float(ks_stat),
            ks_decile=int(ks_decile),
            c_statistic=float(auc)
        )
    
    def _calc_ks_statistic(self, y_true, y_pred_proba) -> Tuple[float, int]:
        """Calculate KS statistic and decile where it occurs."""
        sorted_indices = np.argsort(-y_pred_proba)
        y_true_sorted = y_true[sorted_indices]
        
        n_total = len(y_true)
        n_bad = y_true.sum()
        n_good = n_total - n_bad
        
        if n_bad == 0 or n_good == 0:
            return 0.0, 1
        
        cum_bad = np.cumsum(y_true_sorted) / n_bad
        cum_good = np.cumsum(1 - y_true_sorted) / n_good
        
        ks_values = np.abs(cum_bad - cum_good)
        ks_stat = np.max(ks_values)
        ks_index = np.argmax(ks_values)
        ks_decile = int(np.ceil((ks_index + 1) / n_total * 10))
        
        return ks_stat, ks_decile
    
    def _calc_calibration(self, y_true, y_pred_proba) -> CalibrationMetrics:
        """Calculate log loss and Brier score."""
        # Handle edge case: only one class present
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # Log loss is undefined when only one class is present
            # Return a large finite value to indicate undefined metric
            log_loss_val = 1e10
            brier_score_val = float(brier_score_loss(y_true, y_pred_proba))
        else:
            log_loss_val = float(log_loss(y_true, y_pred_proba))
            brier_score_val = float(brier_score_loss(y_true, y_pred_proba))
        
        return CalibrationMetrics(
            log_loss=log_loss_val,
            brier_score=brier_score_val
        )
    
    def _calc_classification(self, y_true, y_pred_class) -> ClassificationMetrics:
        """Calculate classification metrics."""
        cm = confusion_matrix(y_true, y_pred_class)
        # Handle edge case: only one class present (confusion matrix is 1x1)
        if cm.size == 1:
            # Only one class, so all predictions are the same
            if y_true[0] == 0:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
        else:
            tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred_class)),
            precision=float(precision_score(y_true, y_pred_class, zero_division=0)),
            recall=float(recall_val),
            f1_score=float(f1_score(y_true, y_pred_class, zero_division=0)),
            specificity=float(specificity),
            balanced_accuracy=float((recall_val + specificity) / 2)
        )
    
    def _calc_confusion_matrix(self, y_true, y_pred_class) -> ConfusionMatrixMetrics:
        """Calculate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred_class)
        # Handle edge case: only one class present (confusion matrix is 1x1)
        if cm.size == 1:
            # Only one class, so all predictions are the same
            if y_true[0] == 0:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
        else:
            tn, fp, fn, tp = cm.ravel()
        
        return ConfusionMatrixMetrics(
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            total=int(len(y_true))
        )
    
    def _calc_capture_rates(self, y_true, y_pred_proba) -> CaptureRateMetrics:
        """Calculate bad capture rates at various percentiles."""
        n_total = len(y_true)
        total_bads = y_true.sum()
        
        if total_bads == 0:
            return CaptureRateMetrics(0, 0, 0, 0)
        
        sorted_indices = np.argsort(-y_pred_proba)
        y_true_sorted = y_true[sorted_indices]
        
        def capture_at(pct):
            cutoff = int(n_total * pct)
            return float(y_true_sorted[:cutoff].sum() / total_bads)
        
        return CaptureRateMetrics(
            bad_rate_top_5pct=capture_at(0.05),
            bad_rate_top_10pct=capture_at(0.10),
            bad_rate_top_20pct=capture_at(0.20),
            bad_rate_top_30pct=capture_at(0.30)
        )
    
    def _calc_lift(self, y_true, y_pred_proba) -> LiftMetrics:
        """Calculate lift metrics."""
        n_total = len(y_true)
        overall_bad_rate = y_true.mean()
        
        if overall_bad_rate == 0:
            return LiftMetrics(0, 0, 0, [0] * 10)
        
        sorted_indices = np.argsort(-y_pred_proba)
        y_true_sorted = y_true[sorted_indices]
        
        decile_size = n_total // self.n_deciles
        cumulative_lift = []
        
        for i in range(self.n_deciles):
            end_idx = (i + 1) * decile_size
            # Handle edge case: when decile_size is 0 (n_total < n_deciles)
            if end_idx == 0:
                cumulative_lift.append(0.0)
                continue
            cum_bad_rate = y_true_sorted[:end_idx].sum() / end_idx
            lift = cum_bad_rate / overall_bad_rate
            cumulative_lift.append(float(lift))
        
        return LiftMetrics(
            lift_top_decile=cumulative_lift[0],
            lift_top_2_deciles=cumulative_lift[1],
            lift_top_3_deciles=cumulative_lift[2],
            cumulative_lift=cumulative_lift
        )
    
    def calculate_decile_table(self, y_true, y_pred_proba) -> List[Dict]:
        """Generate decile analysis table for documentation."""
        n_total = len(y_true)
        total_bads = y_true.sum()
        total_goods = n_total - total_bads
        overall_bad_rate = y_true.mean()
        
        sorted_indices = np.argsort(-y_pred_proba)
        y_true_sorted = y_true[sorted_indices]
        proba_sorted = y_pred_proba[sorted_indices]
        
        decile_size = n_total // self.n_deciles
        table = []
        cum_bads, cum_goods = 0, 0
        
        for i in range(self.n_deciles):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < self.n_deciles - 1 else n_total
            
            decile_y = y_true_sorted[start_idx:end_idx]
            decile_proba = proba_sorted[start_idx:end_idx]
            
            decile_bads = decile_y.sum()
            decile_goods = len(decile_y) - decile_bads
            
            cum_bads += decile_bads
            cum_goods += decile_goods
            
            cum_bad_pct = cum_bads / total_bads if total_bads > 0 else 0
            cum_good_pct = cum_goods / total_goods if total_goods > 0 else 0
            
            table.append({
                'decile': i + 1,
                'count': len(decile_y),
                'bad_count': int(decile_bads),
                'bad_rate': float(decile_y.mean()),
                'cum_bad_pct': float(cum_bad_pct),
                'cum_good_pct': float(cum_good_pct),
                'ks': float(abs(cum_bad_pct - cum_good_pct)),
                'lift': float(decile_y.mean() / overall_bad_rate) if overall_bad_rate > 0 else 0,
                'min_prob': float(decile_proba.min()),
                'max_prob': float(decile_proba.max())
            })
        
        return table


def calculate_metrics(y_true, y_pred_proba) -> CompleteMetrics:
    """Convenience function."""
    return MetricsCalculator().calculate_all(y_true, y_pred_proba)
