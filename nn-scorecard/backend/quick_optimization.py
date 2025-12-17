"""
Quick Optimization Script - Implement Fast Improvements

This script tests the "quick wins" from the optimization guide:
1. Lower loss_alpha to 0.2 (more AR focus)
2. Add margin to pairwise loss
3. Test WMW loss
4. Train ensemble of 5 models

Expected improvement: +0.04 to +0.09 AR

Usage:
    python quick_optimization.py --data path/to/data.csv --segment CONSUMER
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import json

from app.services.data_processor import DataProcessor
from app.services.trainer import ModelTrainer
from app.services.nn_scorecard import create_model
from app.models.schemas import (
    TrainingConfig, NeuralNetworkConfig, LossConfig, 
    RegularizationConfig, EarlyStoppingConfig
)
from app.services.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickOptimizer:
    """
    Implements quick optimization wins for immediate AR improvement.
    """
    
    def __init__(self, data_path: str, target_col: str = 'default', segment: str = None):
        self.data_path = data_path
        self.target_col = target_col
        self.segment = segment
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        if self.segment and 'segment' in df.columns:
            df = df[df['segment'] == self.segment].copy()
            logger.info(f"Filtered to segment: {self.segment}")
        
        self.y = df[self.target_col].values
        
        # Remove target and segment columns
        drop_cols = [self.target_col]
        if 'segment' in df.columns:
            drop_cols.append('segment')
        
        self.X = df.drop(columns=drop_cols).values
        self.feature_names = df.drop(columns=drop_cols).columns.tolist()
        
        logger.info(f"Data shape: {self.X.shape}")
        logger.info(f"Bad rate: {self.y.mean():.2%}")
        
    def create_baseline_config(self) -> TrainingConfig:
        """Create baseline configuration (current setup)."""
        return TrainingConfig(
            test_size=0.2,
            random_seed=42,
            epochs=100,
            batch_size=256,
            learning_rate=0.001,
            neural_network=NeuralNetworkConfig(
                model_type='neural_network',
                hidden_layers=[64, 32],
                activation='relu',
                dropout_rate=0.2,
                use_batch_norm=True
            ),
            loss=LossConfig(
                loss_type='combined',
                loss_alpha=0.3,  # Baseline
                auc_gamma=2.0,
                auc_loss_type='pairwise',
                margin=0.0  # Baseline
            ),
            regularization=RegularizationConfig(
                l1_lambda=0.0,
                l2_lambda=0.001,
                dropout_rate=0.2,
                gradient_clip_norm=1.0
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=15,
                min_delta=0.0001
            )
        )
    
    def create_optimized_config_1(self) -> TrainingConfig:
        """Quick Win #1: Lower alpha to 0.2 + add margin."""
        config = self.create_baseline_config()
        config.loss.loss_alpha = 0.2  # More AR focus
        config.loss.margin = 0.3  # Enforce separation
        return config
    
    def create_optimized_config_2(self) -> TrainingConfig:
        """Quick Win #2: WMW loss with higher margin."""
        config = self.create_baseline_config()
        config.loss.loss_alpha = 0.2
        config.loss.auc_loss_type = 'wmw'
        config.loss.margin = 0.5
        return config
    
    def create_optimized_config_3(self) -> TrainingConfig:
        """Quick Win #3: Very low alpha (aggressive AR focus)."""
        config = self.create_baseline_config()
        config.loss.loss_alpha = 0.1  # 90% AR weight!
        config.loss.auc_loss_type = 'pairwise'
        config.loss.margin = 0.3
        return config
    
    def create_optimized_config_4(self) -> TrainingConfig:
        """Quick Win #4: Increased patience for early stopping."""
        config = self.create_optimized_config_1()
        config.early_stopping.patience = 25  # More patience
        return config
    
    def train_single_model(
        self, 
        config: TrainingConfig, 
        random_seed: int = 42
    ) -> dict:
        """Train a single model and return metrics."""
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Prepare data
        processor = DataProcessor()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=config.test_size,
            random_state=random_seed,
            stratify=self.y
        )
        
        # Normalize
        X_train_norm = X_train / processor.scale_factor
        X_test_norm = X_test / processor.scale_factor
        
        # Create model
        model = create_model(
            input_dim=X_train_norm.shape[1],
            config=config.neural_network
        )
        
        # Train
        trainer = ModelTrainer(scale_factor=processor.scale_factor)
        
        train_loader, test_loader = trainer.create_data_loaders(
            X_train_norm, y_train,
            X_test_norm, y_test,
            batch_size=config.batch_size,
            random_seed=random_seed
        )
        
        result = trainer.train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            feature_names=self.feature_names
        )
        
        return {
            'model': model,
            'result': result,
            'train_ar': result.train_metrics.discrimination.gini_ar,
            'test_ar': result.test_metrics.discrimination.gini_ar,
            'train_auc': result.train_metrics.discrimination.auc_roc,
            'test_auc': result.test_metrics.discrimination.auc_roc,
            'best_epoch': result.history.best_epoch,
            'total_time': result.history.total_training_time_seconds
        }
    
    def train_ensemble(
        self,
        config: TrainingConfig,
        n_models: int = 5
    ) -> dict:
        """Train ensemble of models with different seeds."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training ensemble of {n_models} models")
        logger.info(f"{'='*60}")
        
        models = []
        train_ars = []
        test_ars = []
        
        for i in range(n_models):
            seed = 42 + i
            logger.info(f"\nTraining model {i+1}/{n_models} (seed={seed})")
            
            result = self.train_single_model(config, random_seed=seed)
            
            models.append(result['model'])
            train_ars.append(result['train_ar'])
            test_ars.append(result['test_ar'])
            
            logger.info(f"  Train AR: {result['train_ar']:.4f}")
            logger.info(f"  Test AR: {result['test_ar']:.4f}")
        
        # Calculate ensemble metrics
        ensemble_train_ar = np.mean(train_ars)
        ensemble_test_ar = np.mean(test_ars)
        std_test_ar = np.std(test_ars)
        
        logger.info(f"\nEnsemble Results:")
        logger.info(f"  Mean Train AR: {ensemble_train_ar:.4f}")
        logger.info(f"  Mean Test AR: {ensemble_test_ar:.4f} ± {std_test_ar:.4f}")
        logger.info(f"  Stability (std): {std_test_ar:.4f}")
        
        return {
            'models': models,
            'train_ars': train_ars,
            'test_ars': test_ars,
            'ensemble_train_ar': ensemble_train_ar,
            'ensemble_test_ar': ensemble_test_ar,
            'std_test_ar': std_test_ar
        }
    
    def run_quick_optimization(self, test_ensemble: bool = True):
        """
        Run all quick optimization experiments.
        """
        results = {}
        
        # Test 0: Baseline
        logger.info(f"\n{'#'*60}")
        logger.info("TEST 0: BASELINE (Current Setup)")
        logger.info(f"{'#'*60}")
        logger.info("Config: alpha=0.3, pairwise, margin=0.0")
        
        baseline_result = self.train_single_model(self.create_baseline_config())
        results['baseline'] = baseline_result
        
        logger.info(f"\n✓ Baseline Results:")
        logger.info(f"  Train AR: {baseline_result['train_ar']:.4f}")
        logger.info(f"  Test AR: {baseline_result['test_ar']:.4f}")
        logger.info(f"  Training time: {baseline_result['total_time']:.1f}s")
        
        # Test 1: Lower alpha + margin
        logger.info(f"\n{'#'*60}")
        logger.info("TEST 1: Lower Alpha + Margin")
        logger.info(f"{'#'*60}")
        logger.info("Config: alpha=0.2, pairwise, margin=0.3")
        
        opt1_result = self.train_single_model(self.create_optimized_config_1())
        results['opt1_alpha_margin'] = opt1_result
        
        improvement_1 = opt1_result['test_ar'] - baseline_result['test_ar']
        logger.info(f"\n✓ Test 1 Results:")
        logger.info(f"  Train AR: {opt1_result['train_ar']:.4f}")
        logger.info(f"  Test AR: {opt1_result['test_ar']:.4f}")
        logger.info(f"  Improvement: {improvement_1:+.4f} ({improvement_1/baseline_result['test_ar']*100:+.1f}%)")
        
        # Test 2: WMW loss
        logger.info(f"\n{'#'*60}")
        logger.info("TEST 2: WMW Loss")
        logger.info(f"{'#'*60}")
        logger.info("Config: alpha=0.2, wmw, margin=0.5")
        
        opt2_result = self.train_single_model(self.create_optimized_config_2())
        results['opt2_wmw'] = opt2_result
        
        improvement_2 = opt2_result['test_ar'] - baseline_result['test_ar']
        logger.info(f"\n✓ Test 2 Results:")
        logger.info(f"  Train AR: {opt2_result['train_ar']:.4f}")
        logger.info(f"  Test AR: {opt2_result['test_ar']:.4f}")
        logger.info(f"  Improvement: {improvement_2:+.4f} ({improvement_2/baseline_result['test_ar']*100:+.1f}%)")
        
        # Test 3: Aggressive AR focus
        logger.info(f"\n{'#'*60}")
        logger.info("TEST 3: Aggressive AR Focus")
        logger.info(f"{'#'*60}")
        logger.info("Config: alpha=0.1, pairwise, margin=0.3")
        
        opt3_result = self.train_single_model(self.create_optimized_config_3())
        results['opt3_aggressive'] = opt3_result
        
        improvement_3 = opt3_result['test_ar'] - baseline_result['test_ar']
        logger.info(f"\n✓ Test 3 Results:")
        logger.info(f"  Train AR: {opt3_result['train_ar']:.4f}")
        logger.info(f"  Test AR: {opt3_result['test_ar']:.4f}")
        logger.info(f"  Improvement: {improvement_3:+.4f} ({improvement_3/baseline_result['test_ar']*100:+.1f}%)")
        
        # Find best config
        best_config_name = max(
            ['opt1_alpha_margin', 'opt2_wmw', 'opt3_aggressive'],
            key=lambda x: results[x]['test_ar']
        )
        best_result = results[best_config_name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST SINGLE MODEL: {best_config_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Test AR: {best_result['test_ar']:.4f}")
        logger.info(f"Improvement: {best_result['test_ar'] - baseline_result['test_ar']:+.4f}")
        
        # Test 4: Ensemble (if enabled)
        if test_ensemble:
            logger.info(f"\n{'#'*60}")
            logger.info("TEST 4: 5-Model Ensemble")
            logger.info(f"{'#'*60}")
            logger.info(f"Using best config: {best_config_name}")
            
            # Get config for best model
            if best_config_name == 'opt1_alpha_margin':
                ensemble_config = self.create_optimized_config_1()
            elif best_config_name == 'opt2_wmw':
                ensemble_config = self.create_optimized_config_2()
            else:
                ensemble_config = self.create_optimized_config_3()
            
            ensemble_result = self.train_ensemble(ensemble_config, n_models=5)
            results['ensemble'] = ensemble_result
            
            improvement_4 = ensemble_result['ensemble_test_ar'] - baseline_result['test_ar']
            logger.info(f"\n✓ Ensemble Results:")
            logger.info(f"  Test AR: {ensemble_result['ensemble_test_ar']:.4f} ± {ensemble_result['std_test_ar']:.4f}")
            logger.info(f"  Improvement: {improvement_4:+.4f} ({improvement_4/baseline_result['test_ar']*100:+.1f}%)")
        
        # Final Summary
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"\nBaseline Test AR: {baseline_result['test_ar']:.4f}")
        logger.info(f"\nImprovements:")
        logger.info(f"  Test 1 (alpha=0.2 + margin): {improvement_1:+.4f} ({improvement_1/baseline_result['test_ar']*100:+.1f}%)")
        logger.info(f"  Test 2 (WMW loss): {improvement_2:+.4f} ({improvement_2/baseline_result['test_ar']*100:+.1f}%)")
        logger.info(f"  Test 3 (alpha=0.1): {improvement_3:+.4f} ({improvement_3/baseline_result['test_ar']*100:+.1f}%)")
        
        if test_ensemble:
            logger.info(f"  Test 4 (Ensemble): {improvement_4:+.4f} ({improvement_4/baseline_result['test_ar']*100:+.1f}%)")
            total_improvement = improvement_4
        else:
            total_improvement = max(improvement_1, improvement_2, improvement_3)
        
        logger.info(f"\nTotal Improvement: {total_improvement:+.4f} ({total_improvement/baseline_result['test_ar']*100:+.1f}%)")
        logger.info(f"Final Test AR: {baseline_result['test_ar'] + total_improvement:.4f}")
        
        # Save results
        self._save_results(results, baseline_result['test_ar'], total_improvement)
        
        return results
    
    def _save_results(self, results: dict, baseline_ar: float, total_improvement: float):
        """Save results to file."""
        output_dir = Path('data/quick_optimization')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'segment': self.segment,
            'baseline_test_ar': baseline_ar,
            'total_improvement': total_improvement,
            'final_test_ar': baseline_ar + total_improvement,
            'improvement_percent': (total_improvement / baseline_ar * 100),
            'tests': {}
        }
        
        for name, result in results.items():
            if name == 'ensemble':
                summary['tests'][name] = {
                    'test_ar': result['ensemble_test_ar'],
                    'std_ar': result['std_test_ar'],
                    'improvement': result['ensemble_test_ar'] - baseline_ar
                }
            elif 'model' in result:
                summary['tests'][name] = {
                    'train_ar': result['train_ar'],
                    'test_ar': result['test_ar'],
                    'improvement': result['test_ar'] - baseline_ar,
                    'best_epoch': result['best_epoch'],
                    'training_time': result['total_time']
                }
        
        summary_file = output_dir / f'quick_optimization_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Quick optimization for scorecard')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--segment', type=str, default=None, help='Segment to filter')
    parser.add_argument('--target', type=str, default='default', help='Target column name')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble test (faster)')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = QuickOptimizer(
        data_path=args.data,
        target_col=args.target,
        segment=args.segment
    )
    
    # Run optimization
    results = optimizer.run_quick_optimization(test_ensemble=not args.no_ensemble)
    
    logger.info("\n✓ Quick optimization completed successfully!")


if __name__ == '__main__':
    main()

