"""
Hyperparameter Search Script for Neural Network Scorecard Optimization

This script provides systematic hyperparameter optimization using grid search
or random search with cross-validation.

Usage:
    python hyperparameter_search.py --segment CONSUMER --mode quick
    python hyperparameter_search.py --segment CONSUMER --mode thorough
    python hyperparameter_search.py --segment CONSUMER --mode custom --config my_config.json
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import torch

from app.services.data_processor import DataProcessor
from app.services.trainer import ModelTrainer, TrainingResult
from app.services.nn_scorecard import create_model
from app.models.schemas import TrainingConfig, NeuralNetworkConfig, LossConfig, RegularizationConfig, EarlyStoppingConfig
from app.services.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """
    Systematic hyperparameter search for optimal scorecard performance.
    """
    
    def __init__(
        self,
        data_path: str,
        target_col: str = 'default',
        segment_col: str = None,
        segment: str = None,
        output_dir: str = 'data/experiments'
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.segment_col = segment_col
        self.segment = segment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data for experiments."""
        logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Filter by segment if specified
        if self.segment and self.segment_col:
            df = df[df[self.segment_col] == self.segment].copy()
            logger.info(f"Filtered to segment: {self.segment}, rows: {len(df)}")
        
        # Separate features and target
        self.y = df[self.target_col].values
        self.X = df.drop(columns=[self.target_col, self.segment_col] if self.segment_col else [self.target_col]).values
        self.feature_names = df.drop(columns=[self.target_col, self.segment_col] if self.segment_col else [self.target_col]).columns.tolist()
        
        logger.info(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        logger.info(f"Bad rate: {self.y.mean():.2%}")
        
    def get_quick_search_space(self) -> List[Dict[str, Any]]:
        """
        Quick search space for fast experiments (9-12 configurations).
        Focus on loss function and simple architectures.
        """
        configs = []
        
        # Loss configurations (highest impact)
        loss_configs = [
            {'loss_alpha': 0.1, 'auc_loss_type': 'pairwise', 'margin': 0.3},
            {'loss_alpha': 0.2, 'auc_loss_type': 'pairwise', 'margin': 0.3},
            {'loss_alpha': 0.2, 'auc_loss_type': 'wmw', 'margin': 0.5},
        ]
        
        # Simple architectures
        architectures = [
            {'hidden_layers': [], 'model_type': 'linear'},
            {'hidden_layers': [64, 32], 'model_type': 'neural_network'},
            {'hidden_layers': [128, 64, 32], 'model_type': 'neural_network'},
        ]
        
        for loss_cfg in loss_configs:
            for arch in architectures:
                config = {
                    'architecture': arch,
                    'loss': loss_cfg,
                    'learning_rate': 0.001,
                    'dropout_rate': 0.2,
                    'l2_lambda': 0.001,
                    'epochs': 100
                }
                configs.append(config)
        
        return configs
    
    def get_thorough_search_space(self) -> List[Dict[str, Any]]:
        """
        Thorough search space for comprehensive optimization (50-100 configurations).
        """
        configs = []
        
        # Loss configurations
        loss_alpha_values = [0.1, 0.2, 0.3, 0.4]
        auc_types = ['pairwise', 'soft', 'wmw']
        margin_values = [0.0, 0.3, 0.5]
        
        # Architectures
        architectures = [
            {'hidden_layers': [], 'model_type': 'linear'},
            {'hidden_layers': [32], 'model_type': 'neural_network'},
            {'hidden_layers': [64], 'model_type': 'neural_network'},
            {'hidden_layers': [64, 32], 'model_type': 'neural_network'},
            {'hidden_layers': [128, 64], 'model_type': 'neural_network'},
            {'hidden_layers': [128, 64, 32], 'model_type': 'neural_network'},
            {'hidden_layers': [64, 32, 16], 'model_type': 'neural_network'},
        ]
        
        # Regularization
        dropout_values = [0.1, 0.2, 0.3]
        l2_values = [0.0001, 0.001, 0.01]
        
        # Learning rates
        lr_values = [0.0003, 0.001, 0.003]
        
        # Generate combinations (sample to keep reasonable size)
        import itertools
        
        # Focus on most important: loss config × architecture
        for loss_alpha in loss_alpha_values:
            for auc_type in auc_types:
                for arch in architectures:
                    # Use smart defaults for other params
                    margin = 0.3 if auc_type != 'soft' else 0.0
                    dropout = 0.2 if arch['model_type'] != 'linear' else 0.0
                    
                    config = {
                        'architecture': arch,
                        'loss': {
                            'loss_alpha': loss_alpha,
                            'auc_loss_type': auc_type,
                            'margin': margin,
                            'auc_gamma': 5.0 if auc_type == 'soft' else 2.0
                        },
                        'learning_rate': 0.001,
                        'dropout_rate': dropout,
                        'l2_lambda': 0.001,
                        'epochs': 150
                    }
                    configs.append(config)
        
        logger.info(f"Generated {len(configs)} configurations for thorough search")
        return configs
    
    def train_with_cv(
        self,
        config: Dict[str, Any],
        n_folds: int = 5,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.
        
        Returns:
            Dictionary with mean and std of AR across folds
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        ar_scores = []
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            logger.info(f"  Fold {fold + 1}/{n_folds}")
            
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Prepare data
            processor = DataProcessor()
            
            # Create normalized versions
            X_train_norm = X_train / processor.scale_factor
            X_val_norm = X_val / processor.scale_factor
            
            # Create model and trainer
            training_config = self._dict_to_training_config(config)
            model = create_model(
                input_dim=X_train_norm.shape[1],
                config=training_config.neural_network
            )
            
            trainer = ModelTrainer(scale_factor=processor.scale_factor)
            
            # Create data loaders
            train_loader, val_loader = trainer.create_data_loaders(
                X_train_norm, y_train,
                X_val_norm, y_val,
                batch_size=256,
                random_seed=random_seed + fold
            )
            
            # Train
            try:
                result = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    test_loader=val_loader,
                    config=training_config,
                    feature_names=self.feature_names
                )
                
                # Record metrics
                ar_scores.append(result.test_metrics.discrimination.gini_ar)
                auc_scores.append(result.test_metrics.discrimination.auc_roc)
                
            except Exception as e:
                logger.error(f"    Error in fold {fold + 1}: {e}")
                ar_scores.append(0.0)
                auc_scores.append(0.5)
        
        return {
            'mean_ar': float(np.mean(ar_scores)),
            'std_ar': float(np.std(ar_scores)),
            'mean_auc': float(np.mean(auc_scores)),
            'std_auc': float(np.std(auc_scores)),
            'all_ar': ar_scores,
            'all_auc': auc_scores
        }
    
    def _dict_to_training_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Convert dictionary configuration to TrainingConfig object."""
        arch = config_dict['architecture']
        loss = config_dict['loss']
        
        return TrainingConfig(
            test_size=0.2,
            random_seed=42,
            epochs=config_dict.get('epochs', 100),
            batch_size=256,
            learning_rate=config_dict.get('learning_rate', 0.001),
            neural_network=NeuralNetworkConfig(
                model_type=arch.get('model_type', 'neural_network'),
                hidden_layers=arch.get('hidden_layers', [64, 32]),
                activation=config_dict.get('activation', 'relu'),
                dropout_rate=config_dict.get('dropout_rate', 0.2),
                use_batch_norm=True,
                skip_connection=arch.get('skip_connection', False)
            ),
            loss=LossConfig(
                loss_type='combined',
                loss_alpha=loss.get('loss_alpha', 0.3),
                auc_gamma=loss.get('auc_gamma', 2.0),
                auc_loss_type=loss.get('auc_loss_type', 'pairwise'),
                margin=loss.get('margin', 0.0)
            ),
            regularization=RegularizationConfig(
                l1_lambda=config_dict.get('l1_lambda', 0.0),
                l2_lambda=config_dict.get('l2_lambda', 0.001),
                dropout_rate=config_dict.get('dropout_rate', 0.2),
                gradient_clip_norm=1.0
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=15,
                min_delta=0.0001
            )
        )
    
    def run_search(
        self,
        search_space: List[Dict[str, Any]],
        n_folds: int = 5,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Run hyperparameter search over search space.
        
        Args:
            search_space: List of configuration dictionaries
            n_folds: Number of cross-validation folds
            top_k: Number of top configurations to return
            
        Returns:
            DataFrame with results sorted by mean AR
        """
        results = []
        
        logger.info(f"Starting search over {len(search_space)} configurations")
        logger.info(f"Using {n_folds}-fold cross-validation")
        
        for i, config in enumerate(search_space):
            logger.info(f"\n{'='*60}")
            logger.info(f"Configuration {i+1}/{len(search_space)}")
            logger.info(f"{'='*60}")
            logger.info(f"Architecture: {config['architecture']}")
            logger.info(f"Loss: alpha={config['loss']['loss_alpha']}, type={config['loss']['auc_loss_type']}")
            
            try:
                cv_results = self.train_with_cv(config, n_folds=n_folds)
                
                result = {
                    'config_id': i,
                    'architecture': str(config['architecture']['hidden_layers']),
                    'loss_alpha': config['loss']['loss_alpha'],
                    'auc_loss_type': config['loss']['auc_loss_type'],
                    'margin': config['loss']['margin'],
                    'dropout': config.get('dropout_rate', 0.2),
                    'l2_lambda': config.get('l2_lambda', 0.001),
                    'learning_rate': config.get('learning_rate', 0.001),
                    'mean_ar': cv_results['mean_ar'],
                    'std_ar': cv_results['std_ar'],
                    'mean_auc': cv_results['mean_auc'],
                    'std_auc': cv_results['std_auc'],
                    'config': config
                }
                
                results.append(result)
                
                logger.info(f"✓ Results: AR = {cv_results['mean_ar']:.4f} ± {cv_results['std_ar']:.4f}")
                
            except Exception as e:
                logger.error(f"✗ Configuration failed: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('mean_ar', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"search_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Save top K configs
        top_configs_file = self.output_dir / f"top_{top_k}_configs_{timestamp}.json"
        top_configs = df.head(top_k)['config'].tolist()
        with open(top_configs_file, 'w') as f:
            json.dump(top_configs, f, indent=2)
        logger.info(f"Top {top_k} configs saved to: {top_configs_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SEARCH SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"\nTop {top_k} Configurations:\n")
        
        for i, row in df.head(top_k).iterrows():
            logger.info(f"Rank {i+1}:")
            logger.info(f"  AR: {row['mean_ar']:.4f} ± {row['std_ar']:.4f}")
            logger.info(f"  Architecture: {row['architecture']}")
            logger.info(f"  Loss: alpha={row['loss_alpha']}, type={row['auc_loss_type']}, margin={row['margin']}")
            logger.info(f"  Regularization: dropout={row['dropout']}, L2={row['l2_lambda']}")
            logger.info("")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for scorecard optimization')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--segment', type=str, default=None, help='Segment to filter')
    parser.add_argument('--segment-col', type=str, default='segment', help='Segment column name')
    parser.add_argument('--target', type=str, default='default', help='Target column name')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'thorough', 'custom'],
                        help='Search mode: quick (9-12 configs), thorough (50-100 configs), or custom')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config JSON (for custom mode)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top configs to save')
    parser.add_argument('--output', type=str, default='data/experiments', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize search
    search = HyperparameterSearch(
        data_path=args.data,
        target_col=args.target,
        segment_col=args.segment_col if args.segment else None,
        segment=args.segment,
        output_dir=args.output
    )
    
    # Get search space
    if args.mode == 'quick':
        logger.info("Using QUICK search mode (fast, ~1-2 hours)")
        search_space = search.get_quick_search_space()
    elif args.mode == 'thorough':
        logger.info("Using THOROUGH search mode (comprehensive, ~1-2 days)")
        search_space = search.get_thorough_search_space()
    elif args.mode == 'custom':
        if not args.config:
            raise ValueError("--config required for custom mode")
        logger.info(f"Loading custom search space from {args.config}")
        with open(args.config, 'r') as f:
            search_space = json.load(f)
    
    # Run search
    results = search.run_search(
        search_space=search_space,
        n_folds=args.folds,
        top_k=args.top_k
    )
    
    logger.info("\n✓ Search completed successfully!")
    logger.info(f"Best AR: {results.iloc[0]['mean_ar']:.4f} ± {results.iloc[0]['std_ar']:.4f}")


if __name__ == '__main__':
    main()

