"""
Feature Selection and Reduction Script

Automatically identifies and removes low-value features to improve
model performance and reduce training time.

Usage:
    python feature_selection.py --data path/to/data.csv --segment CONSUMER
    python feature_selection.py --data path/to/data.csv --target-features 15
    python feature_selection.py --data path/to/data.csv --method all
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
from app.models.schemas import TrainingConfig, NeuralNetworkConfig, LossConfig, RegularizationConfig, EarlyStoppingConfig
from app.services.metrics import MetricsCalculator

from sklearn.model_selection import train_test_split, StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection and reduction for scorecard optimization.
    """
    
    def __init__(
        self,
        data_path: str,
        target_col: str = 'default',
        segment: str = None,
        output_dir: str = 'data/feature_selection'
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.segment = segment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        drop_cols = [self.target_col]
        if 'segment' in df.columns:
            drop_cols.append('segment')
        
        self.X = df.drop(columns=drop_cols).values
        self.feature_names = df.drop(columns=drop_cols).columns.tolist()
        
        logger.info(f"Data shape: {self.X.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Bad rate: {self.y.mean():.2%}")
    
    def calculate_information_value(self) -> pd.DataFrame:
        """
        Calculate Information Value (IV) for each feature.
        Standard metric in credit risk for feature discriminatory power.
        """
        logger.info("\nCalculating Information Value (IV) for features...")
        
        results = []
        total_good = (self.y == 0).sum()
        total_bad = (self.y == 1).sum()
        
        for i, feat_name in enumerate(self.feature_names):
            feature_values = self.X[:, i]
            unique_vals = np.unique(feature_values)
            
            iv = 0
            n_bins = len(unique_vals)
            
            for val in unique_vals:
                mask = (feature_values == val)
                n_good = ((self.y == 0) & mask).sum()
                n_bad = ((self.y == 1) & mask).sum()
                
                if n_good > 0 and n_bad > 0:
                    pct_good = n_good / total_good
                    pct_bad = n_bad / total_bad
                    
                    woe = np.log(pct_good / pct_bad)
                    iv += (pct_good - pct_bad) * woe
            
            rating = self._get_iv_rating(abs(iv))
            
            results.append({
                'feature': feat_name,
                'iv': abs(iv),
                'rating': rating,
                'n_bins': n_bins
            })
        
        df_iv = pd.DataFrame(results).sort_values('iv', ascending=False)
        
        logger.info("\nInformation Value Results:")
        logger.info(df_iv.to_string(index=False))
        
        logger.info(f"\nSummary:")
        logger.info(f"  Very Strong (IV > 0.5): {(df_iv['rating'] == 'Very Strong').sum()}")
        logger.info(f"  Strong (0.3-0.5):       {(df_iv['rating'] == 'Strong').sum()}")
        logger.info(f"  Medium (0.1-0.3):       {(df_iv['rating'] == 'Medium').sum()}")
        logger.info(f"  Weak (0.02-0.1):        {(df_iv['rating'] == 'Weak').sum()}")
        logger.info(f"  Useless (< 0.02):       {(df_iv['rating'] == 'Useless').sum()}")
        
        return df_iv
    
    def _get_iv_rating(self, iv: float) -> str:
        """Rate IV strength."""
        if iv < 0.02:
            return 'Useless'
        elif iv < 0.1:
            return 'Weak'
        elif iv < 0.3:
            return 'Medium'
        elif iv < 0.5:
            return 'Strong'
        else:
            return 'Very Strong'
    
    def calculate_model_importance(self, quick: bool = True) -> pd.DataFrame:
        """
        Train model and extract feature importance from weights.
        """
        logger.info("\nTraining model to extract feature importance...")
        
        # Quick config for fast training
        epochs = 50 if quick else 100
        config = self._create_config(epochs=epochs)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Normalize
        processor = DataProcessor()
        X_train_norm = X_train / processor.scale_factor
        X_test_norm = X_test / processor.scale_factor
        
        # Train
        model = create_model(
            input_dim=X_train_norm.shape[1],
            config=config.neural_network
        )
        
        trainer = ModelTrainer(scale_factor=processor.scale_factor)
        train_loader, test_loader = trainer.create_data_loaders(
            X_train_norm, y_train,
            X_test_norm, y_test,
            batch_size=256,
            random_seed=42
        )
        
        result = trainer.train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            feature_names=self.feature_names
        )
        
        logger.info(f"Model trained: AR = {result.test_metrics.discrimination.gini_ar:.4f}")
        
        # Extract importance
        importance = self._extract_importance(model)
        importance = importance / importance.sum()  # Normalize
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance * 100
        }).sort_values('importance', ascending=False)
        
        df_importance['cumulative_pct'] = df_importance['importance_pct'].cumsum()
        
        logger.info("\nModel-Based Feature Importance:")
        logger.info(df_importance.to_string(index=False))
        
        logger.info(f"\nCumulative Coverage:")
        for n in [5, 10, 15, 20]:
            if n <= len(df_importance):
                cum_pct = df_importance.iloc[:n]['cumulative_pct'].iloc[-1]
                logger.info(f"  Top {n:2d} features: {cum_pct:5.1f}%")
        
        return df_importance
    
    def _extract_importance(self, model: torch.nn.Module) -> np.ndarray:
        """Extract feature importance from model."""
        # Try get_feature_weights method first
        if hasattr(model, 'get_feature_weights'):
            weights = model.get_feature_weights()
            if isinstance(weights, torch.Tensor):
                return np.abs(weights.cpu().numpy())
            return np.abs(weights)
        
        # Manual extraction
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weights = module.weight.data.cpu().numpy()
                
                if weights.shape[1] == len(self.feature_names):
                    if weights.shape[0] == 1:  # Linear model
                        return np.abs(weights.flatten())
                    else:  # Neural network
                        return np.abs(weights).mean(axis=0)
        
        raise ValueError("Could not extract importance from model")
    
    def calculate_correlation_matrix(self, threshold: float = 0.85) -> pd.DataFrame:
        """
        Identify highly correlated feature pairs.
        """
        logger.info(f"\nCalculating feature correlations (threshold={threshold})...")
        
        df = pd.DataFrame(self.X, columns=self.feature_names)
        corr_matrix = df.corr().abs()
        
        # Find high correlation pairs
        pairs = []
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                corr = corr_matrix.iloc[i, j]
                if corr > threshold:
                    pairs.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'correlation': corr
                    })
        
        df_corr = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
        
        if len(df_corr) > 0:
            logger.info(f"\nFound {len(df_corr)} highly correlated pairs:")
            logger.info(df_corr.to_string(index=False))
        else:
            logger.info(f"\nNo highly correlated pairs found (threshold={threshold})")
        
        return df_corr
    
    def select_features(
        self,
        method: str = 'importance',
        target_n: int = None,
        importance_threshold: float = 1.0,
        iv_threshold: float = 0.1
    ) -> list:
        """
        Select features based on specified method.
        
        Args:
            method: 'importance' (model-based), 'iv' (statistical), or 'combined'
            target_n: Target number of features (if None, use threshold)
            importance_threshold: Min importance % for 'importance' method
            iv_threshold: Min IV for 'iv' method
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Selection: method={method}")
        logger.info(f"{'='*60}")
        
        if method == 'importance':
            df_importance = self.calculate_model_importance()
            
            if target_n:
                selected = df_importance.head(target_n)['feature'].tolist()
            else:
                selected = df_importance[df_importance['importance_pct'] >= importance_threshold]['feature'].tolist()
            
        elif method == 'iv':
            df_iv = self.calculate_information_value()
            
            if target_n:
                selected = df_iv.head(target_n)['feature'].tolist()
            else:
                selected = df_iv[df_iv['iv'] >= iv_threshold]['feature'].tolist()
        
        elif method == 'combined':
            # Use both IV and model importance
            df_iv = self.calculate_information_value()
            df_importance = self.calculate_model_importance()
            
            # Merge
            df_combined = df_iv.merge(
                df_importance[['feature', 'importance_pct']], 
                on='feature'
            )
            
            # Combined score (normalized)
            df_combined['iv_norm'] = df_combined['iv'] / df_combined['iv'].max()
            df_combined['imp_norm'] = df_combined['importance_pct'] / 100
            df_combined['combined_score'] = (df_combined['iv_norm'] + df_combined['imp_norm']) / 2
            df_combined = df_combined.sort_values('combined_score', ascending=False)
            
            logger.info("\nCombined Scores:")
            logger.info(df_combined[['feature', 'iv', 'importance_pct', 'combined_score']].to_string(index=False))
            
            if target_n:
                selected = df_combined.head(target_n)['feature'].tolist()
            else:
                # Use mean of thresholds
                selected = df_combined[
                    (df_combined['iv'] >= iv_threshold) & 
                    (df_combined['importance_pct'] >= importance_threshold)
                ]['feature'].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"\nSelected {len(selected)} features:")
        for i, feat in enumerate(selected, 1):
            logger.info(f"  {i:2d}. {feat}")
        
        return selected
    
    def compare_feature_sets(
        self,
        feature_sets: dict,
        cv_folds: int = 3
    ) -> pd.DataFrame:
        """
        Compare different feature sets using cross-validation.
        
        Args:
            feature_sets: Dict of {name: [feature_list]}
            cv_folds: Number of CV folds
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing Feature Sets (CV={cv_folds})")
        logger.info(f"{'='*60}")
        
        results = []
        
        for name, features in feature_sets.items():
            logger.info(f"\nTesting '{name}' ({len(features)} features)...")
            
            # Get feature indices
            feature_indices = [self.feature_names.index(f) for f in features]
            X_subset = self.X[:, feature_indices]
            
            # Cross-validate
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            ar_scores = []
            training_times = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_subset, self.y)):
                logger.info(f"  Fold {fold+1}/{cv_folds}...")
                
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Train
                start_time = datetime.now()
                result = self._train_quick(X_train, y_train, X_val, y_val, features)
                training_time = (datetime.now() - start_time).total_seconds()
                
                ar_scores.append(result['ar'])
                training_times.append(training_time)
                
                logger.info(f"    AR: {result['ar']:.4f}, Time: {training_time:.1f}s")
            
            mean_ar = np.mean(ar_scores)
            std_ar = np.std(ar_scores)
            mean_time = np.mean(training_times)
            
            results.append({
                'feature_set': name,
                'n_features': len(features),
                'mean_ar': mean_ar,
                'std_ar': std_ar,
                'mean_time_sec': mean_time,
                'ar_scores': ar_scores
            })
            
            logger.info(f"  → Mean AR: {mean_ar:.4f} ± {std_ar:.4f}")
            logger.info(f"  → Mean Time: {mean_time:.1f}s")
        
        df_results = pd.DataFrame(results).sort_values('mean_ar', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON RESULTS")
        logger.info(f"{'='*60}")
        logger.info(df_results[['feature_set', 'n_features', 'mean_ar', 'std_ar', 'mean_time_sec']].to_string(index=False))
        
        # Best feature set
        best = df_results.iloc[0]
        baseline = df_results[df_results['feature_set'] == 'baseline']
        
        if len(baseline) > 0:
            baseline_ar = baseline.iloc[0]['mean_ar']
            improvement = best['mean_ar'] - baseline_ar
            speedup = baseline.iloc[0]['mean_time_sec'] / best['mean_time_sec']
            
            logger.info(f"\nBest Feature Set: '{best['feature_set']}'")
            logger.info(f"  Features: {best['n_features']} (from {len(self.feature_names)})")
            logger.info(f"  AR: {best['mean_ar']:.4f} ± {best['std_ar']:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f} ({improvement/baseline_ar*100:+.1f}%)")
            logger.info(f"  Speedup: {speedup:.1f}x faster")
        
        return df_results
    
    def _train_quick(self, X_train, y_train, X_val, y_val, feature_names):
        """Quick model training for comparison."""
        processor = DataProcessor()
        X_train_norm = X_train / processor.scale_factor
        X_val_norm = X_val / processor.scale_factor
        
        config = self._create_config(epochs=50)
        model = create_model(input_dim=X_train_norm.shape[1], config=config.neural_network)
        
        trainer = ModelTrainer(scale_factor=processor.scale_factor)
        train_loader, val_loader = trainer.create_data_loaders(
            X_train_norm, y_train, X_val_norm, y_val,
            batch_size=256, random_seed=42
        )
        
        result = trainer.train(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            config=config,
            feature_names=feature_names
        )
        
        return {
            'ar': result.test_metrics.discrimination.gini_ar,
            'auc': result.test_metrics.discrimination.auc_roc
        }
    
    def _create_config(self, epochs=50):
        """Create training config."""
        return TrainingConfig(
            test_size=0.2,
            random_seed=42,
            epochs=epochs,
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
                loss_alpha=0.2,
                auc_loss_type='pairwise',
                margin=0.3
            ),
            regularization=RegularizationConfig(
                l2_lambda=0.001,
                dropout_rate=0.2
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=10
            )
        )
    
    def run_full_analysis(self, target_features: int = 15):
        """Run complete feature selection analysis."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"FULL FEATURE SELECTION ANALYSIS")
        logger.info(f"{'#'*60}")
        logger.info(f"Target: {target_features} features (from {len(self.feature_names)})")
        
        # 1. Calculate all metrics
        df_iv = self.calculate_information_value()
        df_importance = self.calculate_model_importance()
        df_corr = self.calculate_correlation_matrix(threshold=0.85)
        
        # 2. Create feature sets
        feature_sets = {
            'baseline': self.feature_names,
            f'top_{target_features}_importance': df_importance.head(target_features)['feature'].tolist(),
            f'top_{target_features}_iv': df_iv.head(target_features)['feature'].tolist(),
            'iv_medium_plus': df_iv[df_iv['rating'].isin(['Medium', 'Strong', 'Very Strong'])]['feature'].tolist(),
        }
        
        # Combined method
        df_combined = df_iv.merge(df_importance[['feature', 'importance_pct']], on='feature')
        df_combined['iv_norm'] = df_combined['iv'] / df_combined['iv'].max()
        df_combined['imp_norm'] = df_combined['importance_pct'] / 100
        df_combined['combined_score'] = (df_combined['iv_norm'] + df_combined['imp_norm']) / 2
        df_combined = df_combined.sort_values('combined_score', ascending=False)
        
        feature_sets[f'top_{target_features}_combined'] = df_combined.head(target_features)['feature'].tolist()
        
        # 3. Compare
        results = self.compare_feature_sets(feature_sets, cv_folds=3)
        
        # 4. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all analysis
        output = {
            'timestamp': timestamp,
            'original_features': len(self.feature_names),
            'target_features': target_features,
            'feature_sets': feature_sets,
            'results': results[['feature_set', 'n_features', 'mean_ar', 'std_ar', 'mean_time_sec']].to_dict('records'),
            'iv_analysis': df_iv.to_dict('records'),
            'importance_analysis': df_importance.to_dict('records'),
            'correlated_pairs': df_corr.to_dict('records') if len(df_corr) > 0 else []
        }
        
        output_file = self.output_dir / f'feature_selection_analysis_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"\n✓ Analysis saved to: {output_file}")
        
        # Save best feature set
        best_set_name = results.iloc[0]['feature_set']
        best_features = feature_sets[best_set_name]
        
        best_features_file = self.output_dir / f'best_features_{timestamp}.txt'
        with open(best_features_file, 'w') as f:
            f.write('\n'.join(best_features))
        
        logger.info(f"✓ Best features saved to: {best_features_file}")
        
        return results, best_features


def main():
    parser = argparse.ArgumentParser(description='Feature selection for scorecard')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--segment', type=str, default=None, help='Segment to filter')
    parser.add_argument('--target', type=str, default='default', help='Target column name')
    parser.add_argument('--target-features', type=int, default=15, help='Target number of features')
    parser.add_argument('--method', type=str, default='combined', 
                        choices=['importance', 'iv', 'combined', 'all'],
                        help='Selection method')
    parser.add_argument('--output', type=str, default='data/feature_selection', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = FeatureSelector(
        data_path=args.data,
        target_col=args.target,
        segment=args.segment,
        output_dir=args.output
    )
    
    if args.method == 'all':
        # Full analysis
        results, best_features = selector.run_full_analysis(target_features=args.target_features)
    else:
        # Single method
        selected_features = selector.select_features(
            method=args.method,
            target_n=args.target_features
        )
        
        # Compare with baseline
        results = selector.compare_feature_sets({
            'baseline': selector.feature_names,
            'selected': selected_features
        })
    
    logger.info("\n✓ Feature selection completed successfully!")


if __name__ == '__main__':
    main()

