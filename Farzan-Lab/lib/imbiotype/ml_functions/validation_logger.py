"""
Validation logger for comparing Python CCA implementation with MATLAB.

This module provides structured logging for hyperparameter optimization
to enable detailed comparison with MATLAB reference implementations.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime


class ValidationLogger:
    """
    Logger for CCA hyperparameter optimization validation.
    
    Provides multi-tier logging:
    - Tier 1 (Always): Summary statistics and final results
    - Tier 2 (Conditional): Detailed logs for first N repeats
    - Tier 3 (Debug): Full trace logging
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 enabled: bool = True,
                 log_repeats: int = 1,
                 sample_size: int = 10,
                 debug_mode: bool = False):
        """
        Initialize validation logger.
        
        Args:
            log_dir: Directory for log files (None = disabled)
            enabled: Whether logging is enabled
            log_repeats: Number of repeats to log in detail (Tier 2)
            sample_size: Number of data rows to save in CSVs
            debug_mode: Enable Tier 3 debug logging
        """
        self.enabled = enabled and log_dir is not None
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_repeats = log_repeats
        self.sample_size = sample_size
        self.debug_mode = debug_mode
        
        # Storage for summary statistics (Tier 1)
        self.lambda_scores = []  # All lambda combinations and scores
        self.fold_results = []   # Results per fold
        self.repeat_results = [] # Results per repeat
        
        if self.enabled:
            self._setup_directories()
    
    def _setup_directories(self):
        """Create directory structure for logs."""
        if not self.log_dir:
            return
            
        # Create main directories
        self.summary_dir = self.log_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug_mode:
            self.debug_dir = self.log_dir / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_repeat_dir(self, repeat_idx: int) -> Optional[Path]:
        """Get directory for a specific repeat (Tier 2 logging)."""
        if not self.enabled or repeat_idx >= self.log_repeats:
            return None
        
        repeat_dir = self.log_dir / f"repeat_{repeat_idx+1:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        return repeat_dir
    
    def _get_fold_dir(self, repeat_idx: int, fold_idx: int) -> Optional[Path]:
        """Get directory for a specific fold (Tier 2 logging)."""
        repeat_dir = self._get_repeat_dir(repeat_idx)
        if repeat_dir is None:
            return None
        
        fold_dir = repeat_dir / f"fold_{fold_idx+1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir
    
    def log_cv_split(self, repeat_idx: int, fold_idx: int, 
                     train_indices: np.ndarray, val_indices: np.ndarray,
                     stratify_values: Optional[np.ndarray] = None):
        """
        Log cross-validation split indices and stratification.
        
        Args:
            repeat_idx: Repeat index (0-based)
            fold_idx: Fold index (0-based)
            train_indices: Training set indices
            val_indices: Validation set indices
            stratify_values: Stratification variable values (e.g., Site)
        """
        fold_dir = self._get_fold_dir(repeat_idx, fold_idx)
        if fold_dir is None:
            return
        
        # Save indices
        pd.DataFrame({'train_idx': train_indices}).to_csv(
            fold_dir / "train_indices.csv", index=False)
        pd.DataFrame({'val_idx': val_indices}).to_csv(
            fold_dir / "val_indices.csv", index=False)
        
        # Save stratification info if provided
        if stratify_values is not None:
            train_strata = stratify_values[train_indices]
            val_strata = stratify_values[val_indices]
            
            strat_info = {
                'train_distribution': pd.Series(train_strata).value_counts().to_dict(),
                'val_distribution': pd.Series(val_strata).value_counts().to_dict()
            }
            
            with open(fold_dir / "stratification.json", 'w') as f:
                json.dump(strat_info, f, indent=2)
    
    def log_preprocessing(self, repeat_idx: int, fold_idx: int,
                         stage: str,
                         train_X: pd.DataFrame, train_Y: pd.DataFrame,
                         val_X: Optional[pd.DataFrame] = None,
                         val_Y: Optional[pd.DataFrame] = None):
        """
        Log preprocessed data at various stages.
        
        Args:
            repeat_idx: Repeat index
            fold_idx: Fold index
            stage: Preprocessing stage name (e.g., 'residualized', 'normalized')
            train_X, train_Y: Training data
            val_X, val_Y: Validation data (optional)
        """
        fold_dir = self._get_fold_dir(repeat_idx, fold_idx)
        if fold_dir is None:
            return
        
        # Save sample of data (first N rows)
        n = min(self.sample_size, len(train_X))
        
        train_X.iloc[:n].to_csv(fold_dir / f"{stage}_train_X_sample.csv", index=False)
        train_Y.iloc[:n].to_csv(fold_dir / f"{stage}_train_Y_sample.csv", index=False)
        
        if val_X is not None and val_Y is not None:
            n_val = min(self.sample_size, len(val_X))
            val_X.iloc[:n_val].to_csv(fold_dir / f"{stage}_val_X_sample.csv", index=False)
            val_Y.iloc[:n_val].to_csv(fold_dir / f"{stage}_val_Y_sample.csv", index=False)
        
        # Save summary statistics
        stats = {
            'stage': stage,
            'train_X_shape': train_X.shape,
            'train_Y_shape': train_Y.shape,
            'train_X_mean': float(train_X.mean().mean()),
            'train_X_std': float(train_X.std().mean()),
            'train_Y_mean': float(train_Y.mean().mean()),
            'train_Y_std': float(train_Y.std().mean()),
        }
        
        if val_X is not None:
            stats.update({
                'val_X_shape': val_X.shape,
                'val_Y_shape': val_Y.shape,
                'val_X_mean': float(val_X.mean().mean()),
                'val_X_std': float(val_X.std().mean()),
                'val_Y_mean': float(val_Y.mean().mean()),
                'val_Y_std': float(val_Y.std().mean()),
            })
        
        with open(fold_dir / f"{stage}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def log_lambda_score(self, repeat_idx: int, fold_idx: int,
                        lambda_x: float, lambda_y: float,
                        score: float, combination_idx: int, total_combinations: int):
        """
        Log score for a specific lambda combination.
        
        Args:
            repeat_idx: Repeat index
            fold_idx: Fold index
            lambda_x: Lambda value for X
            lambda_y: Lambda value for Y
            score: Correlation score
            combination_idx: Index of this combination (0-based)
            total_combinations: Total number of combinations
        """
        # Tier 1: Always store for summary
        self.lambda_scores.append({
            'repeat': repeat_idx,
            'fold': fold_idx,
            'lambda_x': lambda_x,
            'lambda_y': lambda_y,
            'score': score,
            'combination_idx': combination_idx
        })
        
        # Tier 2: Detailed logging for first N repeats
        fold_dir = self._get_fold_dir(repeat_idx, fold_idx)
        if fold_dir is not None:
            # Append to fold-specific lambda scores file
            scores_file = fold_dir / "lambda_scores.csv"
            
            df = pd.DataFrame([{
                'combination_idx': combination_idx,
                'lambda_x': lambda_x,
                'lambda_y': lambda_y,
                'score': score
            }])
            
            # Append or create
            if scores_file.exists():
                df.to_csv(scores_file, mode='a', header=False, index=False)
            else:
                df.to_csv(scores_file, index=False)
    
    def log_fold_best_lambda(self, repeat_idx: int, fold_idx: int,
                            best_lambda_x: float, best_lambda_y: float,
                            best_score: float):
        """
        Log the best lambda selected for a fold.
        
        Args:
            repeat_idx: Repeat index
            fold_idx: Fold index
            best_lambda_x: Best lambda for X
            best_lambda_y: Best lambda for Y
            best_score: Best correlation score
        """
        # Tier 1: Store for summary
        self.fold_results.append({
            'repeat': repeat_idx,
            'fold': fold_idx,
            'best_lambda_x': best_lambda_x,
            'best_lambda_y': best_lambda_y,
            'best_score': best_score
        })
        
        # Tier 2: Save to fold directory
        fold_dir = self._get_fold_dir(repeat_idx, fold_idx)
        if fold_dir is not None:
            result = {
                'best_lambda_x': best_lambda_x,
                'best_lambda_y': best_lambda_y,
                'best_score': best_score,
                'fold': fold_idx,
                'repeat': repeat_idx
            }
            
            with open(fold_dir / "best_lambda.json", 'w') as f:
                json.dump(result, f, indent=2)
    
    def log_repeat_summary(self, repeat_idx: int,
                          optimal_lambda_x: float, optimal_lambda_y: float,
                          median_score: float):
        """
        Log summary for a complete repeat (all folds).
        
        Args:
            repeat_idx: Repeat index
            optimal_lambda_x: Optimal lambda for X (across all folds)
            optimal_lambda_y: Optimal lambda for Y (across all folds)
            median_score: Median score across folds
        """
        # Tier 1: Store for summary
        self.repeat_results.append({
            'repeat': repeat_idx,
            'optimal_lambda_x': optimal_lambda_x,
            'optimal_lambda_y': optimal_lambda_y,
            'median_score': median_score
        })
        
        # Tier 2: Save to repeat directory
        repeat_dir = self._get_repeat_dir(repeat_idx)
        if repeat_dir is not None:
            summary = {
                'repeat': repeat_idx,
                'optimal_lambda_x': optimal_lambda_x,
                'optimal_lambda_y': optimal_lambda_y,
                'median_score': median_score,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(repeat_dir / "repeat_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

    def save_final_summary(self, final_lambda_x: float, final_lambda_y: float,
                          final_score: float, selection_method: str = "max_median"):
        """
        Save final summary of hyperparameter optimization.

        Args:
            final_lambda_x: Final selected lambda for X
            final_lambda_y: Final selected lambda for Y
            final_score: Final score
            selection_method: How lambdas were selected
        """
        if not self.enabled:
            return

        # Save optimal lambdas
        optimal = {
            'final_lambda_x': final_lambda_x,
            'final_lambda_y': final_lambda_y,
            'final_score': final_score,
            'selection_method': selection_method,
            'timestamp': datetime.now().isoformat(),
            'total_repeats': len(set(r['repeat'] for r in self.repeat_results)),
            'total_folds': len(self.fold_results),
            'total_lambda_evaluations': len(self.lambda_scores)
        }

        with open(self.summary_dir / "optimal_lambdas.json", 'w') as f:
            json.dump(optimal, f, indent=2)

        # Save all lambda scores (Tier 1 - always saved)
        if self.lambda_scores:
            df_scores = pd.DataFrame(self.lambda_scores)
            df_scores.to_csv(self.summary_dir / "all_lambda_scores.csv", index=False)

            # Create summary statistics by lambda combination
            lambda_summary = df_scores.groupby(['lambda_x', 'lambda_y']).agg({
                'score': ['mean', 'median', 'std', 'min', 'max', 'count']
            }).reset_index()
            lambda_summary.columns = ['lambda_x', 'lambda_y', 'mean_score', 'median_score',
                                     'std_score', 'min_score', 'max_score', 'n_evaluations']
            lambda_summary = lambda_summary.sort_values('median_score', ascending=False)
            lambda_summary.to_csv(self.summary_dir / "lambda_grid_summary.csv", index=False)

        # Save fold results
        if self.fold_results:
            df_folds = pd.DataFrame(self.fold_results)
            df_folds.to_csv(self.summary_dir / "fold_results.csv", index=False)

        # Save repeat results
        if self.repeat_results:
            df_repeats = pd.DataFrame(self.repeat_results)
            df_repeats.to_csv(self.summary_dir / "repeat_results.csv", index=False)

            # Summary statistics across repeats
            repeat_stats = {
                'lambda_x_mean': float(df_repeats['optimal_lambda_x'].mean()),
                'lambda_x_std': float(df_repeats['optimal_lambda_x'].std()),
                'lambda_x_median': float(df_repeats['optimal_lambda_x'].median()),
                'lambda_y_mean': float(df_repeats['optimal_lambda_y'].mean()),
                'lambda_y_std': float(df_repeats['optimal_lambda_y'].std()),
                'lambda_y_median': float(df_repeats['optimal_lambda_y'].median()),
                'score_mean': float(df_repeats['median_score'].mean()),
                'score_std': float(df_repeats['median_score'].std()),
                'n_repeats': len(df_repeats)
            }

            with open(self.summary_dir / "repeat_statistics.json", 'w') as f:
                json.dump(repeat_stats, f, indent=2)

        print(f"\n{'='*80}")
        print(f"VALIDATION LOGS SAVED TO: {self.log_dir}")
        print(f"{'='*80}")
        print(f"Summary files:")
        print(f"  - optimal_lambdas.json: Final selected lambdas")
        print(f"  - lambda_grid_summary.csv: Performance of all 150 lambda combinations")
        print(f"  - fold_results.csv: Best lambda per fold")
        print(f"  - repeat_results.csv: Optimal lambda per repeat")
        if self.log_repeats > 0:
            print(f"\nDetailed logs for first {self.log_repeats} repeat(s):")
            print(f"  - repeat_XXX/fold_YY/: CV splits, preprocessed data, lambda scores")
        print(f"{'='*80}\n")

    def debug_log(self, message: str):
        """
        Write debug message to log file (Tier 3).

        Args:
            message: Debug message
        """
        if not self.debug_mode or not self.enabled:
            return

        log_file = self.debug_dir / "debug_trace.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

