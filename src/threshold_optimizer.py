"""
Adaptive Threshold Optimization

Implements Section 3.3 and Algorithm 1 from the paper:
- Dynamic threshold optimization using gradient ascent
- Fβ score optimization (β=2 for recall-prioritized)
- Central difference approximation for gradient estimation
- Early stopping with patience mechanism

Key equations:
- Fβ(τ) = (1 + β²) * (prec(τ) * rec(τ)) / (β² * prec(τ) + rec(τ))  [Eq. 4]
- ∇Fβ ≈ (Fβ(τ+ε) - Fβ(τ-ε)) / 2ε  [Eq. 5]
- τt+1 = τt + η * ∂Fβ/∂τ  [Eq. 6]
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
from loguru import logger
import matplotlib.pyplot as plt


@dataclass
class ThresholdOptimizationResult:
    """Container for optimization results"""
    optimal_threshold: float
    initial_threshold: float
    final_f_beta: float
    initial_f_beta: float
    num_iterations: int
    threshold_history: List[float]
    f_beta_history: List[float]
    precision_at_optimal: float
    recall_at_optimal: float


class AdaptiveThresholdOptimizer:
    """
    Implements Algorithm 1: Dynamic Threshold Optimization for Scam Detection
    
    Uses gradient ascent on Fβ score to find optimal classification threshold.
    Particularly suited for imbalanced datasets (1:100 in CryptoScams).
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        beta: float = 2.0,  # Recall weight (paper: β=2)
        learning_rate: float = 0.01,  # η in paper
        epsilon: float = 0.01,  # ε for central difference
        patience: int = 5,  # P for early stopping
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        verbose: bool = True
    ):
        """
        Initialize the adaptive threshold optimizer.
        
        Args:
            initial_threshold: τ₀ = 0.5 (paper default)
            beta: Recall weight for Fβ (paper: 2.0)
            learning_rate: η = 0.01 (paper default)
            epsilon: ε = 0.01 for gradient approximation
            patience: P = 5 for early stopping
            min_threshold: Lower bound for threshold
            max_threshold: Upper bound for threshold
            verbose: Whether to log progress
        """
        self.initial_threshold = initial_threshold
        self.beta = beta
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.patience = patience
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.verbose = verbose
        
        # State
        self.current_threshold = initial_threshold
        self.optimal_threshold = initial_threshold
        self.threshold_history = []
        self.f_beta_history = []
        
        logger.info(
            f"AdaptiveThresholdOptimizer initialized: "
            f"τ₀={initial_threshold}, β={beta}, η={learning_rate}, P={patience}"
        )
    
    def compute_f_beta(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float
    ) -> Tuple[float, float, float]:
        """
        Compute Fβ score at given threshold.
        Implements Equation (4):
        Fβ(τ) = (1 + β²) * (prec(τ) * rec(τ)) / (β² * prec(τ) + rec(τ))
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold τ
            
        Returns:
            Tuple of (f_beta, precision, recall)
        """
        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Handle edge cases
        if np.sum(y_pred) == 0:
            return 0.0, 0.0, 0.0
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Compute Fβ
        if precision + recall == 0:
            f_beta = 0.0
        else:
            beta_sq = self.beta ** 2
            f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        
        return f_beta, precision, recall
    
    def compute_gradient(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float
    ) -> float:
        """
        Compute gradient using central difference approximation.
        Implements Equation (5):
        ∇Fβ ≈ (Fβ(τ+ε) - Fβ(τ-ε)) / 2ε
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities  
            threshold: Current threshold τ
            
        Returns:
            Gradient estimate
        """
        # Ensure we don't go out of bounds
        tau_plus = min(threshold + self.epsilon, self.max_threshold)
        tau_minus = max(threshold - self.epsilon, self.min_threshold)
        
        # Compute Fβ at both points
        f_beta_plus, _, _ = self.compute_f_beta(y_true, y_pred_proba, tau_plus)
        f_beta_minus, _, _ = self.compute_f_beta(y_true, y_pred_proba, tau_minus)
        
        # Central difference
        gradient = (f_beta_plus - f_beta_minus) / (2 * self.epsilon)
        
        return gradient
    
    def optimize(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        max_iterations: int = 100
    ) -> ThresholdOptimizationResult:
        """
        Run threshold optimization algorithm.
        Implements Algorithm 1 from the paper.
        
        Args:
            y_true: Ground truth labels (validation set V)
            y_pred_proba: Predicted probabilities
            max_iterations: Maximum iterations
            
        Returns:
            ThresholdOptimizationResult with optimal threshold
        """
        logger.info("Starting threshold optimization...")
        
        # Initialize (Line 1 of Algorithm 1)
        t = 0  # Iteration counter
        p = 0  # Patience counter (no improvement)
        tau_t = self.initial_threshold
        tau_star = tau_t  # Best threshold
        
        # Reset history
        self.threshold_history = [tau_t]
        self.f_beta_history = []
        
        # Compute initial Fβ
        f_beta_init, prec_init, rec_init = self.compute_f_beta(y_true, y_pred_proba, tau_t)
        f_beta_star = f_beta_init
        self.f_beta_history.append(f_beta_init)
        
        if self.verbose:
            logger.info(f"Initial: τ={tau_t:.4f}, Fβ={f_beta_init:.4f}, P={prec_init:.4f}, R={rec_init:.4f}")
        
        # Main loop (Line 2-12 of Algorithm 1)
        while p < self.patience and t < max_iterations:
            # Compute gradient (Line 4-5)
            gradient = self.compute_gradient(y_true, y_pred_proba, tau_t)
            
            # Update threshold (Line 5, Equation 6)
            tau_new = tau_t + self.learning_rate * gradient
            
            # Clip to valid range
            tau_new = np.clip(tau_new, self.min_threshold, self.max_threshold)
            
            # Compute new Fβ (Line 3)
            f_beta_new, prec_new, rec_new = self.compute_f_beta(y_true, y_pred_proba, tau_new)
            
            # Check improvement (Line 6-10)
            if f_beta_new <= self.f_beta_history[-1]:
                p += 1  # No improvement (Line 7)
            else:
                tau_star = tau_new  # Update best (Line 9)
                f_beta_star = f_beta_new
                p = 0  # Reset patience (Line 9)
            
            # Record history
            tau_t = tau_new
            self.threshold_history.append(tau_t)
            self.f_beta_history.append(f_beta_new)
            
            t += 1  # (Line 11)
            
            if self.verbose and t % 5 == 0:
                logger.info(
                    f"Iter {t}: τ={tau_t:.4f}, Fβ={f_beta_new:.4f}, "
                    f"P={prec_new:.4f}, R={rec_new:.4f}, patience={p}"
                )
        
        # Final results
        f_beta_final, prec_final, rec_final = self.compute_f_beta(y_true, y_pred_proba, tau_star)
        
        self.optimal_threshold = tau_star
        self.current_threshold = tau_star
        
        result = ThresholdOptimizationResult(
            optimal_threshold=tau_star,
            initial_threshold=self.initial_threshold,
            final_f_beta=f_beta_final,
            initial_f_beta=f_beta_init,
            num_iterations=t,
            threshold_history=self.threshold_history,
            f_beta_history=self.f_beta_history,
            precision_at_optimal=prec_final,
            recall_at_optimal=rec_final
        )
        
        logger.info(
            f"Optimization complete: τ*={tau_star:.4f}, Fβ={f_beta_final:.4f}, "
            f"iterations={t}, improvement={(f_beta_final - f_beta_init) / f_beta_init * 100:.1f}%"
        )
        
        return result
    
    def plot_optimization(
        self,
        result: ThresholdOptimizationResult,
        save_path: str = None
    ) -> None:
        """
        Plot optimization trajectory and Fβ landscape.
        Reproduces Figure 2 from the paper.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Threshold trajectory
        ax1 = axes[0]
        iterations = range(len(result.threshold_history))
        ax1.plot(iterations, result.threshold_history, 'b-o', markersize=4, label='Threshold τ')
        ax1.axhline(y=result.optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal τ*={result.optimal_threshold:.4f}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Threshold τ')
        ax1.set_title('Threshold Optimization Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fβ score trajectory
        ax2 = axes[1]
        ax2.plot(iterations, result.f_beta_history, 'g-o', markersize=4, label=f'Fβ (β={self.beta})')
        ax2.axhline(y=result.final_f_beta, color='r', linestyle='--',
                   label=f'Final Fβ={result.final_f_beta:.4f}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fβ Score')
        ax2.set_title('Fβ Score During Optimization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()


class GridSearchBaseline:
    """
    Grid search baseline for comparison.
    Paper mentions this is 8× slower than gradient-based method.
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        grid_size: int = 100
    ):
        self.beta = beta
        self.grid_size = grid_size
    
    def search(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ) -> Tuple[float, float, float, float]:
        """
        Exhaustive grid search for optimal threshold.
        
        Returns:
            Tuple of (optimal_threshold, f_beta, precision, recall)
        """
        thresholds = np.linspace(min_threshold, max_threshold, self.grid_size)
        
        best_threshold = 0.5
        best_f_beta = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        for tau in thresholds:
            y_pred = (y_pred_proba >= tau).astype(int)
            
            if np.sum(y_pred) == 0:
                continue
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if precision + recall == 0:
                continue
            
            beta_sq = self.beta ** 2
            f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
            
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_threshold = tau
                best_precision = precision
                best_recall = recall
        
        return best_threshold, best_f_beta, best_precision, best_recall


class ThresholdComparison:
    """
    Compare different threshold strategies.
    Reproduces Table 4 from the paper.
    """
    
    def __init__(self, beta: float = 2.0):
        self.beta = beta
        self.adaptive_optimizer = AdaptiveThresholdOptimizer(beta=beta, verbose=False)
        self.grid_search = GridSearchBaseline(beta=beta)
    
    def compare(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        fixed_thresholds: List[float] = [0.5, 0.7]
    ) -> Dict[str, Dict]:
        """
        Compare different threshold strategies.
        
        Returns:
            Dictionary with results for each method
        """
        results = {}
        
        # Fixed thresholds
        for tau in fixed_thresholds:
            f_beta, prec, rec = self.adaptive_optimizer.compute_f_beta(
                y_true, y_pred_proba, tau
            )
            results[f"Fixed τ={tau}"] = {
                "threshold": tau,
                "precision": prec,
                "recall": rec,
                "f_beta": f_beta
            }
        
        # Grid search
        import time
        start = time.time()
        tau_grid, f_beta_grid, prec_grid, rec_grid = self.grid_search.search(
            y_true, y_pred_proba
        )
        grid_time = time.time() - start
        results["Grid Search"] = {
            "threshold": tau_grid,
            "precision": prec_grid,
            "recall": rec_grid,
            "f_beta": f_beta_grid,
            "time_ms": grid_time * 1000
        }
        
        # Adaptive optimization
        start = time.time()
        opt_result = self.adaptive_optimizer.optimize(y_true, y_pred_proba)
        adaptive_time = time.time() - start
        results["Ours (Adaptive)"] = {
            "threshold": opt_result.optimal_threshold,
            "precision": opt_result.precision_at_optimal,
            "recall": opt_result.recall_at_optimal,
            "f_beta": opt_result.final_f_beta,
            "time_ms": adaptive_time * 1000,
            "iterations": opt_result.num_iterations
        }
        
        return results
    
    def print_comparison_table(self, results: Dict) -> None:
        """Print comparison table in paper format"""
        print("\n" + "=" * 70)
        print("Threshold Optimization Comparison (Table 4 format)")
        print("=" * 70)
        print(f"{'Method':<20} {'Recall':>10} {'Precision':>10} {'Fβ=2':>10} {'Time(ms)':>10}")
        print("-" * 70)
        
        for method, data in results.items():
            time_str = f"{data.get('time_ms', '-'):.1f}" if 'time_ms' in data else "-"
            print(
                f"{method:<20} {data['recall']:>10.4f} {data['precision']:>10.4f} "
                f"{data['f_beta']:>10.4f} {time_str:>10}"
            )
        
        print("=" * 70)


def compute_performance_retention(
    train_f_beta: float,
    test_f_beta: float
) -> float:
    """
    Compute performance retention metric from Equation (10):
    Retention = (Fβ_test / Fβ_train) × 100%
    """
    if train_f_beta == 0:
        return 0.0
    return (test_f_beta / train_f_beta) * 100


if __name__ == "__main__":
    # Demo usage
    print("Testing Adaptive Threshold Optimization\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Imbalanced dataset (similar to CryptoScams 1:100)
    n_positive = int(n_samples * 0.1)  # 10% scams
    n_negative = n_samples - n_positive
    
    y_true = np.concatenate([
        np.ones(n_positive),
        np.zeros(n_negative)
    ])
    
    # Generate predictions with some noise
    y_pred_proba = np.concatenate([
        np.random.beta(4, 2, n_positive),  # Higher scores for scams
        np.random.beta(2, 4, n_negative)   # Lower scores for legitimate
    ])
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    y_true = y_true[idx]
    y_pred_proba = y_pred_proba[idx]
    
    print(f"Dataset: {n_samples} samples, {n_positive} positive ({n_positive/n_samples:.1%})")
    print(f"Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    
    # Run optimization
    optimizer = AdaptiveThresholdOptimizer(
        initial_threshold=0.5,
        beta=2.0,
        learning_rate=0.01,
        patience=5
    )
    
    result = optimizer.optimize(y_true, y_pred_proba)
    
    print(f"\n--- Optimization Results ---")
    print(f"Initial threshold: {result.initial_threshold:.4f}")
    print(f"Optimal threshold: {result.optimal_threshold:.4f}")
    print(f"Initial Fβ: {result.initial_f_beta:.4f}")
    print(f"Final Fβ: {result.final_f_beta:.4f}")
    print(f"Iterations: {result.num_iterations}")
    print(f"Precision: {result.precision_at_optimal:.4f}")
    print(f"Recall: {result.recall_at_optimal:.4f}")
    
    # Compare methods
    print("\n--- Method Comparison ---")
    comparator = ThresholdComparison(beta=2.0)
    comparison = comparator.compare(y_true, y_pred_proba)
    comparator.print_comparison_table(comparison)
    
    # Save plot
    optimizer.plot_optimization(result, "threshold_optimization.png")
    print("\nPlot saved to threshold_optimization.png")
