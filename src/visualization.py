"""
Visualization utilities for plotting results.
"""
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_layer_response(
    layer_scores: Dict[int, float],
    save_path: str = None,
    show: bool = True
):
    """Visualize how steering effectiveness varies by layer.
    
    Args:
        layer_scores: Dictionary mapping layer numbers to steering effects
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in layers]
    best_layer = layers[np.argmax(scores)]
    
    plt.figure(figsize=(12, 7))
    plt.plot(layers, scores, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Layer', fontsize=14, fontweight='bold')
    plt.ylabel('Steering Effect (Validation)', fontsize=14, fontweight='bold')
    plt.title('Layer Response Curve - Steering Effectiveness by Layer', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axvline(x=best_layer, color='red', linestyle='--', linewidth=2,
                label=f'Best Layer: {best_layer} (Effect: {layer_scores[best_layer]:.3f})')
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiplier_effects(
    multiplier_scores: Dict[float, float],
    baseline_score: float = None,
    save_path: str = None,
    show: bool = True
):
    """Visualize steering effects at different multiplier values.
    
    Args:
        multiplier_scores: Dictionary mapping multipliers to scores
        baseline_score: Baseline score (multiplier=0) for reference
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    multipliers = sorted(multiplier_scores.keys())
    scores = [multiplier_scores[m] for m in multipliers]
    
    plt.figure(figsize=(12, 7))
    plt.plot(multipliers, scores, marker='o', linewidth=2, markersize=8, color='#A23B72')
    
    if baseline_score is not None:
        plt.axhline(y=baseline_score, color='gray', linestyle='--', linewidth=2,
                   label=f'Baseline (no steering): {baseline_score:.3f}')
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.xlabel('Steering Multiplier', fontsize=14, fontweight='bold')
    plt.ylabel('Non-Sycophantic Preference Score', fontsize=14, fontweight='bold')
    plt.title('Steering Effect vs Multiplier Strength', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_bars(
    results: Dict[str, float],
    title: str = "Model Performance Comparison",
    ylabel: str = "Score",
    save_path: str = None,
    show: bool = True
):
    """Create a bar plot comparing different conditions.
    
    Args:
        results: Dictionary mapping condition names to scores
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    conditions = list(results.keys())
    scores = list(results.values())
    colors = ['#E63946', '#F1A208', '#06A77D', '#2E86AB', '#A23B72']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, scores, color=colors[:len(conditions)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results_summary(
    results: Dict,
    filepath: str
):
    """Save results summary to a text file.
    
    Args:
        results: Dictionary containing results
        filepath: Path to save the summary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SYCOPHANCY STEERING RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Results summary saved to {filepath}")