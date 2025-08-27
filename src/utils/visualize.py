"""
Visualization utilities for critical weight analysis.

Provides plotting functions for sensitivity distributions, perturbation effects,
and stability analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_sensitivity_distribution(
    sensitivity_dict: Dict[str, torch.Tensor],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Weight Sensitivity Distribution",
    log_scale: bool = True,
    bins: int = 50
) -> plt.Figure:
    """
    Plot histogram of sensitivity values across all weights.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        output_path: Path to save plot (if None, just return figure)
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure object
        
    Examples:
        >>> fig = plot_sensitivity_distribution(sensitivity, "sensitivity_hist.png")
        >>> plt.show()
    """
    # Collect all sensitivity values
    all_sensitivities = []
    for name, sens_tensor in sensitivity_dict.items():
        all_sensitivities.extend(sens_tensor.flatten().cpu().numpy())
    
    all_sensitivities = np.array(all_sensitivities)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    counts, bins, patches = ax.hist(all_sensitivities, bins=bins, alpha=0.7, edgecolor='black')
    
    # Set scale
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Sensitivity Score')
    ax.set_ylabel('Count (log scale)' if log_scale else 'Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Statistics:
Total weights: {len(all_sensitivities):,}
Mean: {np.mean(all_sensitivities):.6f}
Std: {np.std(all_sensitivities):.6f}
Min: {np.min(all_sensitivities):.6f}
Max: {np.max(all_sensitivities):.6f}
95th percentile: {np.percentile(all_sensitivities, 95):.6f}
99th percentile: {np.percentile(all_sensitivities, 99):.6f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sensitivity distribution plot to {output_path}")
    
    return fig


def plot_layer_sensitivity_comparison(
    sensitivity_dict: Dict[str, torch.Tensor],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Sensitivity by Layer",
    metric_name: str = "Sensitivity"
) -> plt.Figure:
    """
    Plot comparison of sensitivity statistics across layers.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        output_path: Path to save plot
        title: Plot title
        metric_name: Name of the sensitivity metric for labeling
        
    Returns:
        Matplotlib figure object
    """
    # Compute statistics for each layer
    layer_stats = []
    
    for name, sens_tensor in sensitivity_dict.items():
        flat_sens = sens_tensor.flatten().cpu().numpy()
        
        layer_stats.append({
            'layer': name,
            'mean': np.mean(flat_sens),
            'std': np.std(flat_sens),
            'max': np.max(flat_sens),
            'p95': np.percentile(flat_sens, 95),
            'p99': np.percentile(flat_sens, 99),
            'num_weights': len(flat_sens)
        })
    
    df = pd.DataFrame(layer_stats)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Mean sensitivity by layer
    ax1 = axes[0, 0]
    ax1.bar(range(len(df)), df['mean'])
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel(f'Mean {metric_name}')
    ax1.set_title('Mean Sensitivity by Layer')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Max sensitivity by layer
    ax2 = axes[0, 1]
    ax2.bar(range(len(df)), df['max'])
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel(f'Max {metric_name}')
    ax2.set_title('Maximum Sensitivity by Layer')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: 95th percentile by layer
    ax3 = axes[1, 0]
    ax3.bar(range(len(df)), df['p95'])
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel(f'95th Percentile {metric_name}')
    ax3.set_title('95th Percentile Sensitivity by Layer')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Number of weights by layer
    ax4 = axes[1, 1]
    ax4.bar(range(len(df)), df['num_weights'])
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Number of Weights')
    ax4.set_title('Weight Count by Layer')
    ax4.tick_params(axis='x', rotation=45)
    
    # Set x-tick labels to layer names (truncated if too long)
    for ax in axes.flat:
        layer_labels = [name[:20] + "..." if len(name) > 20 else name for name in df['layer']]
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved layer comparison plot to {output_path}")
    
    return fig


def plot_perturbation_effects(
    results: Dict[str, Dict[str, float]],
    k_values: Optional[List[int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Perturbation Effects on Model Performance"
) -> plt.Figure:
    """
    Plot effects of different perturbation methods on model performance.
    
    Args:
        results: Dictionary mapping method names to performance metrics
        k_values: List of K values if results are for different K values
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure object
        
    Examples:
        >>> effects = {
        ...     "baseline": {"perplexity": 10.0},
        ...     "zero": {"perplexity": 12.5, "delta_perplexity": 2.5},
        ...     "sign_flip": {"perplexity": 15.0, "delta_perplexity": 5.0}
        ... }
        >>> fig = plot_perturbation_effects(effects, output_path="perturbation_effects.png")
    """
    # Extract method names and perplexity values
    methods = []
    perplexities = []
    delta_perplexities = []
    
    baseline_ppl = results.get("baseline", {}).get("perplexity", 0)
    
    for method, metrics in results.items():
        if method == "baseline":
            continue
        
        methods.append(method)
        perplexities.append(metrics.get("perplexity", 0))
        delta_perplexities.append(metrics.get("delta_perplexity", 
                                              metrics.get("perplexity", 0) - baseline_ppl))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Absolute perplexity values
    colors = sns.color_palette("husl", len(methods) + 1)
    
    # Plot baseline
    ax1.axhline(y=baseline_ppl, color=colors[0], linestyle='--', 
                linewidth=2, label='Baseline', alpha=0.8)
    
    # Plot perturbation methods
    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, perplexities, color=colors[1:], alpha=0.7)
    
    ax1.set_xlabel('Perturbation Method')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Absolute Perplexity Values')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, perplexities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + baseline_ppl * 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Delta perplexity (change from baseline)
    bars2 = ax2.bar(x_pos, delta_perplexities, color=colors[1:], alpha=0.7)
    
    ax2.set_xlabel('Perturbation Method')
    ax2.set_ylabel('Δ Perplexity')
    ax2.set_title('Change in Perplexity from Baseline')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, delta_perplexities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (max(delta_perplexities) * 0.02 if height >= 0 else max(delta_perplexities) * -0.02),
                f'{value:+.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved perturbation effects plot to {output_path}")
    
    return fig


def plot_k_vs_performance(
    k_values: List[int],
    performance_data: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Performance vs Number of Perturbed Weights (K)",
    metric_name: str = "Perplexity"
) -> plt.Figure:
    """
    Plot model performance vs number of perturbed weights (K).
    
    Args:
        k_values: List of K values (number of weights perturbed)
        performance_data: Dictionary mapping method names to performance lists
        output_path: Path to save plot
        title: Plot title
        metric_name: Name of the performance metric
        
    Returns:
        Matplotlib figure object
        
    Examples:
        >>> k_vals = [10, 50, 100, 500]
        >>> perf_data = {
        ...     "zero": [10.2, 11.5, 13.0, 18.5],
        ...     "sign_flip": [10.5, 12.0, 14.2, 20.1],
        ...     "random": [10.1, 10.3, 10.5, 10.8]
        ... }
        >>> fig = plot_k_vs_performance(k_vals, perf_data, "k_vs_performance.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(performance_data))
    
    for i, (method, values) in enumerate(performance_data.items()):
        ax.plot(k_values, values, 'o-', color=colors[i], label=method, 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Number of Perturbed Weights (K)')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to log scale if K values span multiple orders of magnitude
    if max(k_values) / min(k_values) > 20:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved K vs performance plot to {output_path}")
    
    return fig


def plot_stability_analysis(
    stability_results: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Top-K Selection Stability Analysis"
) -> plt.Figure:
    """
    Plot stability analysis results showing Jaccard similarities.
    
    Args:
        stability_results: Dictionary with stability metrics
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure object
        
    Examples:
        >>> stability = {
        ...     "seed_jaccard_mean": 0.85,
        ...     "seed_jaccard_std": 0.05,
        ...     "batch_jaccard_mean": 0.72,
        ...     "batch_jaccard_std": 0.08
        ... }
        >>> fig = plot_stability_analysis(stability, "stability.png")
    """
    categories = ['Seed Stability', 'Batch Stability']
    means = [
        stability_results.get('seed_jaccard_mean', 0),
        stability_results.get('batch_jaccard_mean', 0)
    ]
    stds = [
        stability_results.get('seed_jaccard_std', 0),
        stability_results.get('batch_jaccard_std', 0)
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=sns.color_palette("husl", 2))
    
    ax.set_xlabel('Stability Type')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add horizontal line at perfect stability (1.0)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Stability')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stability analysis plot to {output_path}")
    
    return fig


def create_sensitivity_heatmap(
    sensitivity_dict: Dict[str, torch.Tensor],
    layer_names: Optional[List[str]] = None,
    max_layers: int = 20,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Sensitivity Heatmap by Layer"
) -> plt.Figure:
    """
    Create heatmap showing sensitivity patterns across layers.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        layer_names: Optional list of layer names to include
        max_layers: Maximum number of layers to include
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Select layers to include
    if layer_names is None:
        layer_names = list(sensitivity_dict.keys())
    
    layer_names = layer_names[:max_layers]
    
    # Compute percentiles for each layer
    percentiles = [50, 75, 90, 95, 99]
    heatmap_data = []
    
    for name in layer_names:
        if name in sensitivity_dict:
            sens_tensor = sensitivity_dict[name]
            flat_sens = sens_tensor.flatten().cpu().numpy()
            
            layer_percentiles = [np.percentile(flat_sens, p) for p in percentiles]
            heatmap_data.append(layer_percentiles)
        else:
            heatmap_data.append([0] * len(percentiles))
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, max(6, len(layer_names) * 0.5)))
    
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(percentiles)))
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([name[:30] + "..." if len(name) > 30 else name for name in layer_names])
    
    ax.set_xlabel('Sensitivity Percentile')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Sensitivity Score')
    
    # Add text annotations
    for i in range(len(layer_names)):
        for j in range(len(percentiles)):
            text = f'{heatmap_data[i, j]:.3f}'
            ax.text(j, i, text, ha="center", va="center", 
                   color="white" if heatmap_data[i, j] > np.max(heatmap_data) * 0.5 else "black",
                   fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sensitivity heatmap to {output_path}")
    
    return fig


def save_all_plots(
    sensitivity_dict: Dict[str, torch.Tensor],
    perturbation_results: Optional[Dict[str, Dict[str, float]]] = None,
    stability_results: Optional[Dict[str, float]] = None,
    output_dir: Union[str, Path] = "plots",
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Generate and save all standard plots for critical weight analysis.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        perturbation_results: Optional perturbation experiment results
        stability_results: Optional stability analysis results
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        
    Returns:
        Dictionary mapping plot names to saved file paths
        
    Examples:
        >>> saved_plots = save_all_plots(
        ...     sensitivity_dict,
        ...     perturbation_results,
        ...     stability_results,
        ...     "outputs/plots",
        ...     "llama_7b_"
        ... )
        >>> print(f"Saved {len(saved_plots)} plots")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Sensitivity distribution
    plot_path = output_dir / f"{prefix}sensitivity_distribution.png"
    plot_sensitivity_distribution(sensitivity_dict, plot_path)
    saved_plots["sensitivity_distribution"] = plot_path
    
    # Layer comparison
    plot_path = output_dir / f"{prefix}layer_comparison.png"
    plot_layer_sensitivity_comparison(sensitivity_dict, plot_path)
    saved_plots["layer_comparison"] = plot_path
    
    # Sensitivity heatmap
    plot_path = output_dir / f"{prefix}sensitivity_heatmap.png"
    create_sensitivity_heatmap(sensitivity_dict, output_path=plot_path)
    saved_plots["sensitivity_heatmap"] = plot_path
    
    # Perturbation effects (if available)
    if perturbation_results:
        plot_path = output_dir / f"{prefix}perturbation_effects.png"
        plot_perturbation_effects(perturbation_results, output_path=plot_path)
        saved_plots["perturbation_effects"] = plot_path
    
    # Stability analysis (if available)
    if stability_results:
        plot_path = output_dir / f"{prefix}stability_analysis.png"
        plot_stability_analysis(stability_results, output_path=plot_path)
        saved_plots["stability_analysis"] = plot_path
    
    logger.info(f"Saved {len(saved_plots)} plots to {output_dir}")
    return saved_plots
