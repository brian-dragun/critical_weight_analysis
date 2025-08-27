"""
Enhanced visualization functions for advanced critical weight analysis.

Provides plotting functions for clustering, architecture analysis, temporal stability,
and advanced perturbation results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_weight_clustering(
    clustering_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Weight Sensitivity Clustering Analysis"
) -> plt.Figure:
    """Plot weight clustering analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    cluster_stats = clustering_results.get("cluster_stats", {})
    
    # Plot 1: Cluster sizes
    ax1 = axes[0, 0]
    clusters = list(cluster_stats.keys())
    sizes = [cluster_stats[c]["size"] for c in clusters]
    
    ax1.bar(clusters, sizes, alpha=0.7)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Weights')
    ax1.set_title('Cluster Sizes')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(sizes):
        ax1.text(i, v + max(sizes) * 0.01, str(v), ha='center', va='bottom')
    
    # Plot 2: Mean sensitivity by cluster
    ax2 = axes[0, 1]
    mean_sens = [cluster_stats[c]["mean_sensitivity"] for c in clusters]
    
    bars = ax2.bar(clusters, mean_sens, alpha=0.7)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Mean Sensitivity')
    ax2.set_title('Mean Sensitivity by Cluster')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color bars by sensitivity level
    colors = sns.color_palette("viridis", len(bars))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Plot 3: Sensitivity range by cluster
    ax3 = axes[1, 0]
    min_sens = [cluster_stats[c]["min_sensitivity"] for c in clusters]
    max_sens = [cluster_stats[c]["max_sensitivity"] for c in clusters]
    
    x_pos = range(len(clusters))
    ax3.errorbar(x_pos, mean_sens, 
                yerr=[np.array(mean_sens) - np.array(min_sens), 
                      np.array(max_sens) - np.array(mean_sens)],
                fmt='o', capsize=5, capthick=2)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Sensitivity Range')
    ax3.set_title('Sensitivity Range by Cluster')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(clusters)
    
    # Plot 4: Cluster quality metrics
    ax4 = axes[1, 1]
    silhouette_score = clustering_results.get("silhouette_score", 0)
    
    ax4.bar(['Silhouette Score'], [silhouette_score], alpha=0.7)
    ax4.set_ylabel('Score')
    ax4.set_title('Clustering Quality')
    ax4.set_ylim(-1, 1)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add score label
    ax4.text(0, silhouette_score + 0.05, f'{silhouette_score:.3f}', 
             ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved clustering plot to {output_path}")
    
    return fig

def plot_architecture_analysis(
    arch_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Architecture Component Sensitivity Analysis"
) -> plt.Figure:
    """Plot architecture analysis results."""
    
    component_analysis = arch_results.get("component_analysis", {})
    
    # Prepare data
    components = []
    mean_sensitivities = []
    total_weights = []
    
    for comp_name, comp_data in component_analysis.items():
        if "overall_stats" in comp_data:
            components.append(comp_name.replace("_", "\n"))
            mean_sensitivities.append(comp_data["overall_stats"].get("mean_sensitivity", 0))
            total_weights.append(comp_data["overall_stats"].get("total_weights", 0))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Mean sensitivity by component
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(components)), mean_sensitivities, alpha=0.7)
    ax1.set_xlabel('Architecture Component')
    ax1.set_ylabel('Mean Sensitivity')
    ax1.set_title('Mean Sensitivity by Component')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right')
    
    # Color bars by sensitivity
    if mean_sensitivities:
        norm_sens = np.array(mean_sensitivities) / max(mean_sensitivities)
        colors = plt.cm.viridis(norm_sens)
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
    
    # Plot 2: Weight distribution by component
    ax2 = axes[0, 1]
    ax2.pie(total_weights, labels=components, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Weight Distribution Across Components')
    
    # Plot 3: Depth analysis (if available)
    ax3 = axes[1, 0]
    depth_analysis = arch_results.get("depth_analysis", {})
    depth_trends = depth_analysis.get("depth_trends", {})
    
    if "layer_ranges" in depth_trends:
        layer_ranges = depth_trends["layer_ranges"]
        early_sens = depth_trends.get("early_layer_sensitivity", 0)
        middle_sens = depth_trends.get("middle_layer_sensitivity", 0)
        late_sens = depth_trends.get("late_layer_sensitivity", 0)
        
        depth_labels = ['Early', 'Middle', 'Late']
        depth_values = [early_sens, middle_sens if middle_sens else 0, late_sens]
        
        ax3.bar(depth_labels, depth_values, alpha=0.7)
        ax3.set_ylabel('Mean Sensitivity')
        ax3.set_title('Sensitivity by Layer Depth')
        
        # Add pattern annotation
        pattern = depth_trends.get("sensitivity_pattern", "unknown")
        ax3.text(0.5, 0.95, f'Pattern: {pattern}', transform=ax3.transAxes,
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    else:
        ax3.text(0.5, 0.5, 'No depth analysis available', transform=ax3.transAxes,
                ha='center', va='center')
        ax3.set_title('Layer Depth Analysis')
    
    # Plot 4: Attention component comparison (if available)
    ax4 = axes[1, 1]
    attention_analysis = arch_results.get("attention_analysis", {})
    component_rankings = attention_analysis.get("component_rankings", {})
    
    if "mean_sensitivities" in component_rankings:
        att_components = list(component_rankings["mean_sensitivities"].keys())
        att_sensitivities = list(component_rankings["mean_sensitivities"].values())
        
        ax4.bar(range(len(att_components)), att_sensitivities, alpha=0.7)
        ax4.set_xlabel('Attention Component')
        ax4.set_ylabel('Mean Sensitivity')
        ax4.set_title('Attention Component Sensitivity')
        ax4.set_xticks(range(len(att_components)))
        ax4.set_xticklabels([c.replace("attention_", "") for c in att_components], 
                           rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, 'No attention analysis available', transform=ax4.transAxes,
                ha='center', va='center')
        ax4.set_title('Attention Component Analysis')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved architecture analysis plot to {output_path}")
    
    return fig

def plot_temporal_stability(
    temporal_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Temporal Stability Analysis"
) -> plt.Figure:
    """Plot temporal stability analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Pairwise stability heatmap
    ax1 = axes[0, 0]
    pairwise_stability = temporal_results.get("pairwise_stability", {})
    
    if pairwise_stability:
        conditions = set()
        for pair_name in pairwise_stability.keys():
            cond1, cond2 = pair_name.split("_vs_")
            conditions.update([cond1, cond2])
        
        conditions = sorted(list(conditions))
        n_conditions = len(conditions)
        
        if n_conditions > 1:
            stability_matrix = np.ones((n_conditions, n_conditions))
            
            for pair_name, pair_data in pairwise_stability.items():
                cond1, cond2 = pair_name.split("_vs_")
                i, j = conditions.index(cond1), conditions.index(cond2)
                stability = pair_data.get("mean_stability", 0)
                stability_matrix[i, j] = stability
                stability_matrix[j, i] = stability
            
            im = ax1.imshow(stability_matrix, cmap='viridis', vmin=0, vmax=1)
            ax1.set_xticks(range(n_conditions))
            ax1.set_yticks(range(n_conditions))
            ax1.set_xticklabels([c.replace("_", "\n") for c in conditions], rotation=45)
            ax1.set_yticklabels([c.replace("_", "\n") for c in conditions])
            ax1.set_title('Pairwise Stability Matrix')
            
            # Add text annotations
            for i in range(n_conditions):
                for j in range(n_conditions):
                    text = f'{stability_matrix[i, j]:.2f}'
                    ax1.text(j, i, text, ha="center", va="center",
                           color="white" if stability_matrix[i, j] < 0.5 else "black")
            
            plt.colorbar(im, ax=ax1, label='Jaccard Similarity')
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for pairwise comparison', 
                    transform=ax1.transAxes, ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, 'No pairwise stability data', transform=ax1.transAxes,
                ha='center', va='center')
    
    ax1.set_title('Pairwise Stability Matrix')
    
    # Plot 2: Condition stability summary
    ax2 = axes[0, 1]
    condition_summary = temporal_results.get("condition_stability_summary", {})
    
    if condition_summary:
        cond_names = list(condition_summary.keys())
        mean_stabilities = [condition_summary[c].get("mean_temporal_stability", 0) 
                           for c in cond_names]
        
        bars = ax2.bar(range(len(cond_names)), mean_stabilities, alpha=0.7)
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Mean Temporal Stability')
        ax2.set_title('Temporal Stability by Condition')
        ax2.set_xticks(range(len(cond_names)))
        ax2.set_xticklabels([c.replace("_", "\n") for c in cond_names], rotation=45)
        ax2.set_ylim(0, 1)
        
        # Color bars by stability
        colors = plt.cm.RdYlGn([s for s in mean_stabilities])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    else:
        ax2.text(0.5, 0.5, 'No condition stability data', transform=ax2.transAxes,
                ha='center', va='center')
    
    # Plot 3: Stability trends (placeholder for time series)
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Stability trends over time\n(requires time series data)', 
            transform=ax3.transAxes, ha='center', va='center')
    ax3.set_title('Stability Trends Over Time')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    
    # Collect summary statistics
    summary_stats = {}
    if pairwise_stability:
        all_stabilities = [data.get("mean_stability", 0) for data in pairwise_stability.values()]
        if all_stabilities:
            summary_stats["Mean Pairwise Stability"] = np.mean(all_stabilities)
            summary_stats["Std Pairwise Stability"] = np.std(all_stabilities)
            summary_stats["Min Pairwise Stability"] = np.min(all_stabilities)
            summary_stats["Max Pairwise Stability"] = np.max(all_stabilities)
    
    if summary_stats:
        labels = list(summary_stats.keys())
        values = list(summary_stats.values())
        
        ax4.barh(range(len(labels)), values, alpha=0.7)
        ax4.set_yticks(range(len(labels)))
        ax4.set_yticklabels([l.replace(" ", "\n") for l in labels])
        ax4.set_xlabel('Value')
        ax4.set_title('Summary Statistics')
        ax4.set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            ax4.text(v + 0.02, i, f'{v:.3f}', va='center')
    else:
        ax4.text(0.5, 0.5, 'No summary statistics available', transform=ax4.transAxes,
                ha='center', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal stability plot to {output_path}")
    
    return fig

def plot_advanced_perturbations(
    perturbation_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Advanced Perturbation Analysis"
) -> plt.Figure:
    """Plot advanced perturbation results."""
    
    n_methods = len(perturbation_results)
    if n_methods == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No perturbation results available', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Determine subplot layout
    if n_methods <= 2:
        rows, cols = 1, n_methods
        fig_size = (8 * cols, 6)
    elif n_methods <= 4:
        rows, cols = 2, 2
        fig_size = (12, 10)
    else:
        rows, cols = 2, 3
        fig_size = (15, 10)
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16)
    
    for i, (method, results) in enumerate(perturbation_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if method == "progressive" and isinstance(results, list):
            # Progressive perturbation results
            severity_levels = [r.get("severity_level", 0) for r in results]
            eval_results = [r.get("evaluation_results", {}) for r in results]
            
            if eval_results and "perplexity" in eval_results[0]:
                perplexities = [er.get("perplexity", 0) for er in eval_results]
                ax.plot(severity_levels, perplexities, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Severity Level')
                ax.set_ylabel('Perplexity')
                ax.set_title(f'{method.title()} Perturbation')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No evaluation data for {method}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{method.title()} Perturbation')
        
        elif isinstance(results, dict):
            # Layer-wise perturbation statistics
            if any("total_perturbation" in layer_data or "weights_changed" in layer_data 
                   for layer_data in results.values() if isinstance(layer_data, dict)):
                
                layers = []
                values = []
                
                for layer_name, layer_data in results.items():
                    if isinstance(layer_data, dict):
                        layers.append(layer_name[:15] + "..." if len(layer_name) > 15 else layer_name)
                        
                        if "total_perturbation" in layer_data:
                            values.append(layer_data["total_perturbation"])
                            ylabel = "Total Perturbation"
                        elif "weights_changed" in layer_data:
                            values.append(layer_data["weights_changed"])
                            ylabel = "Weights Changed"
                        else:
                            values.append(0)
                            ylabel = "Value"
                
                if layers and values:
                    bars = ax.bar(range(len(layers)), values, alpha=0.7)
                    ax.set_xlabel('Layer')
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'{method.title()} Perturbation')
                    ax.set_xticks(range(len(layers)))
                    ax.set_xticklabels(layers, rotation=45, ha='right')
                    
                    # Color bars by value
                    if values:
                        norm_values = np.array(values) / max(values) if max(values) > 0 else np.zeros_like(values)
                        colors = plt.cm.viridis(norm_values)
                        for bar, color in zip(bars, colors):
                            bar.set_color(color)
                else:
                    ax.text(0.5, 0.5, f'No valid data for {method}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{method.title()} Perturbation')
            else:
                ax.text(0.5, 0.5, f'Unsupported data format for {method}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{method.title()} Perturbation')
        else:
            ax.text(0.5, 0.5, f'No data available for {method}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{method.title()} Perturbation')
    
    # Hide unused subplots
    for i in range(len(perturbation_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved advanced perturbations plot to {output_path}")
    
    return fig

def plot_downstream_task_comparison(
    task_results: Dict[str, float],
    baseline_results: Optional[Dict[str, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Downstream Task Performance"
) -> plt.Figure:
    """Plot downstream task evaluation results."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract task names and scores
    tasks = []
    scores = []
    baseline_scores = []
    
    for task_name, score in task_results.items():
        if task_name.endswith("_accuracy"):
            task_display = task_name.replace("_accuracy", "").upper()
            tasks.append(task_display)
            scores.append(score)
            
            if baseline_results and task_name in baseline_results:
                baseline_scores.append(baseline_results[task_name])
            else:
                baseline_scores.append(None)
    
    if not tasks:
        ax.text(0.5, 0.5, 'No downstream task results available', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
        return fig
    
    x_pos = np.arange(len(tasks))
    width = 0.35
    
    # Plot current results
    bars1 = ax.bar(x_pos - width/2, scores, width, label='Current', alpha=0.7)
    
    # Plot baseline results if available
    if any(bs is not None for bs in baseline_scores):
        valid_baseline = [bs if bs is not None else 0 for bs in baseline_scores]
        bars2 = ax.bar(x_pos + width/2, valid_baseline, width, label='Baseline', alpha=0.7)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved downstream task comparison plot to {output_path}")
    
    return fig

def save_enhanced_plots(
    enhanced_results: Dict[str, Any],
    output_dir: Union[str, Path] = "plots",
    prefix: str = "enhanced_"
) -> Dict[str, Path]:
    """Save all enhanced analysis plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Weight clustering plot
    if "weight_analysis" in enhanced_results:
        weight_analysis = enhanced_results["weight_analysis"]
        if "clustering_results" in weight_analysis:
            plot_path = output_dir / f"{prefix}weight_clustering.png"
            plot_weight_clustering(weight_analysis["clustering_results"], plot_path)
            saved_plots["weight_clustering"] = plot_path
    
    # Architecture analysis plot
    if "architecture_analysis" in enhanced_results:
        plot_path = output_dir / f"{prefix}architecture_analysis.png"
        plot_architecture_analysis(enhanced_results["architecture_analysis"], plot_path)
        saved_plots["architecture_analysis"] = plot_path
    
    # Temporal stability plot
    if "temporal_stability" in enhanced_results:
        plot_path = output_dir / f"{prefix}temporal_stability.png"
        plot_temporal_stability(enhanced_results["temporal_stability"], plot_path)
        saved_plots["temporal_stability"] = plot_path
    
    # Advanced perturbations plot
    if "advanced_perturbations" in enhanced_results:
        plot_path = output_dir / f"{prefix}advanced_perturbations.png"
        plot_advanced_perturbations(enhanced_results["advanced_perturbations"], plot_path)
        saved_plots["advanced_perturbations"] = plot_path
    
    # Downstream tasks plot
    if "downstream_tasks" in enhanced_results:
        plot_path = output_dir / f"{prefix}downstream_tasks.png"
        plot_downstream_task_comparison(enhanced_results["downstream_tasks"], output_path=plot_path)
        saved_plots["downstream_tasks"] = plot_path
    
    logger.info(f"Saved {len(saved_plots)} enhanced analysis plots")
    return saved_plots
