"""
Advanced weight analysis and clustering for critical weight research.

Provides clustering, correlation analysis, and weight relationship discovery
to understand patterns in critical weights.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

class WeightAnalyzer:
    """Advanced analysis of weight sensitivity patterns."""
    
    def __init__(self):
        self.results = {}
    
    def cluster_weights(
        self, 
        sensitivity_dict: Dict[str, Any],  # Can accept numpy arrays or tensors
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[str, Any]:
        """Cluster weights by sensitivity patterns."""
        
        # Flatten all sensitivity scores
        all_scores = []
        layer_info = []
        
        for layer_name, data in sensitivity_dict.items():
            # Handle both torch tensors and numpy arrays
            if isinstance(data, torch.Tensor):
                flat_scores = data.flatten().cpu().numpy()
            else:
                flat_scores = np.array(data).flatten()
            
            all_scores.extend(flat_scores)
            layer_info.extend([layer_name] * len(flat_scores))
        
        scores_array = np.array(all_scores).reshape(-1, 1)
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(scores_array)
        
        # Analyze clusters
        cluster_stats = {}
        for i in range(n_clusters if method == "kmeans" else len(set(cluster_labels))):
            mask = cluster_labels == i
            cluster_scores = scores_array[mask]
            cluster_stats[f"cluster_{i}"] = {
                "size": len(cluster_scores),
                "mean_sensitivity": float(cluster_scores.mean()),
                "std_sensitivity": float(cluster_scores.std()),
                "min_sensitivity": float(cluster_scores.min()),
                "max_sensitivity": float(cluster_scores.max())
            }
        
        return {
            "cluster_labels": cluster_labels,
            "cluster_stats": cluster_stats,
            "silhouette_score": silhouette_score(scores_array, cluster_labels) if len(set(cluster_labels)) > 1 else -1
        }
    
    def analyze_sensitivity_correlations(
        self, 
        sensitivity_dict: Dict[str, Any]  # Can accept numpy arrays or tensors
    ) -> Dict[str, Any]:
        """Analyze correlations between layer sensitivities."""
        
        # Convert all to numpy and collect layer-wise statistics
        layer_stats = {}
        for layer_name, data in sensitivity_dict.items():
            # Handle both torch tensors and numpy arrays
            if isinstance(data, torch.Tensor):
                flat_scores = data.flatten().cpu().numpy()
            else:
                flat_scores = np.array(data).flatten()
            
            layer_stats[layer_name] = {
                "mean": np.mean(flat_scores),
                "std": np.std(flat_scores),
                "median": np.median(flat_scores),
                "p95": np.percentile(flat_scores, 95)
            }
    
    def find_weight_hotspots(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor],
        percentile_threshold: float = 95.0
    ) -> Dict[str, Any]:
        """Identify weight 'hotspots' - regions of consistently high sensitivity."""
        
        hotspots = {}
        
        for layer_name, tensor in sensitivity_dict.items():
            scores = tensor.cpu().numpy()
            threshold = np.percentile(scores.flatten(), percentile_threshold)
            
            # Find coordinates of high-sensitivity weights
            high_sens_coords = np.where(scores >= threshold)
            
            hotspots[layer_name] = {
                "threshold": float(threshold),
                "num_hotspot_weights": len(high_sens_coords[0]),
                "total_weights": scores.size,
                "hotspot_percentage": (len(high_sens_coords[0]) / scores.size) * 100,
                "coordinates": list(zip(*high_sens_coords)) if len(high_sens_coords[0]) > 0 else [],
                "mean_hotspot_score": float(scores[high_sens_coords].mean()) if len(high_sens_coords[0]) > 0 else 0.0
            }
        
        return hotspots
    
    def compute_sensitivity_statistics(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive statistics for each layer's sensitivity."""
        
        stats = {}
        
        for layer_name, tensor in sensitivity_dict.items():
            scores = tensor.flatten().cpu().numpy()
            
            stats[layer_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "skewness": float(self._compute_skewness(scores)),
                "kurtosis": float(self._compute_kurtosis(scores)),
                "num_weights": len(scores),
                "num_zeros": int(np.sum(scores == 0)),
                "zero_percentage": float((np.sum(scores == 0) / len(scores)) * 100)
            }
        
        return stats
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurt
    
    def generate_analysis_report(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        report = {
            "summary_statistics": self.compute_sensitivity_statistics(sensitivity_dict),
            "layer_correlations": self.analyze_layer_correlations(sensitivity_dict),
            "weight_hotspots": self.find_weight_hotspots(sensitivity_dict),
            "clustering_analysis": self.cluster_weights(sensitivity_dict)
        }
        
        return report
