"""
Temporal stability analysis for critical weight research.

Analyzes how weight sensitivity rankings change across different conditions,
data batches, and model states to understand stability and consistency.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

class TemporalStabilityAnalyzer:
    """Analyzes temporal stability of weight importance rankings."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.rankings_history = defaultdict(list)
    
    def record_sensitivity_snapshot(
        self, 
        condition_name: str, 
        sensitivity_dict: Dict[str, Any]  # Can accept numpy arrays or tensors
    ) -> None:
        """Record a sensitivity snapshot for a specific condition."""
        
        snapshot = {
            "timestamp": time.time(),
            "condition": condition_name,
            "sensitivity_data": {}
        }
        
        for layer_name, data in sensitivity_dict.items():
            # Handle both torch tensors and numpy arrays
            if isinstance(data, torch.Tensor):
                tensor_data = data
            else:
                tensor_data = torch.from_numpy(np.array(data))
            
            snapshot["sensitivity_data"][layer_name] = {
                "mean": float(tensor_data.mean()),
                "std": float(tensor_data.std()),
                "max": float(tensor_data.max()),
                "min": float(tensor_data.min()),
                "shape": list(tensor_data.shape)
            }
    
    def record_ranking_snapshot(
        self,
        rankings: Dict[str, List[Tuple]],
        condition: str,
        k: int,
        timestamp: Optional[str] = None
    ) -> None:
        """Record a snapshot of top-k rankings."""
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        
        # Convert rankings to serializable format
        ranking_snapshot = {
            "timestamp": timestamp,
            "condition": condition,
            "k": k,
            "rankings": {}
        }
        
        for layer_name, layer_rankings in rankings.items():
            ranking_snapshot["rankings"][layer_name] = [
                {
                    "param_name": item[0] if len(item) > 0 else "",
                    "index": item[1] if len(item) > 1 else -1,
                    "score": float(item[2]) if len(item) > 2 else 0.0
                }
                for item in layer_rankings[:k]
            ]
        
        self.rankings_history[condition].append(ranking_snapshot)
    
    def compute_ranking_stability(
        self, 
        condition1: str, 
        condition2: str,
        k: int = 100
    ) -> Dict[str, float]:
        """Compute stability between two ranking conditions."""
        
        if condition1 not in self.rankings_history or condition2 not in self.rankings_history:
            return {}
        
        stability_scores = {}
        
        # Get latest snapshots for each condition
        snapshot1 = self.rankings_history[condition1][-1]
        snapshot2 = self.rankings_history[condition2][-1]
        
        for layer_name in snapshot1["rankings"]:
            if layer_name not in snapshot2["rankings"]:
                continue
            
            # Extract top-k indices for each snapshot
            indices1 = set([item["index"] for item in snapshot1["rankings"][layer_name][:k]])
            indices2 = set([item["index"] for item in snapshot2["rankings"][layer_name][:k]])
            
            # Compute Jaccard similarity
            intersection = len(indices1.intersection(indices2))
            union = len(indices1.union(indices2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            stability_scores[layer_name] = jaccard_similarity
        
        return stability_scores
    
    def analyze_drift_over_time(
        self, 
        condition: str,
        window_size: int = 3
    ) -> Dict[str, Any]:
        """Analyze how rankings drift over multiple snapshots."""
        
        if condition not in self.rankings_history:
            return {}
        
        snapshots = self.rankings_history[condition]
        if len(snapshots) < 2:
            return {"error": "Need at least 2 snapshots for drift analysis"}
        
        drift_analysis = {
            "condition": condition,
            "num_snapshots": len(snapshots),
            "drift_metrics": {},
            "stability_over_time": []
        }
        
        # Compute pairwise stability between consecutive snapshots
        for i in range(len(snapshots) - 1):
            snapshot1 = snapshots[i]
            snapshot2 = snapshots[i + 1]
            
            layer_stabilities = {}
            for layer_name in snapshot1["rankings"]:
                if layer_name not in snapshot2["rankings"]:
                    continue
                
                indices1 = set([item["index"] for item in snapshot1["rankings"][layer_name]])
                indices2 = set([item["index"] for item in snapshot2["rankings"][layer_name]])
                
                intersection = len(indices1.intersection(indices2))
                union = len(indices1.union(indices2))
                jaccard = intersection / union if union > 0 else 0.0
                
                layer_stabilities[layer_name] = jaccard
            
            drift_analysis["stability_over_time"].append({
                "snapshot_pair": f"{i}_{i+1}",
                "timestamp1": snapshot1["timestamp"],
                "timestamp2": snapshot2["timestamp"],
                "layer_stabilities": layer_stabilities,
                "mean_stability": np.mean(list(layer_stabilities.values())) if layer_stabilities else 0.0
            })
        
        # Compute overall drift metrics
        if drift_analysis["stability_over_time"]:
            mean_stabilities = [entry["mean_stability"] for entry in drift_analysis["stability_over_time"]]
            drift_analysis["drift_metrics"] = {
                "mean_temporal_stability": float(np.mean(mean_stabilities)),
                "std_temporal_stability": float(np.std(mean_stabilities)),
                "min_stability": float(np.min(mean_stabilities)),
                "max_stability": float(np.max(mean_stabilities)),
                "stability_trend": "increasing" if mean_stabilities[-1] > mean_stabilities[0] else "decreasing"
            }
        
        return drift_analysis
    
    def compare_across_conditions(self) -> Dict[str, Any]:
        """Compare stability across all recorded conditions."""
        
        conditions = list(self.rankings_history.keys())
        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for comparison"}
        
        comparison = {
            "conditions": conditions,
            "pairwise_stability": {},
            "condition_stability_summary": {}
        }
        
        # Pairwise comparisons
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions[i+1:], i+1):
                stability = self.compute_ranking_stability(cond1, cond2)
                comparison["pairwise_stability"][f"{cond1}_vs_{cond2}"] = {
                    "layer_stabilities": stability,
                    "mean_stability": float(np.mean(list(stability.values()))) if stability else 0.0
                }
        
        # Per-condition stability summaries
        for condition in conditions:
            drift_analysis = self.analyze_drift_over_time(condition)
            if "drift_metrics" in drift_analysis:
                comparison["condition_stability_summary"][condition] = drift_analysis["drift_metrics"]
        
        return comparison
    
    def save_analysis(self, output_path: Path) -> None:
        """Save all stability analysis data."""
        
        analysis_data = {
            "sensitivity_history": dict(self.history),
            "rankings_history": dict(self.rankings_history),
            "stability_analysis": self.compare_across_conditions()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Stability analysis saved to {output_path}")
    
    def load_analysis(self, input_path: Path) -> None:
        """Load previously saved stability analysis data."""
        
        with open(input_path, 'r') as f:
            analysis_data = json.load(f)
        
        self.history = defaultdict(list, analysis_data.get("sensitivity_history", {}))
        self.rankings_history = defaultdict(list, analysis_data.get("rankings_history", {}))
        
        logger.info(f"Stability analysis loaded from {input_path}")
