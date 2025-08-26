"""
Top-K ranking utilities for critical weight analysis.

Provides functions to rank and select the most sensitive weights.
"""

import logging
from typing import Dict, List, Tuple, Optional
import torch
import pandas as pd

logger = logging.getLogger(__name__)


def rank_topk(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
    per_layer: bool = True,
    global_ranking: bool = False,
) -> Dict[str, List[Tuple[str, Tuple, float]]]:
    """
    Rank and select top-K weights by sensitivity metric.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        k: Number of top weights to select
        per_layer: If True, select top-K per layer; if False, select top-K globally
        global_ranking: If True, rank all weights globally first
        
    Returns:
        Dictionary mapping layer names to lists of (param_name, indices, score) tuples
        
    Examples:
        >>> sensitivity = compute_sensitivity(model, tokenizer, texts, "grad_x_weight")
        >>> topk = rank_topk(sensitivity, k=100, per_layer=True)
        >>> print(f"Top weights in first layer: {topk[list(topk.keys())[0]][:5]}")
    """
    logger.info(f"Ranking top-{k} weights (per_layer={per_layer}, global={global_ranking})")
    
    if global_ranking:
        return _rank_topk_global(sensitivity_dict, k, per_layer)
    else:
        return _rank_topk_per_layer(sensitivity_dict, k)


def _rank_topk_per_layer(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, List[Tuple[str, Tuple, float]]]:
    """Rank top-K weights within each layer separately."""
    topk_dict = {}
    
    for param_name, sens_tensor in sensitivity_dict.items():
        # Flatten tensor and get top-k indices
        flat_sens = sens_tensor.flatten()
        values, indices = torch.topk(flat_sens, k=min(k, flat_sens.numel()), largest=True)
        
        # Convert flat indices back to multi-dimensional indices
        topk_weights = []
        for i, (value, flat_idx) in enumerate(zip(values, indices)):
            # Convert flat index to multidimensional index
            multi_idx = torch.unravel_index(flat_idx, sens_tensor.shape)
            multi_idx_tuple = tuple(idx.item() for idx in multi_idx)
            
            topk_weights.append((param_name, multi_idx_tuple, float(value)))
        
        topk_dict[param_name] = topk_weights
        
        logger.debug(f"Selected top-{len(topk_weights)} weights from {param_name}")
    
    return topk_dict


def _rank_topk_global(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
    per_layer: bool,
) -> Dict[str, List[Tuple[str, Tuple, float]]]:
    """Rank weights globally across all layers, then optionally distribute per layer."""
    # Collect all weights with their scores
    all_weights = []
    
    for param_name, sens_tensor in sensitivity_dict.items():
        flat_sens = sens_tensor.flatten()
        
        for flat_idx, value in enumerate(flat_sens):
            multi_idx = torch.unravel_index(torch.tensor(flat_idx), sens_tensor.shape)
            multi_idx_tuple = tuple(idx.item() for idx in multi_idx)
            
            all_weights.append((param_name, multi_idx_tuple, float(value)))
    
    # Sort globally by sensitivity
    all_weights.sort(key=lambda x: x[2], reverse=True)
    
    if per_layer:
        # Distribute top-k across layers proportionally
        total_weights = len(all_weights)
        layer_weights = {name: sens.numel() for name, sens in sensitivity_dict.items()}
        
        topk_dict = {}
        assigned = 0
        
        for param_name in sensitivity_dict.keys():
            # Calculate proportion for this layer
            layer_prop = layer_weights[param_name] / sum(layer_weights.values())
            layer_k = max(1, int(k * layer_prop))
            
            # Find top weights for this layer
            layer_topk = [w for w in all_weights if w[0] == param_name][:layer_k]
            topk_dict[param_name] = layer_topk
            assigned += len(layer_topk)
        
        # Distribute remaining slots to layers with highest sensitivity
        remaining = k - assigned
        if remaining > 0:
            unassigned_weights = [w for w in all_weights 
                                if not any(w in layer_list for layer_list in topk_dict.values())]
            
            for weight in unassigned_weights[:remaining]:
                topk_dict[weight[0]].append(weight)
    
    else:
        # Simply take top-k globally
        global_topk = all_weights[:k]
        
        topk_dict = {}
        for param_name in sensitivity_dict.keys():
            topk_dict[param_name] = [w for w in global_topk if w[0] == param_name]
    
    return topk_dict


def get_topk_statistics(
    topk_dict: Dict[str, List[Tuple[str, Tuple, float]]]
) -> Dict[str, Dict]:
    """
    Compute statistics for top-K weights.
    
    Args:
        topk_dict: Dictionary from rank_topk
        
    Returns:
        Dictionary mapping parameter names to statistics
    """
    stats = {}
    
    for param_name, weight_list in topk_dict.items():
        if not weight_list:
            stats[param_name] = {
                "count": 0,
                "mean_sensitivity": 0.0,
                "max_sensitivity": 0.0,
                "min_sensitivity": 0.0,
            }
            continue
        
        sensitivities = [w[2] for w in weight_list]
        
        stats[param_name] = {
            "count": len(weight_list),
            "mean_sensitivity": sum(sensitivities) / len(sensitivities),
            "max_sensitivity": max(sensitivities),
            "min_sensitivity": min(sensitivities),
            "std_sensitivity": torch.tensor(sensitivities).std().item(),
        }
    
    return stats


def export_topk_to_csv(
    topk_dict: Dict[str, List[Tuple[str, Tuple, float]]],
    output_path: str,
) -> None:
    """
    Export top-K weights to CSV format.
    
    Args:
        topk_dict: Dictionary from rank_topk
        output_path: Path to save CSV file
    """
    rows = []
    
    for param_name, weight_list in topk_dict.items():
        for rank, (_, indices, sensitivity) in enumerate(weight_list):
            row = {
                "parameter_name": param_name,
                "rank": rank + 1,
                "sensitivity": sensitivity,
                "indices": str(indices),  # Store as string for CSV
            }
            
            # Add individual index columns for easier analysis
            for i, idx in enumerate(indices):
                row[f"index_{i}"] = idx
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(rows)} top-K weights to {output_path}")


def load_topk_from_csv(csv_path: str) -> Dict[str, List[Tuple[str, Tuple, float]]]:
    """
    Load top-K weights from CSV format.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary in rank_topk format
    """
    df = pd.read_csv(csv_path)
    
    topk_dict = {}
    
    for _, row in df.iterrows():
        param_name = row["parameter_name"]
        sensitivity = row["sensitivity"]
        
        # Reconstruct indices tuple
        indices_str = row["indices"]
        indices = eval(indices_str)  # Note: eval is unsafe, consider ast.literal_eval
        
        if param_name not in topk_dict:
            topk_dict[param_name] = []
        
        topk_dict[param_name].append((param_name, indices, sensitivity))
    
    # Sort by sensitivity within each parameter
    for param_name in topk_dict:
        topk_dict[param_name].sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Loaded top-K weights from {csv_path}")
    return topk_dict


def compare_topk_rankings(
    topk_dict1: Dict[str, List[Tuple[str, Tuple, float]]],
    topk_dict2: Dict[str, List[Tuple[str, Tuple, float]]],
    name1: str = "ranking1",
    name2: str = "ranking2",
) -> Dict[str, float]:
    """
    Compare two top-K rankings to measure overlap.
    
    Args:
        topk_dict1: First ranking
        topk_dict2: Second ranking
        name1: Name for first ranking
        name2: Name for second ranking
        
    Returns:
        Dictionary with overlap statistics
    """
    results = {
        "jaccard_similarity": 0.0,
        "overlap_percentage": 0.0,
        "rank_correlation": 0.0,
    }
    
    # Collect all weight positions from both rankings
    weights1 = set()
    weights2 = set()
    
    for param_name in topk_dict1:
        for _, indices, _ in topk_dict1[param_name]:
            weights1.add((param_name, indices))
    
    for param_name in topk_dict2:
        for _, indices, _ in topk_dict2[param_name]:
            weights2.add((param_name, indices))
    
    # Compute Jaccard similarity
    intersection = weights1.intersection(weights2)
    union = weights1.union(weights2)
    
    if union:
        results["jaccard_similarity"] = len(intersection) / len(union)
        results["overlap_percentage"] = len(intersection) / min(len(weights1), len(weights2))
    
    logger.info(f"Ranking comparison: {len(intersection)} overlapping weights")
    logger.info(f"Jaccard similarity: {results['jaccard_similarity']:.3f}")
    
    return results
