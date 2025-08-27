"""
Perturbation methods for critical weight analysis.

Implements various weight perturbation strategies for testing sensitivity.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np

try:
    from transformers import PreTrainedModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    class PreTrainedModel:
        pass

logger = logging.getLogger(__name__)


class WeightPerturber:
    """
    Weight perturbation utility for sensitivity testing.
    
    Supports various perturbation methods including zeroing, sign flipping,
    Gaussian noise, and bit flipping.
    """
    
    def __init__(self, model: PreTrainedModel):
        """
        Initialize perturber with model.
        
        Args:
            model: The transformer model to perturb
        """
        self.model = model
        self.original_weights = {}
        self._backup_weights()
    
    def _backup_weights(self):
        """Backup original weights for restoration."""
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()
    
    def restore_weights(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])
    
    def perturb_weights(
        self,
        target_weights: Dict[str, List[Tuple[str, Tuple, float]]],
        method: str = "zero",
        **kwargs
    ) -> Dict[str, int]:
        """
        Apply perturbation to specified weights.
        
        Args:
            target_weights: Dictionary mapping layer names to lists of 
                          (param_name, indices, score) tuples
            method: Perturbation method ('zero', 'sign_flip', 'gauss_noise', 'bit_flip')
            **kwargs: Additional parameters for perturbation methods
            
        Returns:
            Dictionary with perturbation statistics
        """
        stats = {"total_perturbed": 0, "layers_affected": 0}
        
        for layer_name, weight_list in target_weights.items():
            if not weight_list:
                continue
                
            stats["layers_affected"] += 1
            
            for param_name, indices, score in weight_list:
                if param_name in dict(self.model.named_parameters()):
                    param = dict(self.model.named_parameters())[param_name]
                    
                    if method == "zero":
                        self._zero_weight(param, indices)
                    elif method == "sign_flip":
                        self._sign_flip_weight(param, indices)
                    elif method == "gauss_noise":
                        scale = kwargs.get("scale", 0.1)
                        self._gauss_noise_weight(param, indices, scale)
                    elif method == "bit_flip":
                        prob = kwargs.get("prob", 0.1)
                        self._bit_flip_weight(param, indices, prob)
                    else:
                        raise ValueError(f"Unknown perturbation method: {method}")
                    
                    stats["total_perturbed"] += 1
        
        logger.info(f"Applied {method} perturbation to {stats['total_perturbed']} weights "
                   f"across {stats['layers_affected']} layers")
        
        return stats
    
    def _zero_weight(self, param: torch.Tensor, indices: Tuple):
        """Zero out specific weight."""
        param.data[indices] = 0.0
    
    def _sign_flip_weight(self, param: torch.Tensor, indices: Tuple):
        """Flip sign of specific weight."""
        param.data[indices] = -param.data[indices]
    
    def _gauss_noise_weight(self, param: torch.Tensor, indices: Tuple, scale: float):
        """Add Gaussian noise to specific weight."""
        noise = torch.randn_like(param.data[indices]) * scale * torch.abs(param.data[indices])
        param.data[indices] = param.data[indices] + noise
    
    def _bit_flip_weight(self, param: torch.Tensor, indices: Tuple, prob: float):
        """Apply bit flip perturbation to specific weight."""
        # Convert to bits and randomly flip with given probability
        original_value = param.data[indices].clone()
        
        # Simple bit flip approximation: randomly change sign or scale
        if torch.rand(1).item() < prob:
            if torch.rand(1).item() < 0.5:
                # Flip sign
                param.data[indices] = -param.data[indices]
            else:
                # Scale by random factor
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                param.data[indices] = param.data[indices] * scale


def apply_perturbation(
    model: PreTrainedModel,
    target_weights: Dict[str, List[Tuple[str, Tuple, float]]],
    method: str = "zero",
    **kwargs
) -> WeightPerturber:
    """
    Apply perturbation to model weights and return perturber for restoration.
    
    Args:
        model: The transformer model
        target_weights: Dictionary mapping layer names to weight specifications
        method: Perturbation method ('zero', 'sign_flip', 'gauss_noise', 'bit_flip')
        **kwargs: Additional parameters for perturbation methods
        
    Returns:
        WeightPerturber instance for weight restoration
        
    Examples:
        >>> perturber = apply_perturbation(model, top_weights, "sign_flip")
        >>> # Test perturbed model
        >>> perturber.restore_weights()  # Restore original weights
    """
    perturber = WeightPerturber(model)
    perturber.perturb_weights(target_weights, method, **kwargs)
    return perturber


def create_control_groups(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
    methods: List[str] = ["random", "bottom"]
) -> Dict[str, Dict[str, List[Tuple[str, Tuple, float]]]]:
    """
    Create control groups for perturbation experiments.
    
    Args:
        sensitivity_dict: Dictionary mapping parameter names to sensitivity tensors
        k: Number of weights to select for each control group
        methods: List of control methods ('random', 'bottom')
        
    Returns:
        Dictionary mapping control method names to weight selections
        
    Examples:
        >>> controls = create_control_groups(sensitivity, k=100, methods=["random", "bottom"])
        >>> random_weights = controls["random"]
        >>> bottom_weights = controls["bottom"]
    """
    controls = {}
    
    # Collect all weights with their scores
    all_weights = []
    for param_name, sens_tensor in sensitivity_dict.items():
        flat_sens = sens_tensor.flatten()
        for flat_idx, value in enumerate(flat_sens):
            multi_idx = torch.unravel_index(torch.tensor(flat_idx), sens_tensor.shape)
            multi_idx_tuple = tuple(idx.item() for idx in multi_idx)
            all_weights.append((param_name, multi_idx_tuple, float(value)))
    
    if "random" in methods:
        # Random selection
        random_indices = np.random.choice(len(all_weights), size=min(k, len(all_weights)), replace=False)
        random_weights = [all_weights[i] for i in random_indices]
        
        # Group by parameter name
        controls["random"] = {}
        for param_name, indices, score in random_weights:
            if param_name not in controls["random"]:
                controls["random"][param_name] = []
            controls["random"][param_name].append((param_name, indices, score))
    
    if "bottom" in methods:
        # Bottom-K selection (least sensitive)
        all_weights.sort(key=lambda x: x[2])  # Sort by sensitivity score ascending
        bottom_weights = all_weights[:k]
        
        # Group by parameter name
        controls["bottom"] = {}
        for param_name, indices, score in bottom_weights:
            if param_name not in controls["bottom"]:
                controls["bottom"][param_name] = []
            controls["bottom"][param_name].append((param_name, indices, score))
    
    logger.info(f"Created control groups: {list(controls.keys())} with {k} weights each")
    return controls


def compute_perturbation_effects(
    model: PreTrainedModel,
    tokenizer,
    eval_texts: List[str],
    target_weights: Dict[str, List[Tuple[str, Tuple, float]]],
    methods: List[str] = ["zero", "sign_flip"],
    eval_metric: str = "perplexity",
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute effects of different perturbation methods on model performance.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        eval_texts: List of texts for evaluation
        target_weights: Dictionary mapping layer names to weight specifications
        methods: List of perturbation methods to test
        eval_metric: Evaluation metric ('perplexity', 'loss')
        **kwargs: Additional parameters for perturbation methods
        
    Returns:
        Dictionary mapping perturbation methods to evaluation results
        
    Examples:
        >>> effects = compute_perturbation_effects(
        ...     model, tokenizer, eval_texts, top_weights, 
        ...     methods=["zero", "sign_flip", "gauss_noise"],
        ...     scale=0.1  # for gauss_noise
        ... )
        >>> print(f"Zero perturbation effect: {effects['zero']['perplexity']}")
    """
    from src.eval.perplexity import compute_perplexity
    
    results = {}
    
    # Compute baseline performance
    baseline_ppl = compute_perplexity(model, tokenizer, eval_texts)
    results["baseline"] = {"perplexity": baseline_ppl}
    
    for method in methods:
        logger.info(f"Testing perturbation method: {method}")
        
        # Apply perturbation
        perturber = apply_perturbation(model, target_weights, method, **kwargs)
        
        try:
            # Evaluate perturbed model
            perturbed_ppl = compute_perplexity(model, tokenizer, eval_texts)
            results[method] = {
                "perplexity": perturbed_ppl,
                "delta_perplexity": perturbed_ppl - baseline_ppl,
                "relative_change": (perturbed_ppl - baseline_ppl) / baseline_ppl
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {method} perturbation: {e}")
            results[method] = {"error": str(e)}
        
        finally:
            # Restore original weights
            perturber.restore_weights()
    
    return results


def stability_analysis(
    model: PreTrainedModel,
    tokenizer,
    texts: List[str],
    metric: str = "grad_x_weight",
    k: int = 100,
    num_seeds: int = 3,
    num_batches: int = 3
) -> Dict[str, float]:
    """
    Analyze stability of top-K weight selection across different seeds and data batches.
    
    Computes Jaccard similarity of top-K selections across runs.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of calibration texts
        metric: Sensitivity metric to use
        k: Number of top weights to compare
        num_seeds: Number of random seeds to test
        num_batches: Number of different data batches to test
        
    Returns:
        Dictionary with stability metrics
        
    Examples:
        >>> stability = stability_analysis(model, tokenizer, texts, k=100, num_seeds=5)
        >>> print(f"Seed stability (Jaccard): {stability['seed_jaccard_mean']:.3f}")
        >>> print(f"Batch stability (Jaccard): {stability['batch_jaccard_mean']:.3f}")
    """
    from src.sensitivity.metrics import compute_sensitivity
    from src.sensitivity.rank import rank_topk
    
    def get_weight_set(weights_dict):
        """Convert weight dictionary to set of (param_name, indices) tuples."""
        weight_set = set()
        for layer_name, weight_list in weights_dict.items():
            for param_name, indices, score in weight_list:
                weight_set.add((param_name, indices))
        return weight_set
    
    def jaccard_similarity(set1, set2):
        """Compute Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    # Test across different seeds
    seed_sets = []
    original_seed = torch.initial_seed()
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        sensitivity = compute_sensitivity(model, tokenizer, texts, metric)
        top_weights = rank_topk(sensitivity, k, per_layer=False, global_ranking=True)
        weight_set = get_weight_set(top_weights)
        seed_sets.append(weight_set)
    
    # Restore original seed
    torch.manual_seed(original_seed)
    
    # Test across different data batches
    batch_sets = []
    for i in range(num_batches):
        # Use different subset of texts
        start_idx = (i * len(texts)) // num_batches
        end_idx = ((i + 1) * len(texts)) // num_batches
        batch_texts = texts[start_idx:end_idx] if end_idx > start_idx else texts
        
        if batch_texts:
            sensitivity = compute_sensitivity(model, tokenizer, batch_texts, metric)
            top_weights = rank_topk(sensitivity, k, per_layer=False, global_ranking=True)
            weight_set = get_weight_set(top_weights)
            batch_sets.append(weight_set)
    
    # Compute pairwise Jaccard similarities
    seed_jaccards = []
    for i in range(len(seed_sets)):
        for j in range(i + 1, len(seed_sets)):
            jaccard = jaccard_similarity(seed_sets[i], seed_sets[j])
            seed_jaccards.append(jaccard)
    
    batch_jaccards = []
    for i in range(len(batch_sets)):
        for j in range(i + 1, len(batch_sets)):
            jaccard = jaccard_similarity(batch_sets[i], batch_sets[j])
            batch_jaccards.append(jaccard)
    
    results = {
        "seed_jaccard_mean": np.mean(seed_jaccards) if seed_jaccards else 0.0,
        "seed_jaccard_std": np.std(seed_jaccards) if seed_jaccards else 0.0,
        "batch_jaccard_mean": np.mean(batch_jaccards) if batch_jaccards else 0.0,
        "batch_jaccard_std": np.std(batch_jaccards) if batch_jaccards else 0.0,
        "num_seed_comparisons": len(seed_jaccards),
        "num_batch_comparisons": len(batch_jaccards)
    }
    
    logger.info(f"Stability analysis completed: "
               f"Seed Jaccard = {results['seed_jaccard_mean']:.3f} ± {results['seed_jaccard_std']:.3f}, "
               f"Batch Jaccard = {results['batch_jaccard_mean']:.3f} ± {results['batch_jaccard_std']:.3f}")
    
    return results
