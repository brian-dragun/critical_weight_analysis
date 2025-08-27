"""
Advanced perturbation methods for critical weight analysis.

Extends basic perturbations with sophisticated techniques for research:
- Adaptive perturbations based on weight magnitudes
- Structured perturbations (layer-wise, attention-head-wise)
- Progressive perturbations with increasing severity
- Adversarial perturbations targeting specific metrics
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class PerturbationType(Enum):
    """Enhanced perturbation types."""
    ZERO = "zero"
    SIGN_FLIP = "sign_flip"
    GAUSS_NOISE = "gauss_noise"
    BIT_FLIP = "bit_flip"
    ADAPTIVE_NOISE = "adaptive_noise"
    MAGNITUDE_SCALING = "magnitude_scaling"
    QUANTIZATION = "quantization"
    DROPOUT_MASK = "dropout_mask"
    ADVERSARIAL = "adversarial"

class AdvancedPerturbationEngine:
    """Advanced perturbation methods for research experiments."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.perturbation_history = []
        self.original_state = {}
    
    def save_original_state(self, model: torch.nn.Module) -> None:
        """Save original model state for restoration."""
        self.original_state = {}
        for name, param in model.named_parameters():
            self.original_state[name] = param.data.clone()
    
    def restore_original_state(self, model: torch.nn.Module) -> None:
        """Restore model to original state."""
        for name, param in model.named_parameters():
            if name in self.original_state:
                param.data.copy_(self.original_state[name])
    
    def adaptive_noise_perturbation(
        self,
        model: torch.nn.Module,
        target_weights: Dict[str, List[Tuple]],
        noise_scale: float = 0.1,
        magnitude_adaptive: bool = True
    ) -> Dict[str, float]:
        """Apply noise scaled by weight magnitude."""
        
        perturbation_stats = {}
        
        for layer_name, weight_list in target_weights.items():
            if not weight_list:
                continue
            
            # Get the parameter
            param = dict(model.named_parameters())[layer_name]
            original_data = param.data.clone()
            
            # Apply perturbations to selected weights
            total_perturbation = 0.0
            for param_name, indices, score in weight_list:
                if isinstance(indices, tuple) and len(indices) == 1:
                    flat_idx = indices[0]
                else:
                    continue
                
                # Convert flat index to multi-dimensional indices
                unravel_idx = np.unravel_index(flat_idx, param.shape)
                
                if magnitude_adaptive:
                    # Scale noise by weight magnitude
                    weight_magnitude = abs(param.data[unravel_idx].item())
                    effective_scale = noise_scale * weight_magnitude
                else:
                    effective_scale = noise_scale
                
                # Generate and apply noise
                noise = torch.randn(1, device=param.device) * effective_scale
                param.data[unravel_idx] += noise.item()
                total_perturbation += abs(noise.item())
            
            perturbation_stats[layer_name] = {
                "total_perturbation": total_perturbation,
                "num_weights_perturbed": len(weight_list),
                "mean_perturbation": total_perturbation / len(weight_list) if weight_list else 0.0
            }
        
        return perturbation_stats
    
    def magnitude_scaling_perturbation(
        self,
        model: torch.nn.Module,
        target_weights: Dict[str, List[Tuple]],
        scale_factor: float = 0.5
    ) -> Dict[str, float]:
        """Scale weight magnitudes by a factor."""
        
        perturbation_stats = {}
        
        for layer_name, weight_list in target_weights.items():
            if not weight_list:
                continue
            
            param = dict(model.named_parameters())[layer_name]
            
            weights_changed = 0
            total_change = 0.0
            
            for param_name, indices, score in weight_list:
                if isinstance(indices, tuple) and len(indices) == 1:
                    flat_idx = indices[0]
                    unravel_idx = np.unravel_index(flat_idx, param.shape)
                    
                    original_value = param.data[unravel_idx].item()
                    new_value = original_value * scale_factor
                    param.data[unravel_idx] = new_value
                    
                    total_change += abs(new_value - original_value)
                    weights_changed += 1
            
            perturbation_stats[layer_name] = {
                "weights_changed": weights_changed,
                "total_magnitude_change": total_change,
                "scale_factor": scale_factor
            }
        
        return perturbation_stats
    
    def quantization_perturbation(
        self,
        model: torch.nn.Module,
        target_weights: Dict[str, List[Tuple]],
        num_bits: int = 8
    ) -> Dict[str, float]:
        """Apply quantization to target weights."""
        
        perturbation_stats = {}
        
        for layer_name, weight_list in target_weights.items():
            if not weight_list:
                continue
            
            param = dict(model.named_parameters())[layer_name]
            
            # Determine quantization range
            all_values = []
            for param_name, indices, score in weight_list:
                if isinstance(indices, tuple) and len(indices) == 1:
                    flat_idx = indices[0]
                    unravel_idx = np.unravel_index(flat_idx, param.shape)
                    all_values.append(param.data[unravel_idx].item())
            
            if not all_values:
                continue
            
            min_val, max_val = min(all_values), max(all_values)
            num_levels = 2 ** num_bits
            step_size = (max_val - min_val) / (num_levels - 1)
            
            quantization_error = 0.0
            weights_quantized = 0
            
            for param_name, indices, score in weight_list:
                if isinstance(indices, tuple) and len(indices) == 1:
                    flat_idx = indices[0]
                    unravel_idx = np.unravel_index(flat_idx, param.shape)
                    
                    original_value = param.data[unravel_idx].item()
                    
                    # Quantize
                    quantized_level = round((original_value - min_val) / step_size)
                    quantized_level = max(0, min(quantized_level, num_levels - 1))
                    quantized_value = min_val + quantized_level * step_size
                    
                    param.data[unravel_idx] = quantized_value
                    quantization_error += abs(quantized_value - original_value)
                    weights_quantized += 1
            
            perturbation_stats[layer_name] = {
                "weights_quantized": weights_quantized,
                "total_quantization_error": quantization_error,
                "bits": num_bits,
                "quantization_range": (min_val, max_val)
            }
        
        return perturbation_stats
    
    def progressive_perturbation(
        self,
        model: torch.nn.Module,
        target_weights: Dict[str, List[Tuple]],
        perturbation_type: PerturbationType,
        severity_levels: List[float] = [0.1, 0.5, 1.0, 2.0],
        evaluation_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Apply perturbations with increasing severity levels."""
        
        results = []
        
        for severity in severity_levels:
            # Restore original state
            self.restore_original_state(model)
            
            # Apply perturbation with current severity
            if perturbation_type == PerturbationType.ADAPTIVE_NOISE:
                stats = self.adaptive_noise_perturbation(model, target_weights, severity)
            elif perturbation_type == PerturbationType.MAGNITUDE_SCALING:
                stats = self.magnitude_scaling_perturbation(model, target_weights, severity)
            elif perturbation_type == PerturbationType.QUANTIZATION:
                # For quantization, severity maps to number of bits
                bits = max(1, int(8 - severity * 2))
                stats = self.quantization_perturbation(model, target_weights, bits)
            else:
                logger.warning(f"Progressive perturbation not implemented for {perturbation_type}")
                continue
            
            # Evaluate if function provided
            eval_results = {}
            if evaluation_func:
                eval_results = evaluation_func(model)
            
            results.append({
                "severity_level": severity,
                "perturbation_stats": stats,
                "evaluation_results": eval_results
            })
        
        return results
    
    def structured_layer_perturbation(
        self,
        model: torch.nn.Module,
        layer_groups: Dict[str, List[str]],
        perturbation_type: PerturbationType,
        **perturbation_kwargs
    ) -> Dict[str, Any]:
        """Apply perturbations to structured groups of layers."""
        
        results = {}
        
        for group_name, layer_names in layer_groups.items():
            # Create weight list for this group
            group_weights = {}
            for layer_name in layer_names:
                if layer_name in dict(model.named_parameters()):
                    param = dict(model.named_parameters())[layer_name]
                    # Select all weights in the layer
                    flat_size = param.numel()
                    group_weights[layer_name] = [
                        (layer_name, (i,), 1.0) for i in range(flat_size)
                    ]
            
            # Apply perturbation to group
            if perturbation_type == PerturbationType.ADAPTIVE_NOISE:
                stats = self.adaptive_noise_perturbation(
                    model, group_weights, 
                    perturbation_kwargs.get('noise_scale', 0.1)
                )
            elif perturbation_type == PerturbationType.MAGNITUDE_SCALING:
                stats = self.magnitude_scaling_perturbation(
                    model, group_weights,
                    perturbation_kwargs.get('scale_factor', 0.5)
                )
            else:
                logger.warning(f"Structured perturbation not implemented for {perturbation_type}")
                continue
            
            results[group_name] = stats
        
        return results
    
    def generate_perturbation_report(
        self,
        model: torch.nn.Module,
        target_weights: Dict[str, List[Tuple]],
        perturbation_types: List[PerturbationType],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate comprehensive perturbation analysis report."""
        
        report = {
            "perturbation_summary": {},
            "comparative_analysis": {},
            "recommendations": []
        }
        
        # Test each perturbation type
        for pert_type in perturbation_types:
            # Restore original state
            self.restore_original_state(model)
            
            try:
                if pert_type == PerturbationType.ADAPTIVE_NOISE:
                    stats = self.adaptive_noise_perturbation(model, target_weights)
                elif pert_type == PerturbationType.MAGNITUDE_SCALING:
                    stats = self.magnitude_scaling_perturbation(model, target_weights)
                elif pert_type == PerturbationType.QUANTIZATION:
                    stats = self.quantization_perturbation(model, target_weights)
                else:
                    continue
                
                report["perturbation_summary"][pert_type.value] = stats
                
            except Exception as e:
                logger.warning(f"Failed to apply {pert_type}: {e}")
                report["perturbation_summary"][pert_type.value] = {"error": str(e)}
        
        # Restore original state
        self.restore_original_state(model)
        
        return report
