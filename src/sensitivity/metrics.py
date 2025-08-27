"""
Sensitivity metrics computation for critical weight analysis.

Implements various sensitivity measures including gradient-based and Hessian approximations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    # Create dummy classes for type hints
    class PreTrainedModel:
        pass
    class PreTrainedTokenizer:
        pass

logger = logging.getLogger(__name__)


def _compute_hutchinson_diagonal(
    model: PreTrainedModel,
    param: torch.Tensor,
    param_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_samples: int = 10
) -> torch.Tensor:
    """
    Compute Hutchinson diagonal estimator for Hessian diagonal.
    
    Uses random Rademacher vectors to estimate diagonal of Hessian matrix.
    H_ii ≈ E[z_i * (∇²L z)_i] where z ~ Rademacher distribution
    
    Args:
        model: The transformer model
        param: Parameter tensor for which to compute diagonal
        param_name: Name of the parameter
        input_ids: Input token IDs
        attention_mask: Attention mask
        num_samples: Number of random samples for estimation
        
    Returns:
        Estimated diagonal Hessian tensor
    """
    diagonal_estimate = torch.zeros_like(param)
    
    for _ in range(num_samples):
        # Generate Rademacher random vector (+1 or -1)
        z = torch.randint_like(param, 0, 2, dtype=param.dtype) * 2 - 1
        
        # Compute Hessian-vector product using automatic differentiation
        # First, compute gradient of loss
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Compute first-order gradients
        grads = torch.autograd.grad(loss, param, create_graph=True)[0]
        
        # Compute gradient-vector product
        gvp = torch.sum(grads * z)
        
        # Compute second-order derivative (Hessian-vector product)
        hvp = torch.autograd.grad(gvp, param, retain_graph=False)[0]
        
        # Accumulate diagonal estimate
        diagonal_estimate += z * hvp
    
    # Average over samples
    diagonal_estimate /= num_samples
    
    return torch.abs(diagonal_estimate)


def compute_non_gradient_sensitivity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    metric: str = "magnitude",
    layers: Optional[List[int]] = None,
    max_length: int = 512,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Compute non-gradient based sensitivity metrics.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of calibration texts (used for activation-based metrics)
        metric: Non-gradient metric ('magnitude', 'act_mag')
        layers: List of layer indices to analyze (None for all)
        max_length: Maximum sequence length
        batch_size: Batch size for activation computation
        
    Returns:
        Dictionary mapping parameter names to sensitivity tensors
    """
    logger.info(f"Computing {metric} sensitivity (non-gradient)")
    
    model.eval()
    sensitivity_dict = {}
    
    if metric == "magnitude":
        # Simple weight magnitude
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
                
            # Filter by layers if specified
            if layers is not None:
                layer_match = False
                for layer_idx in layers:
                    if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                        layer_match = True
                        break
                if not layer_match:
                    continue
            
            sensitivity = torch.abs(param.detach())
            sensitivity_dict[name] = sensitivity.cpu()
            
    elif metric == "act_mag":
        # Activation-weighted magnitude
        activation_dict = _compute_layer_activations(model, tokenizer, texts, layers, max_length, batch_size)
        
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
                
            # Filter by layers if specified
            if layers is not None:
                layer_match = False
                for layer_idx in layers:
                    if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                        layer_match = True
                        break
                if not layer_match:
                    continue
            
            # Get corresponding activation if available
            base_name = name.split('.weight')[0] if '.weight' in name else name
            if base_name in activation_dict:
                # Weight magnitude weighted by activation magnitude
                activations = activation_dict[base_name]
                if activations.shape[-1] == param.shape[-1]:  # Match last dimension
                    activation_weights = torch.mean(torch.abs(activations), dim=0)
                    if len(activation_weights.shape) == 1 and len(param.shape) == 2:
                        # Broadcast to weight matrix shape
                        activation_weights = activation_weights.unsqueeze(0).expand(param.shape[0], -1)
                    sensitivity = torch.abs(param.detach()) * activation_weights.to(param.device)
                else:
                    # Fallback to simple magnitude
                    sensitivity = torch.abs(param.detach())
            else:
                # Fallback to simple magnitude
                sensitivity = torch.abs(param.detach())
                
            sensitivity_dict[name] = sensitivity.cpu()
    else:
        raise ValueError(f"Unknown non-gradient metric: {metric}")
    
    logger.info(f"Computed non-gradient sensitivity for {len(sensitivity_dict)} parameters")
    return sensitivity_dict


def _compute_layer_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layers: Optional[List[int]],
    max_length: int,
    batch_size: int
) -> Dict[str, torch.Tensor]:
    """Compute layer activations for activation-weighted metrics."""
    activation_dict = {}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activation_dict[name] = output.detach().cpu()
        return hook
    
    # Register hooks for specified layers
    for name, module in model.named_modules():
        if layers is not None:
            layer_match = False
            for layer_idx in layers:
                if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                    layer_match = True
                    break
            if not layer_match:
                continue
        
        if hasattr(module, 'weight') and module.weight is not None:
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    # Run forward pass to capture activations
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoding = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            
            input_ids = encoding["input_ids"].to(model.device)
            attention_mask = encoding["attention_mask"].to(model.device)
            
            model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Only need one batch for activation estimation
            break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_dict


def get_model_layers(model: PreTrainedModel) -> List[str]:
    """
    Get all layer names with trainable parameters from a model.
    
    Args:
        model: The transformer model
        
    Returns:
        List of layer names that have weights
        
    Examples:
        >>> layers = get_model_layers(model)
        >>> print(f"Found {len(layers)} layers with parameters")
    """
    layer_names = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.requires_grad:
                layer_names.append(name)
    
    return layer_names


def compute_sensitivity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    metric: str = "grad_x_weight",
    layers: Optional[List[int]] = None,
    max_length: int = 512,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Compute weight sensitivity for specified layers using various metrics.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of calibration texts for gradient computation
        metric: Sensitivity metric ('grad_x_weight', 'grad_squared', 'hessian_diag', 
                'hutchinson_diag', 'magnitude', 'act_mag')
        layers: List of layer indices to analyze (None for all)
        max_length: Maximum sequence length
        batch_size: Batch size for gradient computation
        
    Returns:
        Dictionary mapping parameter names to sensitivity tensors
        
    Examples:
        >>> texts = ["The quick brown fox", "jumps over the lazy dog"]
        >>> sensitivity = compute_sensitivity(model, tokenizer, texts, "grad_x_weight", [2, 4])
        >>> sensitivity = compute_sensitivity(model, tokenizer, texts, "magnitude")
        >>> sensitivity = compute_sensitivity(model, tokenizer, texts, "hutchinson_diag")
    """
    # Handle non-gradient metrics separately
    if metric in ["magnitude", "act_mag"]:
        return compute_non_gradient_sensitivity(model, tokenizer, texts, metric, layers, max_length, batch_size)
    
    logger.info(f"Computing {metric} sensitivity on {len(texts)} texts")
    
    model.train()  # Enable gradients
    
    # Filter texts and prepare for batching
    valid_texts = [text for text in texts if text.strip()]
    if not valid_texts:
        raise ValueError("No valid texts provided")
    
    # Initialize gradient accumulation
    model.zero_grad()
    
    # Store batch data for Hutchinson estimator
    stored_batch_data = []
    
    # Accumulate gradients over multiple batches
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i + batch_size]
        
        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        
        # Store for Hutchinson estimator
        if metric == "hutchinson_diag":
            stored_batch_data.append((input_ids, attention_mask))
        
        # Create labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padded tokens
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass to accumulate gradients
        loss = outputs.loss
        loss.backward()
        
        total_loss += loss.item()
        num_batches += 1
        
        logger.debug(f"Processed batch {num_batches}, loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"Average loss for sensitivity computation: {avg_loss:.4f}")
    
    # Compute sensitivity metrics
    sensitivity_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is None or param.dim() < 2:
            continue
            
        # Filter by layers if specified
        if layers is not None:
            layer_match = False
            for layer_idx in layers:
                if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                    layer_match = True
                    break
            if not layer_match:
                continue
        
        # Compute sensitivity based on metric
        if metric == "grad_x_weight":
            sensitivity = torch.abs(param.grad.detach() * param.detach())
        elif metric == "grad_squared":
            sensitivity = param.grad.detach() ** 2
        elif metric == "hessian_diag":
            # Simple diagonal Hessian approximation using gradient squared
            sensitivity = param.grad.detach() ** 2
        elif metric == "hutchinson_diag":
            # Hutchinson diagonal estimator for curvature
            if stored_batch_data:
                input_ids, attention_mask = stored_batch_data[0]  # Use first batch
                sensitivity = _compute_hutchinson_diagonal(model, param, name, input_ids, attention_mask, num_samples=10)
            else:
                # Fallback to grad squared
                sensitivity = param.grad.detach() ** 2
        else:
            raise ValueError(f"Unknown sensitivity metric: {metric}")
        
        sensitivity_dict[name] = sensitivity.cpu()
        
        logger.debug(f"Computed {metric} for {name}: shape {sensitivity.shape}, "
                    f"mean: {sensitivity.mean().item():.6f}, "
                    f"max: {sensitivity.max().item():.6f}")
    
    model.eval()  # Return to evaluation mode
    logger.info(f"Computed sensitivity for {len(sensitivity_dict)} parameters")
    
    return sensitivity_dict


def compute_layer_statistics(sensitivity_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """
    Compute statistical summaries for each layer's sensitivity.
    
    Args:
        sensitivity_dict: Dictionary of parameter name to sensitivity tensor
        
    Returns:
        Dictionary mapping parameter names to statistics
    """
    stats = {}
    
    for name, sens_tensor in sensitivity_dict.items():
        flat_sens = sens_tensor.flatten()
        
        stats[name] = {
            "mean": float(flat_sens.mean()),
            "std": float(flat_sens.std()),
            "min": float(flat_sens.min()),
            "max": float(flat_sens.max()),
            "median": float(flat_sens.median()),
            "q25": float(flat_sens.quantile(0.25)),
            "q75": float(flat_sens.quantile(0.75)),
            "q95": float(flat_sens.quantile(0.95)),
            "q99": float(flat_sens.quantile(0.99)),
            "num_weights": int(flat_sens.numel()),
            "shape": list(sens_tensor.shape),
        }
    
    return stats


def get_sensitivity_percentiles(
    sensitivity_dict: Dict[str, torch.Tensor],
    percentiles: List[float] = [50, 75, 90, 95, 99],
) -> Dict[str, Dict[float, float]]:
    """
    Get sensitivity percentiles for each parameter.
    
    Args:
        sensitivity_dict: Dictionary of parameter name to sensitivity tensor
        percentiles: List of percentiles to compute
        
    Returns:
        Dictionary mapping parameter names to percentile values
    """
    percentile_dict = {}
    
    for name, sens_tensor in sensitivity_dict.items():
        flat_sens = sens_tensor.flatten()
        
        percentile_dict[name] = {}
        for p in percentiles:
            percentile_dict[name][p] = float(flat_sens.quantile(p / 100.0))
    
    return percentile_dict


def compare_sensitivity_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    metrics: List[str] = ["grad_x_weight", "grad_squared"],
    layers: Optional[List[int]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compare multiple sensitivity metrics on the same model and data.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of calibration texts
        metrics: List of sensitivity metrics to compare
        layers: List of layer indices to analyze
        
    Returns:
        Dictionary mapping metric names to sensitivity dictionaries
    """
    results = {}
    
    for metric in metrics:
        logger.info(f"Computing sensitivity metric: {metric}")
        results[metric] = compute_sensitivity(
            model, tokenizer, texts, metric, layers
        )
    
    return results


def validate_sensitivity_computation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_text: str = "The quick brown fox jumps over the lazy dog.",
) -> bool:
    """
    Validate that sensitivity computation is working correctly.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        test_text: Simple test text
        
    Returns:
        True if validation passes
    """
    try:
        # Test basic sensitivity computation
        sensitivity = compute_sensitivity(
            model, tokenizer, [test_text], "grad_x_weight", layers=[0]
        )
        
        if not sensitivity:
            logger.error("No sensitivity computed")
            return False
        
        # Check that sensitivities are non-negative and finite
        for name, sens_tensor in sensitivity.items():
            if torch.any(torch.isnan(sens_tensor)) or torch.any(torch.isinf(sens_tensor)):
                logger.error(f"Invalid sensitivity values in {name}")
                return False
            
            if torch.any(sens_tensor < 0):
                logger.error(f"Negative sensitivity values in {name}")
                return False
        
        logger.info("Sensitivity computation validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Sensitivity computation validation failed: {e}")
        return False
