"""
Perplexity evaluation utilities for critical weight analysis.

Provides functions to compute perplexity on text datasets for model evaluation.
"""

import math
import logging
from typing import List, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity of a model on a list of texts.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of text strings to evaluate
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for evaluation (1 for individual texts)
        device: Target device (inferred from model if None)
        
    Returns:
        Perplexity score (exp(average cross-entropy loss))
        
    Examples:
        >>> texts = ["The quick brown fox", "jumps over the lazy dog"]
        >>> ppl = compute_perplexity(model, tokenizer, texts)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    logger.info(f"Computing perplexity on {len(texts)} texts")
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            if not text.strip():
                continue
                
            # Tokenize individual text
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,  # No padding for individual texts
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            # Skip very short sequences
            if input_ids.size(1) < 2:
                continue
            
            # For causal LM, labels are the same as input_ids
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Count valid tokens (non-padded)
            valid_tokens = attention_mask.sum().item()
            
            # Accumulate loss
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
            
            if (i + 1) % 50 == 0:
                current_ppl = math.exp(total_loss / max(total_tokens, 1))
                logger.debug(f"Processed {i+1}/{len(texts)} texts, current PPL: {current_ppl:.2f}")
    
    if total_tokens == 0:
        logger.warning("No valid tokens found in texts")
        return float('inf')
    
    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    logger.info(f"Perplexity: {perplexity:.4f} (avg loss: {avg_loss:.4f}, tokens: {total_tokens})")
    
    return perplexity


def compute_perplexity_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity using batched evaluation (may be less accurate due to padding).
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of text strings to evaluate
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for evaluation
        device: Target device (inferred from model if None)
        
    Returns:
        Perplexity score
        
    Note:
        This method uses padding which may affect accuracy. Use compute_perplexity()
        for more accurate results.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Filter empty texts
    valid_texts = [text for text in texts if text.strip()]
    logger.info(f"Computing batched perplexity on {len(valid_texts)} texts")
    
    with torch.no_grad():
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
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Create labels, masking padded tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padded tokens in loss
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Count valid tokens
            valid_tokens = (labels != -100).sum().item()
            
            # Accumulate loss
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    if total_tokens == 0:
        logger.warning("No valid tokens found in texts")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    logger.info(f"Batched perplexity: {perplexity:.4f} (avg loss: {avg_loss:.4f}, tokens: {total_tokens})")
    
    return perplexity


def evaluate_model_quality(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    metrics: List[str] = ["perplexity"],
) -> dict:
    """
    Evaluate model quality using multiple metrics.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        texts: List of text strings to evaluate
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric names to values
    """
    results = {}
    
    if "perplexity" in metrics:
        results["perplexity"] = compute_perplexity(model, tokenizer, texts)
    
    if "loss" in metrics:
        ppl = results.get("perplexity", compute_perplexity(model, tokenizer, texts))
        results["loss"] = math.log(ppl)
    
    # Add more metrics as needed
    # if "accuracy" in metrics:
    #     results["accuracy"] = compute_accuracy(model, tokenizer, texts)
    
    return results


def quick_perplexity_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_text: str = "The quick brown fox jumps over the lazy dog.",
) -> float:
    """
    Quick perplexity test on a single sentence for sanity checking.
    
    Args:
        model: The transformer model
        tokenizer: The corresponding tokenizer
        test_text: Test sentence
        
    Returns:
        Perplexity on the test sentence
    """
    return compute_perplexity(model, tokenizer, [test_text])
