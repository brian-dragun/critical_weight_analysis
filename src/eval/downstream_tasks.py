"""
Downstream task evaluation for critical weight analysis.

Evaluates model performance on specific tasks after weight perturbation
to measure functional impact beyond perplexity.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)

def evaluate_hellaswag(model, tokenizer, num_samples: int = 100) -> Dict[str, float]:
    """Evaluate on HellaSwag commonsense reasoning task."""
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in dataset:
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])
        
        # Compute likelihood for each ending
        scores = []
        for ending in endings:
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Compute average log likelihood
                log_probs = torch.log_softmax(logits, dim=-1)
                score = log_probs.mean().item()
                scores.append(score)
        
        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {"hellaswag_accuracy": accuracy, "total_samples": total}

def evaluate_lambada(model, tokenizer, num_samples: int = 100) -> Dict[str, float]:
    """Evaluate on LAMBADA last word prediction task."""
    dataset = load_dataset("lambada", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in dataset:
        text = example["text"]
        words = text.split()
        context = " ".join(words[:-1])
        target_word = words[-1].lower()
        
        inputs = tokenizer(context, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get top prediction
            predicted_token_id = torch.argmax(logits).item()
            predicted_word = tokenizer.decode([predicted_token_id]).strip().lower()
            
            if predicted_word == target_word:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {"lambada_accuracy": accuracy, "total_samples": total}

def evaluate_all_tasks(model, tokenizer, num_samples: int = 100) -> Dict[str, float]:
    """Evaluate on all available downstream tasks."""
    results = {}
    
    try:
        results.update(evaluate_hellaswag(model, tokenizer, num_samples))
    except Exception as e:
        logger.warning(f"HellaSwag evaluation failed: {e}")
    
    try:
        results.update(evaluate_lambada(model, tokenizer, num_samples))
    except Exception as e:
        logger.warning(f"LAMBADA evaluation failed: {e}")
    
    return results
