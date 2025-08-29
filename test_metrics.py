#!/usr/bin/env python3
"""
Quick test of the fixed metric functions
"""
import torch
import math

def compute_basic_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, T, V], labels: [B, T]
    Returns: loss, ppl, token_accuracy
    """
    # Reshape for cross-entropy: logits [B*T, V], labels [B*T]
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)
    labels_flat = labels.view(-1)
    
    # Compute cross-entropy loss (handles ignore_index=-100 automatically)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fn(logits_flat, labels_flat)
    
    # Compute token accuracy (excluding ignored tokens)
    with torch.no_grad():
        preds = torch.argmax(logits_flat, dim=-1)
        valid_mask = (labels_flat != -100)
        if valid_mask.sum() > 0:
            accuracy = (preds == labels_flat)[valid_mask].float().mean()
        else:
            accuracy = torch.tensor(0.0)
    
    # Compute perplexity
    perplexity = torch.exp(loss) if not torch.isnan(loss) else torch.tensor(float('inf'))
    
    return {
        "loss": float(loss.item()),
        "perplexity": float(perplexity.item()),
        "token_accuracy": float(accuracy.item())
    }

if __name__ == "__main__":
    # Create dummy data
    B, T, V = 2, 10, 1000  # batch=2, seq_len=10, vocab=1000
    
    # Random logits and labels
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    
    # Test the function
    metrics = compute_basic_metrics(logits, labels)
    
    print("✅ Metrics function test:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Perplexity: {metrics['perplexity']:.4f}")
    print(f"   Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    # Check for NaN
    has_nan = any(math.isnan(v) for v in metrics.values())
    print(f"   Contains NaN: {has_nan}")
    
    if not has_nan:
        print("🎉 SUCCESS: No more NaN values!")
    else:
        print("❌ FAILED: Still has NaN values")
