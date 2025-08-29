"""
Model loader for Critical Weight Analysis
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str, device: str = 'cpu'):
    """
    Loads a HuggingFace model and tokenizer for causal language modeling.
    Args:
        model_name (str): Model name or path.
        device (str): 'cpu' or 'cuda'.
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def set_seed(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
