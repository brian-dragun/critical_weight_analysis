#!/usr/bin/env python3
"""
Baseline script for Critical Weight Analysis
Runs a basic model evaluation and saves results.
"""
import argparse
import os
from src.models.loader import load_model
from src.eval.perplexity import compute_perplexity
import torch


def main():
    parser = argparse.ArgumentParser(description="Run baseline model evaluation.")
    parser.add_argument('--model', type=str, required=True, help='Model name or path (e.g. gpt2, meta-llama/Llama-3.1-8B)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--max-samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--out-dir', type=str, default='outputs/baseline', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device)

    data_path = "src/data/dev_small.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    texts = texts[:args.max_samples]

    print(f"Evaluating perplexity on {len(texts)} samples...")
    avg_perplexity = compute_perplexity(model, tokenizer, texts, device=args.device, batch_size=args.batch_size)

    out_file = os.path.join(args.out_dir, 'perplexity_results.txt')
    with open(out_file, 'w') as f:
        f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
