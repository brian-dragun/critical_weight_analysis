#!/usr/bin/env python3
"""
Perturbation Evaluation Script for Critical Weight Analysis

This script performs targeted perturbations on specific weights and evaluates
the impact on model performance. Supports various perturbation types including
zeroing, scaling, noise injection, and bit-flipping.

Usage Examples:
    # Zero a single superweight
    python scripts/run_perturb_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --perturbation zero \\
        --target model.embed_tokens.weight:128000,3303 \\
        --out-dir outputs/p3_zero_single
    
    # Scale multiple weights
    python scripts/run_perturb_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --perturbation scale --scale 1.2 \\
        --target_list files/top5_superweights.csv \\
        --out-dir outputs/p3_scale1p2_top5
    
    # Add Gaussian noise
    python scripts/run_perturb_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --perturbation noise --noise_sigma 0.05 \\
        --target_list files/top5_superweights.csv \\
        --out-dir outputs/p3_noise005_top5
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm

# Project imports
from src.models.loader import load_model
from src.eval.perplexity import compute_perplexity
from src.sensitivity.perturb import WeightPerturber
from src.utils.manifest import create_manifest

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perturbation Evaluation for Critical Weight Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model name or path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: cuda, cpu, or auto (default: cuda)'
    )
    
    # Perturbation settings
    parser.add_argument(
        '--perturbation',
        type=str,
        required=True,
        choices=['zero', 'scale', 'noise', 'bitflip'],
        help='Perturbation method to apply'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.2,
        help='Scaling factor for scale perturbation (default: 1.2)'
    )
    
    parser.add_argument(
        '--noise_sigma',
        type=float,
        default=0.05,
        help='Sigma for Gaussian noise perturbation (default: 0.05)'
    )
    
    parser.add_argument(
        '--bits',
        type=int,
        default=1,
        help='Number of bits to flip for bitflip perturbation (default: 1)'
    )
    
    # Target specification
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Single target weight specification: param_name:indices (e.g., model.embed_tokens.weight:128000,3303)'
    )
    
    parser.add_argument(
        '--target_list',
        type=str,
        default=None,
        help='Path to CSV file containing target weights'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of evaluation samples (default: 100)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    # Output settings
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Output directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def parse_target_weight(target_spec: str) -> Tuple[str, Tuple]:
    """
    Parse target weight specification.
    
    Args:
        target_spec: String in format "param_name:indices" 
                    e.g., "model.embed_tokens.weight:128000,3303"
    
    Returns:
        Tuple of (parameter_name, indices_tuple)
    """
    if ':' not in target_spec:
        raise ValueError(f"Invalid target specification: {target_spec}")
    
    param_name, indices_str = target_spec.split(':', 1)
    
    # Parse indices - handle both single index and tuple
    if ',' in indices_str:
        indices = tuple(int(x.strip()) for x in indices_str.split(','))
    else:
        indices = (int(indices_str.strip()),)
    
    return param_name, indices


def load_target_weights(csv_path: str) -> List[Tuple[str, Tuple]]:
    """
    Load target weights from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: parameter, indices
    
    Returns:
        List of (parameter_name, indices_tuple) tuples
    """
    df = pd.read_csv(csv_path)
    targets = []
    
    for _, row in df.iterrows():
        param_name = row['parameter']
        
        # Handle indices - may be string representation of tuple
        indices_str = str(row['indices'])
        if indices_str.startswith('(') and indices_str.endswith(')'):
            # Remove parentheses and parse
            indices_str = indices_str[1:-1]
        
        if ',' in indices_str:
            indices = tuple(int(x.strip()) for x in indices_str.split(','))
        else:
            indices = (int(indices_str.strip()),)
        
        targets.append((param_name, indices))
    
    return targets


def load_evaluation_data(max_samples: int = 100) -> List[str]:
    """Load evaluation texts."""
    # Default evaluation texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Artificial intelligence is transforming the way we work and live.",
        "Deep learning models require large amounts of data to train effectively.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Robotics combines mechanical engineering, electrical engineering, and computer science.",
        "Data science involves extracting knowledge and insights from data.",
        "Cloud computing provides on-demand access to computing resources.",
        "Cybersecurity protects computer systems from digital attacks."
    ] * (max_samples // 10 + 1)
    
    return texts[:max_samples]


def evaluate_model_performance(
    model, 
    tokenizer, 
    eval_texts: List[str], 
    max_length: int = 512
) -> Dict[str, float]:
    """
    Evaluate model performance on evaluation texts.
    
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Evaluating model performance on {len(eval_texts)} texts")
    
    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, eval_texts, max_length=max_length)
    
    # TODO: Add more metrics as needed (accuracy, loss, etc.)
    
    return {
        'perplexity': perplexity,
        'log_perplexity': np.log(perplexity)
    }


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("🚀 Starting Perturbation Evaluation")
    logger.info(f"📁 Output directory: {args.out_dir}")
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"🤖 Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device)
    
    # Load evaluation data
    logger.info("📚 Loading evaluation data")
    eval_texts = load_evaluation_data(args.max_samples)
    logger.info(f"Using {len(eval_texts)} evaluation texts")
    
    # Parse target weights
    if args.target:
        targets = [parse_target_weight(args.target)]
    elif args.target_list:
        targets = load_target_weights(args.target_list)
    else:
        raise ValueError("Must specify either --target or --target_list")
    
    logger.info(f"🎯 Targeting {len(targets)} weight(s) for perturbation")
    
    # Baseline evaluation
    logger.info("📊 Computing baseline performance")
    baseline_metrics = evaluate_model_performance(
        model, tokenizer, eval_texts, args.max_length
    )
    logger.info(f"Baseline perplexity: {baseline_metrics['perplexity']:.4f}")
    
    # Initialize weight perturber
    perturber = WeightPerturber(model)
    
    # Apply perturbations and evaluate
    results = {
        'baseline': baseline_metrics,
        'perturbations': [],
        'config': vars(args),
        'targets': [{'param': param, 'indices': indices} for param, indices in targets]
    }
    
    for i, (param_name, indices) in enumerate(targets):
        logger.info(f"🔧 Applying {args.perturbation} perturbation to {param_name}{indices} ({i+1}/{len(targets)})")
        
        # Apply perturbation
        if args.perturbation == 'zero':
            perturber.perturb_weights([(param_name, indices)], 'zero')
        elif args.perturbation == 'scale':
            perturber.perturb_weights([(param_name, indices)], 'scale', scale_factor=args.scale)
        elif args.perturbation == 'noise':
            perturber.perturb_weights([(param_name, indices)], 'gauss_noise', noise_scale=args.noise_sigma)
        elif args.perturbation == 'bitflip':
            perturber.perturb_weights([(param_name, indices)], 'bit_flip', bit_prob=args.bits/32.0)  # Approximate
        
        # Evaluate perturbed model
        perturbed_metrics = evaluate_model_performance(
            model, tokenizer, eval_texts, args.max_length
        )
        
        # Calculate deltas
        delta_ppl = perturbed_metrics['perplexity'] - baseline_metrics['perplexity']
        delta_ppl_pct = (delta_ppl / baseline_metrics['perplexity']) * 100
        
        logger.info(f"Perturbed perplexity: {perturbed_metrics['perplexity']:.4f} (Δ={delta_ppl:+.4f}, {delta_ppl_pct:+.2f}%)")
        
        # Store results
        results['perturbations'].append({
            'target': {'param': param_name, 'indices': indices},
            'method': args.perturbation,
            'metrics': perturbed_metrics,
            'deltas': {
                'perplexity': delta_ppl,
                'perplexity_pct': delta_ppl_pct
            }
        })
        
        # Restore weights for next iteration
        perturber.restore_weights()
    
    # Save results
    results_file = output_dir / "perturbation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_file}")
    
    # Create manifest
    manifest = create_manifest(
        experiment_name=f"{args.model}_{args.perturbation}_perturbation",
        config=vars(args),
        output_dir=str(output_dir)
    )
    
    manifest_file = output_dir / "experiment_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"📋 Experiment manifest saved to {manifest_file}")
    logger.info("🎉 Perturbation evaluation completed successfully!")


if __name__ == "__main__":
    main()
