#!/usr/bin/env python3
"""
Critical Weight Analysis - Enhanced Phase 1 CLI Runner

Advanced sensitivity analysis pipeline for LLM weight prioritization research.
Supports gradient-based and non-gradient metrics, perturbation experiments,
control baselines, and comprehensive evaluation.

Usage Examples:
    # Basic gradient-based analysis
    python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100
    
    # Hutchinson diagonal estimator
    python phase1_runner.py --model meta-llama/Llama-3.1-8B --metric hutchinson_diag --topk 100 --mode global
    
    # Non-gradient analysis with controls
    python phase1_runner.py --model gpt2 --metric magnitude --topk 100 --controls random_k,bottom_k
    
    # Perturbation experiments
    python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100 --perturb sign_flip --perturb-scale 1.0
    
    # Multiple seeds for stability
    python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100 --seeds 0,1,2 --stability-check
    
    # Full workflow as described
    python phase1_runner.py \\
        --model meta-llama/Llama-3.1-8B \\
        --metric hutchinson_diag \\
        --topk 100 --mode global \\
        --perturb sign_flip --perturb-scale 1.0 \\
        --controls random_k,bottom_k \\
        --seeds 0,1,2 \\
        --out-dir outputs/llama31_8b_hutch_diag_k100
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
from src.sensitivity.metrics import compute_sensitivity, compute_non_gradient_sensitivity
from src.sensitivity.rank import rank_topk, create_all_controls, compute_jaccard_overlap
from src.sensitivity.perturb import apply_perturbation, compute_perturbation_effects, stability_analysis
from src.utils.manifest import create_manifest
from src.utils.visualize import save_all_plots

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Critical Weight Analysis - Enhanced Phase 1 Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt2',
        help='Model name or path (default: gpt2)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: cuda, cpu, or auto (default: auto)'
    )
    
    # Sensitivity analysis settings
    parser.add_argument(
        '--metric',
        type=str,
        default='grad_x_weight',
        choices=['grad_x_weight', 'grad_squared', 'hessian_diag', 'hutchinson_diag', 'magnitude', 'act_mag'],
        help='Sensitivity metric to compute (default: grad_x_weight)'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        help='Number of top weights to select (default: 100)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='per_layer',
        choices=['per_layer', 'global'],
        help='Ranking mode: per_layer or global (default: per_layer)'
    )
    
    # Perturbation settings
    parser.add_argument(
        '--perturb',
        type=str,
        default=None,
        choices=['zero', 'sign_flip', 'gauss_noise', 'bit_flip'],
        help='Perturbation method to apply (default: None)'
    )
    
    parser.add_argument(
        '--perturb-scale',
        type=float,
        default=1.0,
        help='Scale parameter for perturbation (default: 1.0)'
    )
    
    parser.add_argument(
        '--perturb-prob',
        type=float,
        default=0.1,
        help='Probability parameter for bit_flip perturbation (default: 0.1)'
    )
    
    # Control baselines
    parser.add_argument(
        '--controls',
        type=str,
        default=None,
        help='Comma-separated list of control methods: random_k,bottom_k (default: None)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seeds',
        type=str,
        default='42',
        help='Comma-separated list of random seeds (default: 42)'
    )
    
    parser.add_argument(
        '--stability-check',
        action='store_true',
        help='Perform stability analysis across seeds'
    )
    
    # Data settings
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to evaluation data file (default: use internal data)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of data samples to use (default: 100)'
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
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save visualization plots'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_evaluation_data(
    data_file: Optional[str] = None,
    max_samples: int = 100
) -> List[str]:
    """Load evaluation texts."""
    if data_file and Path(data_file).exists():
        logger.info(f"Loading evaluation data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default evaluation texts
        logger.info("Using default evaluation texts")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Large language models have revolutionized natural language processing in recent years.",
            "The transformer architecture introduced attention mechanisms that changed everything.",
            "Critical weight analysis helps identify the most important parameters in neural networks.",
            "Gradient-based sensitivity metrics provide insights into weight importance.",
            "Perturbation experiments validate the significance of identified critical weights.",
            "Reproducibility in machine learning requires careful tracking of experimental conditions.",
            "The future of artificial intelligence depends on understanding these fundamental mechanisms.",
            "Research methodologies must balance innovation with rigorous scientific practices."
        ] * 10  # Repeat to get more samples
    
    # Limit to max_samples
    texts = texts[:max_samples]
    logger.info(f"Using {len(texts)} evaluation texts")
    return texts


def run_sensitivity_analysis(
    model, tokenizer, texts: List[str], metric: str,
    layers: Optional[List[int]] = None,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """Run sensitivity analysis with timing."""
    logger.info(f"🔍 Computing {metric} sensitivity...")
    start_time = time.time()
    
    if metric in ['magnitude', 'act_mag']:
        sensitivity_dict = compute_non_gradient_sensitivity(
            model, tokenizer, texts, metric, layers, max_length
        )
    else:
        sensitivity_dict = compute_sensitivity(
            model, tokenizer, texts, metric, layers, max_length
        )
    
    elapsed_time = time.time() - start_time
    total_weights = sum(tensor.numel() for tensor in sensitivity_dict.values())
    
    logger.info(f"✅ Sensitivity analysis completed in {elapsed_time:.2f}s")
    logger.info(f"📊 Processed {total_weights:,} weights across {len(sensitivity_dict)} layers")
    
    return sensitivity_dict


def run_ranking_analysis(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
    mode: str = 'per_layer'
) -> Dict[str, List[Tuple[str, Tuple, float]]]:
    """Run weight ranking analysis."""
    logger.info(f"🏆 Computing Top-{k} ranking ({mode} mode)...")
    start_time = time.time()
    
    per_layer = (mode == 'per_layer')
    global_ranking = (mode == 'global')
    
    top_weights = rank_topk(
        sensitivity_dict, k, 
        per_layer=per_layer, 
        global_ranking=global_ranking
    )
    
    elapsed_time = time.time() - start_time
    total_selected = sum(len(weight_list) for weight_list in top_weights.values())
    
    logger.info(f"✅ Ranking completed in {elapsed_time:.2f}s")
    logger.info(f"🎯 Selected {total_selected} weights across {len(top_weights)} parameters")
    
    return top_weights


def run_control_analysis(
    sensitivity_dict: Dict[str, torch.Tensor],
    k: int,
    control_methods: List[str],
    seed: int = 42
) -> Dict[str, Dict[str, List[Tuple[str, Tuple, float]]]]:
    """Run control baseline analysis."""
    logger.info(f"🎲 Creating control baselines: {control_methods}")
    start_time = time.time()
    
    controls = create_all_controls(sensitivity_dict, k, control_methods, seed)
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Control analysis completed in {elapsed_time:.2f}s")
    
    return controls


def run_perturbation_experiments(
    model, tokenizer, eval_texts: List[str],
    target_weights: Dict[str, List[Tuple[str, Tuple, float]]],
    method: str,
    controls: Optional[Dict[str, Dict[str, List[Tuple[str, Tuple, float]]]]] = None,
    **perturb_kwargs
) -> Dict[str, Dict[str, float]]:
    """Run perturbation experiments."""
    logger.info(f"⚡ Running {method} perturbation experiments...")
    start_time = time.time()
    
    # Test main method
    results = compute_perturbation_effects(
        model, tokenizer, eval_texts, target_weights, 
        methods=[method], **perturb_kwargs
    )
    
    # Test controls if provided
    if controls:
        for control_name, control_weights in controls.items():
            logger.info(f"  Testing {control_name} control...")
            control_results = compute_perturbation_effects(
                model, tokenizer, eval_texts, control_weights,
                methods=[method], **perturb_kwargs
            )
            # Add control results with prefix
            for key, value in control_results.items():
                results[f"{control_name}_{key}"] = value
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Perturbation experiments completed in {elapsed_time:.2f}s")
    
    return results


def run_stability_analysis(
    model, tokenizer, texts: List[str], metric: str,
    k: int, seeds: List[int]
) -> Dict[str, float]:
    """Run stability analysis across seeds."""
    logger.info(f"🔬 Running stability analysis across {len(seeds)} seeds...")
    start_time = time.time()
    
    stability_results = stability_analysis(
        model, tokenizer, texts, metric, k, 
        num_seeds=len(seeds), num_batches=3
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stability analysis completed in {elapsed_time:.2f}s")
    
    return stability_results


def save_results(
    output_dir: Path,
    sensitivity_dict: Dict[str, torch.Tensor],
    top_weights: Dict[str, List[Tuple[str, Tuple, float]]],
    controls: Optional[Dict] = None,
    perturbation_results: Optional[Dict] = None,
    stability_results: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Dict[str, Path]:
    """Save all results to files."""
    logger.info(f"💾 Saving results to {output_dir}")
    
    saved_files = {}
    
    # Save sensitivity statistics
    stats_file = output_dir / "sensitivity_stats.json"
    sensitivity_stats = {}
    for name, tensor in sensitivity_dict.items():
        flat_sens = tensor.flatten().cpu().numpy()
        sensitivity_stats[name] = {
            "mean": float(np.mean(flat_sens)),
            "std": float(np.std(flat_sens)),
            "min": float(np.min(flat_sens)),
            "max": float(np.max(flat_sens)),
            "p95": float(np.percentile(flat_sens, 95)),
            "p99": float(np.percentile(flat_sens, 99)),
            "num_weights": len(flat_sens)
        }
    
    with open(stats_file, 'w') as f:
        json.dump(sensitivity_stats, f, indent=2)
    saved_files["sensitivity_stats"] = stats_file
    
    # Save top weights as CSV
    top_weights_data = []
    for param_name, weight_list in top_weights.items():
        for weight_param, indices, score in weight_list:
            top_weights_data.append({
                "parameter": weight_param,
                "indices": str(indices),
                "sensitivity_score": score,
                "layer_group": param_name
            })
    
    if top_weights_data:
        csv_file = output_dir / "top_weights.csv"
        pd.DataFrame(top_weights_data).to_csv(csv_file, index=False)
        saved_files["top_weights"] = csv_file
    
    # Save control results
    if controls:
        controls_file = output_dir / "control_baselines.json"
        # Convert to serializable format
        controls_serializable = {}
        for method, control_weights in controls.items():
            controls_serializable[method] = {}
            for param_name, weight_list in control_weights.items():
                controls_serializable[method][param_name] = [
                    {"parameter": wp, "indices": list(idx), "score": score}
                    for wp, idx, score in weight_list
                ]
        
        with open(controls_file, 'w') as f:
            json.dump(controls_serializable, f, indent=2)
        saved_files["controls"] = controls_file
    
    # Save perturbation results
    if perturbation_results:
        perturb_file = output_dir / "perturbation_results.json"
        with open(perturb_file, 'w') as f:
            json.dump(perturbation_results, f, indent=2)
        saved_files["perturbation_results"] = perturb_file
    
    # Save stability results
    if stability_results:
        stability_file = output_dir / "stability_results.json"
        with open(stability_file, 'w') as f:
            json.dump(stability_results, f, indent=2)
        saved_files["stability_results"] = stability_file
    
    # Save configuration
    if config:
        config_file = output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        saved_files["config"] = config_file
    
    logger.info(f"✅ Saved {len(saved_files)} result files")
    return saved_files


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("🚀 Starting Critical Weight Analysis - Enhanced Phase 1")
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    primary_seed = seeds[0]
    
    # Set random seeds
    torch.manual_seed(primary_seed)
    np.random.seed(primary_seed)
    
    # Parse control methods
    control_methods = []
    if args.controls:
        control_methods = [m.strip() for m in args.controls.split(',')]
    
    # Setup output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split('/')[-1].replace('-', '_')
        output_dir = Path("outputs") / f"{model_short}_{args.metric}_k{args.topk}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Create experiment manifest
    config = vars(args)
    manifest = create_manifest(
        experiment_name=f"{args.model}_{args.metric}_k{args.topk}",
        output_dir=output_dir,
        config=config,
        model_name=args.model,
        seeds=seeds
    )
    
    try:
        # Load model and tokenizer
        logger.info(f"🤖 Loading model: {args.model}")
        start_time = time.time()
        model, tokenizer = load_model(args.model, device=args.device)
        load_time = time.time() - start_time
        manifest.log_timing("model_loading", load_time)
        
        # Load evaluation data
        logger.info("📚 Loading evaluation data")
        eval_texts = load_evaluation_data(args.data_file, args.max_samples)
        manifest.log_data_info({
            "num_texts": len(eval_texts),
            "data_source": args.data_file or "default",
            "max_length": args.max_length
        })
        
        # Run sensitivity analysis
        sensitivity_dict = run_sensitivity_analysis(
            model, tokenizer, eval_texts, args.metric, max_length=args.max_length
        )
        manifest.log_timing("sensitivity_analysis", time.time() - start_time)
        
        # Run ranking analysis
        top_weights = run_ranking_analysis(sensitivity_dict, args.topk, args.mode)
        
        # Run control analysis if requested
        controls = None
        if control_methods:
            controls = run_control_analysis(
                sensitivity_dict, args.topk, control_methods, primary_seed
            )
        
        # Run perturbation experiments if requested
        perturbation_results = None
        if args.perturb:
            perturb_kwargs = {}
            if args.perturb == 'gauss_noise':
                perturb_kwargs['scale'] = args.perturb_scale
            elif args.perturb == 'bit_flip':
                perturb_kwargs['prob'] = args.perturb_prob
            
            perturbation_results = run_perturbation_experiments(
                model, tokenizer, eval_texts, top_weights, 
                args.perturb, controls, **perturb_kwargs
            )
        
        # Run stability analysis if requested
        stability_results = None
        if args.stability_check and len(seeds) > 1:
            stability_results = run_stability_analysis(
                model, tokenizer, eval_texts, args.metric, args.topk, seeds
            )
        
        # Save results
        saved_files = save_results(
            output_dir, sensitivity_dict, top_weights, controls,
            perturbation_results, stability_results, config
        )
        manifest.log_files({k: str(v) for k, v in saved_files.items()})
        
        # Generate plots if requested
        if args.save_plots:
            logger.info("📈 Generating visualization plots")
            plot_files = save_all_plots(
                sensitivity_dict, perturbation_results, stability_results,
                output_dir / "plots", f"{args.metric}_k{args.topk}_"
            )
            manifest.log_files({f"plot_{k}": str(v) for k, v in plot_files.items()})
        
        # Log summary results
        summary_results = {
            "total_weights_analyzed": sum(t.numel() for t in sensitivity_dict.values()),
            "top_k_selected": args.topk,
            "ranking_mode": args.mode,
            "metric_used": args.metric
        }
        
        if perturbation_results and 'baseline' in perturbation_results:
            baseline_ppl = perturbation_results['baseline']['perplexity']
            summary_results["baseline_perplexity"] = baseline_ppl
            
            if args.perturb in perturbation_results:
                perturb_ppl = perturbation_results[args.perturb]['perplexity']
                delta_ppl = perturb_ppl - baseline_ppl
                summary_results[f"{args.perturb}_perplexity"] = perturb_ppl
                summary_results[f"{args.perturb}_delta_perplexity"] = delta_ppl
        
        if stability_results:
            summary_results["seed_jaccard_stability"] = stability_results.get("seed_jaccard_mean", 0)
            summary_results["batch_jaccard_stability"] = stability_results.get("batch_jaccard_mean", 0)
        
        manifest.log_results(summary_results)
        
        # Print summary
        logger.info("🎉 Analysis completed successfully!")
        logger.info(f"📊 Results summary:")
        logger.info(f"  • Total weights analyzed: {summary_results['total_weights_analyzed']:,}")
        logger.info(f"  • Top-{args.topk} weights selected using {args.metric}")
        logger.info(f"  • Ranking mode: {args.mode}")
        
        if perturbation_results:
            logger.info(f"  • Perturbation method: {args.perturb}")
            if 'baseline' in perturbation_results and args.perturb in perturbation_results:
                baseline_ppl = perturbation_results['baseline']['perplexity']
                perturb_ppl = perturbation_results[args.perturb]['perplexity']
                delta_ppl = perturb_ppl - baseline_ppl
                logger.info(f"  • Baseline perplexity: {baseline_ppl:.3f}")
                logger.info(f"  • Perturbed perplexity: {perturb_ppl:.3f} (Δ = +{delta_ppl:.3f})")
        
        if stability_results:
            seed_stability = stability_results.get("seed_jaccard_mean", 0)
            batch_stability = stability_results.get("batch_jaccard_mean", 0)
            logger.info(f"  • Seed stability (Jaccard): {seed_stability:.3f}")
            logger.info(f"  • Batch stability (Jaccard): {batch_stability:.3f}")
        
        logger.info(f"📁 All results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        manifest.log_results({"error": str(e), "success": False})
        raise
    
    finally:
        # Finalize manifest
        manifest_path = manifest.finalize()
        logger.info(f"📋 Experiment manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
