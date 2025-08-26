#!/usr/bin/env python3
"""
Critical Weight Analysis - Phase 1 CLI Runner

Automated sensitivity analysis pipeline for LLM weight prioritization research.
Implements gradient-based sensitivity metrics and perturbation experiments.

Usage:
    python phase1_runner.py --model gpt2 --metric grad_weight --topk 100 --output results/
          metric_rankings = {}
        total_weights = sum(tensor.numel() for tensor in layer_sensitivities.values())
        print(f"  📊 Processing {total_weights:,} weights across {len(layer_sensitivities)} layers")
        
        for k in topk_values:
            print(f"  🔄 Computing Top-{k} ranking...")
            start_time = time.time()
            
            top_weights = rank_weights_topk(layer_sensitivities, k=k)
            elapsed_time = time.time() - start_timehon phase1_runner.py --model EleutherAI/pythia-410m --metric grad_squared --topk 500 --eval-limit 50
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

# Project imports
from src.models.loader import load_model
from src.eval.perplexity import compute_perplexity
from src.sensitivity.metrics import compute_sensitivity, get_model_layers
from src.sensitivity.rank import rank_topk


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Critical Weight Analysis - Phase 1 Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with GPT-2
  python phase1_runner.py --model gpt2 --metric grad_weight --topk 100

  # Full analysis with Pythia model
  python phase1_runner.py --model EleutherAI/pythia-410m --metric grad_squared --topk 500 --eval-limit 100

  # Multiple metrics and top-K values
  python phase1_runner.py --model gpt2 --metric grad_weight grad_squared --topk 50 100 500 --perturbation-ratios 0.1 0.5 1.0

  # Custom output directory and data
  python phase1_runner.py --model gpt2 --metric grad_weight --topk 100 --output custom_results/ --data-file custom_eval.txt
        """
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt2',
        help='Model name or path (default: gpt2). Examples: gpt2, EleutherAI/pythia-410m'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: cuda, cpu, or auto (default: auto)'
    )
    
    # Analysis settings
    parser.add_argument(
        '--metric',
        type=str,
        nargs='+',
        default=['grad_x_weight'],
        choices=['grad_x_weight', 'grad_squared', 'hessian_diag'],
        help='Sensitivity metrics to compute (default: grad_x_weight)'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        nargs='+',
        default=[100],
        help='Top-K values for weight ranking (default: 100)'
    )
    
    parser.add_argument(
        '--perturbation-ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help='Ratios of top-K weights to mask (default: 0.1 0.25 0.5 0.75 1.0)'
    )
    
    # Data settings
    parser.add_argument(
        '--data-file',
        type=str,
        default='src/data/dev_small.txt',
        help='Path to evaluation text file (default: src/data/dev_small.txt)'
    )
    
    parser.add_argument(
        '--eval-limit',
        type=int,
        default=100,
        help='Maximum number of evaluation texts to use (default: 100)'
    )
    
    # Output settings
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='Custom experiment ID (default: timestamp)'
    )
    
    # Performance settings
    parser.add_argument(
        '--no-perturbation',
        action='store_true',
        help='Skip perturbation analysis (only compute sensitivity)'
    )
    
    parser.add_argument(
        '--fast-perturbation',
        action='store_true',
        help='Use subset of texts for perturbation analysis (faster)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> Tuple[str, Path]:
    """Setup device and output directory."""
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    # Output directory setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Experiment ID
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    session_dir = output_dir / f"critical_analysis_{experiment_id}"
    session_dir.mkdir(exist_ok=True)
    
    return device, session_dir


def load_evaluation_data(args: argparse.Namespace) -> List[str]:
    """Load evaluation texts from file."""
    data_file = Path(args.data_file)
    
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print(f"⚠️ Data file {data_file} not found, using fallback texts")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful optimization and tuning.",
            "Natural language processing has advanced significantly in recent years.",
            "Deep neural networks learn complex patterns from large datasets.",
            "Gradient-based optimization methods are fundamental to model training.",
            "Transformer architectures have revolutionized language understanding.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Large language models demonstrate emergent capabilities at scale.",
        ] * 15  # Repeat to get more samples
    
    # Limit texts
    if args.eval_limit > 0:
        texts = texts[:args.eval_limit]
    
    return texts


def print_header(args: argparse.Namespace, device: str, session_dir: Path):
    """Print analysis header."""
    print("🔬 Critical Weight Analysis - Phase 1 Runner")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model: {args.model}")
    print(f"🔧 Device: {device}")
    print(f"📊 Metrics: {', '.join(args.metric)}")
    print(f"🏆 Top-K values: {args.topk}")
    print(f"🎯 Perturbation ratios: {args.perturbation_ratios}")
    print(f"📚 Evaluation limit: {args.eval_limit}")
    print(f"💾 Output directory: {session_dir}")
    print("=" * 60)


def rank_weights_topk(sensitivity_results: Dict[str, torch.Tensor], k: int) -> List[Dict]:
    """
    Fast top-K ranking using torch operations for efficiency.
    
    Args:
        sensitivity_results: Dict mapping layer names to sensitivity tensors
        k: Number of top weights to select
        
    Returns:
        List of dicts with 'layer', 'indices', 'sensitivity' keys
    """
    import heapq
    
    print(f"    🔄 Finding top-{k} weights from {len(sensitivity_results)} layers...")
    
    # Use a min-heap to efficiently track top-k weights
    top_weights_heap = []
    total_weights = sum(tensor.numel() for tensor in sensitivity_results.values())
    
    with tqdm(total=len(sensitivity_results), desc="    Processing layers", leave=False, ascii=True) as pbar:
        for layer_name, sens_tensor in sensitivity_results.items():
            # Get top weights from this layer first (much faster than processing all)
            flat_sens = sens_tensor.flatten()
            
            # Get top min(k*2, total_in_layer) to have good candidates from each layer
            layer_k = min(k * 2, flat_sens.numel())
            if layer_k > 0:
                layer_top_values, layer_top_indices = torch.topk(flat_sens, k=layer_k, largest=True)
                
                # Convert to the format we need
                for value, flat_idx in zip(layer_top_values, layer_top_indices):
                    # Convert flat index to multi-dimensional
                    multi_idx = torch.unravel_index(flat_idx, sens_tensor.shape)
                    indices = tuple(idx.item() for idx in multi_idx)
                    
                    weight_info = {
                        'layer': layer_name,
                        'indices': indices,
                        'sensitivity': float(value)
                    }
                    
                    # Use heap to maintain top-k efficiently
                    if len(top_weights_heap) < k:
                        heapq.heappush(top_weights_heap, (float(value), weight_info))
                    elif float(value) > top_weights_heap[0][0]:
                        heapq.heapreplace(top_weights_heap, (float(value), weight_info))
            
            pbar.update(1)
    
    # Extract the weight info and sort by sensitivity (highest first)
    top_k_weights = [weight_info for _, weight_info in top_weights_heap]
    top_k_weights.sort(key=lambda x: x['sensitivity'], reverse=True)
    
    print(f"    ✅ Selected top-{len(top_k_weights)} weights")
    return top_k_weights


def run_sensitivity_analysis(
    model, tokenizer, eval_texts: List[str], 
    metrics: List[str], device: str, 
    verbose: bool = False
) -> Dict[str, Dict]:
    """Run sensitivity analysis for all specified metrics."""
    print("\n🧮 Computing Sensitivity Metrics")
    print("-" * 40)
    
    sensitivity_results = {}
    
    for metric in metrics:
        print(f"🔄 Computing {metric} sensitivity...")
        
        start_time = time.time()
        layer_sensitivities = compute_sensitivity(
            model, tokenizer, eval_texts, 
            metric=metric, batch_size=1
        )
        elapsed_time = time.time() - start_time
        
        sensitivity_results[metric] = layer_sensitivities
        
        # Summary statistics
        total_weights = sum(len(sens) for sens in layer_sensitivities.values())
        all_sensitivities = torch.cat([sens.flatten() for sens in layer_sensitivities.values()])
        
        print(f"  ✅ Layers: {len(layer_sensitivities)}")
        print(f"  📊 Weights: {total_weights:,}")
        print(f"  📈 Mean: {all_sensitivities.mean().item():.2e}")
        print(f"  📉 Std: {all_sensitivities.std().item():.2e}")
        print(f"  🔝 Max: {all_sensitivities.max().item():.2e}")
        print(f"  ⏱️ Time: {elapsed_time:.1f}s")
        
        if verbose:
            print(f"  📋 Layer details:")
            for layer_name, sens in list(layer_sensitivities.items())[:3]:
                print(f"    {layer_name}: {sens.shape}, mean={sens.mean().item():.2e}")
            if len(layer_sensitivities) > 3:
                print(f"    ... and {len(layer_sensitivities) - 3} more layers")
    
    return sensitivity_results


def run_weight_ranking(
    sensitivity_results: Dict[str, Dict], 
    topk_values: List[int],
    verbose: bool = False
) -> Dict[str, Dict]:
    """Rank weights for all metrics and top-K values."""
    print("\n🏆 Ranking Critical Weights")
    print("-" * 40)
    
    ranking_results = {}
    
    for metric in sensitivity_results:
        print(f"🔄 Ranking weights for {metric}...")
        layer_sensitivities = sensitivity_results[metric]
        
        metric_rankings = {}
        for k in topk_values:
            start_time = time.time()
            top_weights = rank_weights_topk(layer_sensitivities, k=k)
            elapsed_time = time.time() - start_time
            
            metric_rankings[k] = top_weights
            
            # Analyze layer distribution
            layer_counts = {}
            for weight_info in top_weights:
                layer = weight_info['layer']
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            print(f"  Top-{k}: {len(top_weights)} weights across {len(layer_counts)} layers ({elapsed_time:.1f}s)")
            
            if verbose and layer_counts:
                top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                layer_str = ', '.join([f'{layer.split(".")[-1]}({count})' for layer, count in top_layers])
                print(f"    Most critical: {layer_str}")
        
        ranking_results[metric] = metric_rankings
    
    return ranking_results


def run_perturbation_analysis(
    model, tokenizer, eval_texts: List[str],
    ranking_results: Dict[str, Dict], 
    perturbation_ratios: List[float],
    baseline_ppl: float,
    fast_mode: bool = False,
    verbose: bool = False
) -> List[Dict]:
    """Run perturbation experiments."""
    print("\n🎯 Perturbation Analysis")
    print("-" * 40)
    
    perturbation_results = []
    
    # Use subset for speed in fast mode
    if fast_mode:
        perturb_texts = eval_texts[:20]
        print(f"⚡ Fast mode: using {len(perturb_texts)} texts for perturbation")
    else:
        perturb_texts = eval_texts
    
    total_experiments = sum(
        len(ranking_results[metric]) * len(perturbation_ratios) 
        for metric in ranking_results
    )
    
    print(f"📊 Running {total_experiments} perturbation experiments...")
    
    with tqdm(total=total_experiments, desc="Perturbation") as pbar:
        for metric in ranking_results:
            for k, top_weights in ranking_results[metric].items():
                for ratio in perturbation_ratios:
                    # Calculate weights to mask
                    num_to_mask = int(len(top_weights) * ratio)
                    if num_to_mask == 0:
                        pbar.update(1)
                        continue
                    
                    weights_to_mask = top_weights[:num_to_mask]
                    
                    try:
                        # Store original values
                        original_values = {}
                        
                        # Apply masking
                        for weight_info in weights_to_mask:
                            layer_name = weight_info['layer']
                            indices = weight_info['indices']
                            
                            # Get the layer module - handle nested module access
                            try:
                                layer = model
                                for part in layer_name.split('.'):
                                    layer = getattr(layer, part)
                                
                                if hasattr(layer, 'weight') and layer.weight is not None:
                                    key = (layer_name, tuple(indices))
                                    original_values[key] = layer.weight.data[tuple(indices)].clone()
                                    layer.weight.data[tuple(indices)] = 0.0
                            except AttributeError as e:
                                if verbose:
                                    print(f"      ⚠️ Could not access {layer_name}: {e}")
                                continue
                        
                        # Evaluate with masked weights
                        perturbed_ppl = compute_perplexity(model, tokenizer, perturb_texts)
                        
                        # Restore original values
                        for weight_info in weights_to_mask:
                            layer_name = weight_info['layer']
                            indices = weight_info['indices']
                            
                            try:
                                layer = model
                                for part in layer_name.split('.'):
                                    layer = getattr(layer, part)
                                
                                if hasattr(layer, 'weight') and layer.weight is not None:
                                    key = (layer_name, tuple(indices))
                                    if key in original_values:
                                        layer.weight.data[tuple(indices)] = original_values[key]
                            except AttributeError:
                                continue
                        
                        # Calculate impact
                        ppl_increase = perturbed_ppl - baseline_ppl
                        ppl_ratio = perturbed_ppl / baseline_ppl
                        
                        # Store results
                        result = {
                            'metric': metric,
                            'topk': k,
                            'mask_ratio': ratio,
                            'weights_masked': num_to_mask,
                            'baseline_ppl': baseline_ppl,
                            'perturbed_ppl': perturbed_ppl,
                            'ppl_increase': ppl_increase,
                            'ppl_ratio': ppl_ratio,
                            'eval_texts_count': len(perturb_texts)
                        }
                        perturbation_results.append(result)
                        
                        if verbose:
                            print(f"  {metric} Top-{k} {ratio*100:.0f}%: "
                                  f"PPL {perturbed_ppl:.1f} (+{ppl_increase:.1f})")
                    
                    except Exception as e:
                        if verbose:
                            print(f"  ❌ Error in {metric} Top-{k} {ratio*100:.0f}%: {e}")
                        
                        # Restore weights even on error
                        try:
                            for weight_info in weights_to_mask:
                                layer_name = weight_info['layer']
                                indices = weight_info['indices']
                                
                                try:
                                    layer = model
                                    for part in layer_name.split('.'):
                                        layer = getattr(layer, part)
                                    
                                    if hasattr(layer, 'weight') and layer.weight is not None:
                                        key = (layer_name, tuple(indices))
                                        if key in original_values:
                                            layer.weight.data[tuple(indices)] = original_values[key]
                                except AttributeError:
                                    continue
                        except:
                            pass
                    
                    pbar.update(1)
    
    print(f"✅ Completed {len(perturbation_results)}/{total_experiments} experiments")
    return perturbation_results


def export_results(
    args: argparse.Namespace,
    session_dir: Path,
    baseline_ppl: float,
    sensitivity_results: Dict,
    ranking_results: Dict,
    perturbation_results: List[Dict],
    model_info: Dict
):
    """Export all results to files."""
    print("\n💾 Exporting Results")
    print("-" * 40)
    
    # 1. Experiment summary
    summary = {
        'experiment_id': session_dir.name.split('_', 2)[-1],
        'timestamp': datetime.now().isoformat(),
        'model_name': args.model,
        'device': model_info.get('device', 'unknown'),
        'num_parameters': model_info.get('num_parameters', 0),
        'baseline_perplexity': baseline_ppl,
        'eval_texts_count': args.eval_limit,
        'sensitivity_metrics': args.metric,
        'topk_values': args.topk,
        'perturbation_ratios': args.perturbation_ratios,
        'experiments_completed': len(perturbation_results),
        'skip_perturbation': args.no_perturbation,
        'fast_perturbation': args.fast_perturbation
    }
    
    if perturbation_results:
        df = pd.DataFrame(perturbation_results)
        summary.update({
            'max_ppl_increase': float(df['ppl_increase'].max()),
            'max_ppl_ratio': float(df['ppl_ratio'].max()),
            'mean_ppl_increase': float(df['ppl_increase'].mean())
        })
    
    with open(session_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary: experiment_summary.json")
    
    # 2. Configuration
    config = vars(args)
    with open(session_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Config: config.json")
    
    # 3. Perturbation results
    if perturbation_results:
        df = pd.DataFrame(perturbation_results)
        df.to_csv(session_dir / 'perturbation_results.csv', index=False)
        print(f"  ✅ Perturbation: perturbation_results.csv ({len(df)} rows)")
    
    # 4. Top-K weights for each metric
    for metric in args.metric:
        if metric in ranking_results:
            for k in args.topk:
                if k in ranking_results[metric]:
                    top_weights = ranking_results[metric][k]
                    
                    weights_data = []
                    for i, weight_info in enumerate(top_weights):
                        weights_data.append({
                            'rank': i + 1,
                            'layer': weight_info['layer'],
                            'indices': str(weight_info['indices']),
                            'sensitivity': float(weight_info['sensitivity'])
                        })
                    
                    weights_df = pd.DataFrame(weights_data)
                    filename = f'top_{k}_weights_{metric}.csv'
                    weights_df.to_csv(session_dir / filename, index=False)
                    print(f"  ✅ Top-{k} {metric}: {filename}")
    
    # 5. Sensitivity statistics
    sens_stats = {}
    for metric, layer_sensitivities in sensitivity_results.items():
        all_sens = torch.cat([sens.flatten() for sens in layer_sensitivities.values()])
        sens_stats[metric] = {
            'num_layers': len(layer_sensitivities),
            'total_weights': int(all_sens.numel()),
            'mean_sensitivity': float(all_sens.mean()),
            'std_sensitivity': float(all_sens.std()),
            'max_sensitivity': float(all_sens.max()),
            'min_sensitivity': float(all_sens.min())
        }
    
    with open(session_dir / 'sensitivity_statistics.json', 'w') as f:
        json.dump(sens_stats, f, indent=2)
    print(f"  ✅ Statistics: sensitivity_statistics.json")
    
    print(f"\n📁 All results saved to: {session_dir}")


def print_summary(
    args: argparse.Namespace,
    baseline_ppl: float,
    perturbation_results: List[Dict],
    sensitivity_results: Dict,
    session_dir: Path
):
    """Print final analysis summary."""
    print("\n🔬 Analysis Complete - Summary")
    print("=" * 60)
    print(f"🤖 Model: {args.model}")
    print(f"📏 Baseline PPL: {baseline_ppl:.2f}")
    print(f"🧮 Sensitivity metrics: {len(args.metric)}")
    print(f"🏆 Top-K values: {len(args.topk)}")
    
    if perturbation_results:
        df = pd.DataFrame(perturbation_results)
        print(f"🎯 Perturbation experiments: {len(df)}")
        print(f"📈 Max PPL increase: {df['ppl_increase'].max():.2f}")
        print(f"📊 Max PPL ratio: {df['ppl_ratio'].max():.2f}x")
        
        # Best result
        best_idx = df['ppl_increase'].idxmax()
        best = df.iloc[best_idx]
        print(f"\n🏆 Most Impactful Perturbation:")
        print(f"  {best['metric']} Top-{best['topk']} at {best['mask_ratio']*100:.0f}% masking")
        print(f"  PPL: {best['baseline_ppl']:.1f} → {best['perturbed_ppl']:.1f} (+{best['ppl_increase']:.1f})")
    
    # Sensitivity summary
    total_weights = 0
    for metric, layer_sensitivities in sensitivity_results.items():
        weights_in_metric = sum(sens.numel() for sens in layer_sensitivities.values())
        total_weights = max(total_weights, weights_in_metric)
    
    print(f"\n📊 Model Analysis:")
    print(f"  Total weights analyzed: {total_weights:,}")
    print(f"  Layers with computed sensitivity: {len(list(sensitivity_results.values())[0])}")
    
    print(f"\n💾 Results: {session_dir}")
    print(f"🎉 Analysis complete!")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    device, session_dir = setup_environment(args)
    
    # Print header
    print_header(args, device, session_dir)
    
    try:
        # Load evaluation data
        print("\n📚 Loading Evaluation Data")
        eval_texts = load_evaluation_data(args)
        print(f"✅ Loaded {len(eval_texts)} evaluation texts")
        
        # Load model
        print("\n🤖 Loading Model")
        model, tokenizer = load_model(args.model, device=device)
        
        model_info = {
            'device': str(next(model.parameters()).device),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_type': type(model).__name__,
            'tokenizer_type': type(tokenizer).__name__
        }
        
        print(f"✅ Model: {model_info['model_type']}")
        print(f"✅ Parameters: {model_info['num_parameters']:,}")
        print(f"✅ Device: {model_info['device']}")
        
        # Baseline evaluation
        print("\n📏 Baseline Evaluation")
        baseline_ppl = compute_perplexity(model, tokenizer, eval_texts)
        print(f"✅ Baseline perplexity: {baseline_ppl:.2f}")
        
        # Sensitivity analysis
        sensitivity_results = run_sensitivity_analysis(
            model, tokenizer, eval_texts, args.metric, device, args.verbose
        )
        
        # Weight ranking
        ranking_results = run_weight_ranking(
            sensitivity_results, args.topk, args.verbose
        )
        
        # Perturbation analysis
        perturbation_results = []
        if not args.no_perturbation:
            perturbation_results = run_perturbation_analysis(
                model, tokenizer, eval_texts, ranking_results,
                args.perturbation_ratios, baseline_ppl,
                args.fast_perturbation, args.verbose
            )
        else:
            print("\n⏭️ Skipping perturbation analysis")
        
        # Export results
        export_results(
            args, session_dir, baseline_ppl, sensitivity_results,
            ranking_results, perturbation_results, model_info
        )
        
        # Print summary
        print_summary(args, baseline_ppl, perturbation_results, sensitivity_results, session_dir)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
