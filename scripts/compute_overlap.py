#!/usr/bin/env python3
"""
Overlap Analysis Script for Critical Weight Analysis

This script computes overlap and similarity metrics between different
weight rankings from various sensitivity methods or experimental conditions.
Supports Jaccard similarity, rank correlation, and other overlap measures.

Usage Examples:
    # Cross-metric overlap analysis
    python scripts/compute_overlap.py \\
        --lists \\
            outputs/p2/llama31_8b/gradxw_perlayer_k100/top_weights.csv \\
            outputs/p2/llama31_8b/gradsq_perlayer_k100/top_weights.csv \\
            outputs/p2/llama31_8b/actmag_perlayer_k100/top_weights.csv \\
        --by tensor \\
        --out outputs/p2/llama31_8b/cross_metric_overlap.json
    
    # Cross-seed stability analysis
    python scripts/compute_overlap.py \\
        --lists \\
            outputs/p2/llama31_8b/gradxw_perlayer_k100_seed0/top_weights.csv \\
            outputs/p2/llama31_8b/gradxw_perlayer_k100_seed1/top_weights.csv \\
        --by tensor \\
        --out outputs/p2/llama31_8b/seed_stability.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

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
        description="Overlap Analysis for Critical Weight Rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--lists',
        nargs='+',
        required=True,
        help='List of CSV files containing weight rankings'
    )
    
    parser.add_argument(
        '--by',
        type=str,
        choices=['tensor', 'parameter', 'layer'],
        default='tensor',
        help='Granularity for overlap computation (default: tensor)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output JSON file for overlap results'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=None,
        help='Consider only top-k weights from each list (default: all)'
    )
    
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Labels for the input lists (default: use filenames)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_weight_ranking(csv_path: str, topk: int = None) -> pd.DataFrame:
    """
    Load weight ranking from CSV file.
    
    Args:
        csv_path: Path to CSV file with weight rankings
        topk: Limit to top-k weights (None for all)
    
    Returns:
        DataFrame with weight rankings
    """
    df = pd.read_csv(csv_path)
    
    if topk:
        df = df.head(topk)
    
    return df


def extract_identifiers(df: pd.DataFrame, granularity: str) -> Set[str]:
    """
    Extract weight identifiers at specified granularity.
    
    Args:
        df: DataFrame with columns 'parameter' and 'indices'
        granularity: 'tensor', 'parameter', or 'layer'
    
    Returns:
        Set of identifiers
    """
    identifiers = set()
    
    for _, row in df.iterrows():
        param_name = row['parameter']
        indices = row['indices']
        
        if granularity == 'tensor':
            # Full tensor:indices specification
            identifiers.add(f"{param_name}:{indices}")
        elif granularity == 'parameter':
            # Just the parameter name
            identifiers.add(param_name)
        elif granularity == 'layer':
            # Extract layer number from parameter name
            if 'layers.' in param_name:
                layer_part = param_name.split('layers.')[1].split('.')[0]
                identifiers.add(f"layer_{layer_part}")
            else:
                identifiers.add("non_layer")
    
    return identifiers


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set1: Set, set2: Set) -> float:
    """Compute overlap coefficient (Szymkiewicz-Simpson coefficient)."""
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0


def compute_rank_correlation(df1: pd.DataFrame, df2: pd.DataFrame, granularity: str) -> Dict[str, float]:
    """
    Compute rank correlation between two weight rankings.
    
    Args:
        df1, df2: DataFrames with weight rankings
        granularity: Granularity for comparison
    
    Returns:
        Dictionary with correlation metrics
    """
    # Extract common identifiers
    ids1 = extract_identifiers(df1, granularity)
    ids2 = extract_identifiers(df2, granularity)
    common_ids = ids1.intersection(ids2)
    
    if len(common_ids) < 2:
        return {'spearman': 0.0, 'kendall': 0.0}
    
    # Create rank mappings
    rank1_map = {}
    rank2_map = {}
    
    for i, (_, row) in enumerate(df1.iterrows()):
        param_name = row['parameter']
        indices = row['indices']
        
        if granularity == 'tensor':
            identifier = f"{param_name}:{indices}"
        elif granularity == 'parameter':
            identifier = param_name
        elif granularity == 'layer':
            if 'layers.' in param_name:
                layer_part = param_name.split('layers.')[1].split('.')[0]
                identifier = f"layer_{layer_part}"
            else:
                identifier = "non_layer"
        
        if identifier in common_ids:
            rank1_map[identifier] = i
    
    for i, (_, row) in enumerate(df2.iterrows()):
        param_name = row['parameter']
        indices = row['indices']
        
        if granularity == 'tensor':
            identifier = f"{param_name}:{indices}"
        elif granularity == 'parameter':
            identifier = param_name
        elif granularity == 'layer':
            if 'layers.' in param_name:
                layer_part = param_name.split('layers.')[1].split('.')[0]
                identifier = f"layer_{layer_part}"
            else:
                identifier = "non_layer"
        
        if identifier in common_ids:
            rank2_map[identifier] = i
    
    # Extract ranks for common identifiers
    common_sorted = sorted(common_ids)
    ranks1 = [rank1_map[id_] for id_ in common_sorted]
    ranks2 = [rank2_map[id_] for id_ in common_sorted]
    
    # Compute correlations
    spearman_corr, _ = spearmanr(ranks1, ranks2)
    kendall_corr, _ = kendalltau(ranks1, ranks2)
    
    return {
        'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'kendall': float(kendall_corr) if not np.isnan(kendall_corr) else 0.0
    }


def analyze_pairwise_overlap(
    rankings: List[pd.DataFrame], 
    labels: List[str], 
    granularity: str
) -> Dict[str, Any]:
    """
    Analyze pairwise overlap between all ranking pairs.
    
    Args:
        rankings: List of DataFrames with weight rankings
        labels: Labels for each ranking
        granularity: Granularity for overlap computation
    
    Returns:
        Dictionary with pairwise overlap metrics
    """
    n_rankings = len(rankings)
    results = {
        'pairwise': {},
        'summary': {
            'mean_jaccard': 0.0,
            'mean_overlap': 0.0,
            'mean_spearman': 0.0,
            'mean_kendall': 0.0
        }
    }
    
    jaccard_values = []
    overlap_values = []
    spearman_values = []
    kendall_values = []
    
    for i in range(n_rankings):
        for j in range(i + 1, n_rankings):
            label_i, label_j = labels[i], labels[j]
            df_i, df_j = rankings[i], rankings[j]
            
            # Extract identifier sets
            ids_i = extract_identifiers(df_i, granularity)
            ids_j = extract_identifiers(df_j, granularity)
            
            # Compute overlap metrics
            jaccard = jaccard_similarity(ids_i, ids_j)
            overlap = overlap_coefficient(ids_i, ids_j)
            
            # Compute rank correlations
            rank_corr = compute_rank_correlation(df_i, df_j, granularity)
            
            pair_key = f"{label_i}_vs_{label_j}"
            results['pairwise'][pair_key] = {
                'jaccard_similarity': jaccard,
                'overlap_coefficient': overlap,
                'spearman_correlation': rank_corr['spearman'],
                'kendall_correlation': rank_corr['kendall'],
                'set_sizes': {'first': len(ids_i), 'second': len(ids_j)},
                'intersection_size': len(ids_i.intersection(ids_j))
            }
            
            jaccard_values.append(jaccard)
            overlap_values.append(overlap)
            spearman_values.append(rank_corr['spearman'])
            kendall_values.append(rank_corr['kendall'])
            
            logger.info(f"{pair_key}: Jaccard={jaccard:.3f}, Overlap={overlap:.3f}, Spearman={rank_corr['spearman']:.3f}")
    
    # Compute summary statistics
    if jaccard_values:
        results['summary']['mean_jaccard'] = np.mean(jaccard_values)
        results['summary']['mean_overlap'] = np.mean(overlap_values)
        results['summary']['mean_spearman'] = np.mean(spearman_values)
        results['summary']['mean_kendall'] = np.mean(kendall_values)
        results['summary']['std_jaccard'] = np.std(jaccard_values)
        results['summary']['std_overlap'] = np.std(overlap_values)
        results['summary']['std_spearman'] = np.std(spearman_values)
        results['summary']['std_kendall'] = np.std(kendall_values)
    
    return results


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("🚀 Starting Overlap Analysis")
    logger.info(f"📊 Analyzing {len(args.lists)} weight ranking files")
    logger.info(f"🎯 Granularity: {args.by}")
    
    # Load rankings
    rankings = []
    labels = args.labels if args.labels else []
    
    for i, csv_path in enumerate(args.lists):
        if not Path(csv_path).exists():
            logger.error(f"File not found: {csv_path}")
            continue
        
        df = load_weight_ranking(csv_path, args.topk)
        rankings.append(df)
        
        # Generate label if not provided
        if i >= len(labels):
            label = Path(csv_path).stem
            labels.append(label)
        
        logger.info(f"Loaded {len(df)} weights from {csv_path} (label: {labels[i]})")
    
    if len(rankings) < 2:
        logger.error("Need at least 2 rankings for overlap analysis")
        return
    
    # Perform overlap analysis
    logger.info("🔍 Computing pairwise overlaps...")
    results = analyze_pairwise_overlap(rankings, labels, args.by)
    
    # Add metadata
    results['metadata'] = {
        'input_files': args.lists,
        'labels': labels,
        'granularity': args.by,
        'topk': args.topk,
        'num_rankings': len(rankings)
    }
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_path}")
    
    # Print summary
    summary = results['summary']
    logger.info("📋 Summary Statistics:")
    logger.info(f"  Mean Jaccard Similarity: {summary['mean_jaccard']:.3f} ± {summary.get('std_jaccard', 0):.3f}")
    logger.info(f"  Mean Overlap Coefficient: {summary['mean_overlap']:.3f} ± {summary.get('std_overlap', 0):.3f}")
    logger.info(f"  Mean Spearman Correlation: {summary['mean_spearman']:.3f} ± {summary.get('std_spearman', 0):.3f}")
    logger.info(f"  Mean Kendall Correlation: {summary['mean_kendall']:.3f} ± {summary.get('std_kendall', 0):.3f}")
    
    logger.info("🎉 Overlap analysis completed successfully!")


if __name__ == "__main__":
    main()
