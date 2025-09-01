# Critical Weight Analysis Results Export

**Export Date:** September 1, 2025  
**Model:** meta-llama/Llama-3.1-8B  
**Analysis Type:** Gradient x Weight sensitivity analysis

## Contents Overview

### Phase 1 Baseline Results (`p1_baseline/llama31_8b/`)
- **Configuration:** `config.json` - Analysis parameters and settings
- **Experiment Manifest:** `experiment_manifest.json` - Detailed experiment metadata
- **Statistics:** `sensitivity_stats.json` - Computed sensitivity statistics
- **Top Weights:** `top_weights.csv` - Most critical weights identified
- **Visualizations:** `plots/`
  - `grad_x_weight_k100_layer_comparison.png` - Layer-wise comparison
  - `grad_x_weight_k100_sensitivity_distribution.png` - Distribution analysis
  - `grad_x_weight_k100_sensitivity_heatmap.png` - Sensitivity heatmap

### Phase 2 Enhanced Results (`p2/llama31_8b/gradxw_perlayer_k100/`)
- **Configuration:** `config.json` - Enhanced analysis parameters
- **Experiment Manifest:** `experiment_manifest.json` - Enhanced experiment metadata
- **Statistics:** `sensitivity_stats.json` - Per-layer sensitivity statistics
- **Top Weights:** `top_weights.csv` - Layer-specific critical weights (k=100)
- **Visualizations:** `plots/`
  - `grad_x_weight_k100_layer_comparison.png` - Enhanced layer comparison
  - `grad_x_weight_k100_sensitivity_distribution.png` - Distribution analysis
  - `grad_x_weight_k100_sensitivity_heatmap.png` - Detailed sensitivity heatmap

## Analysis Parameters
- **Metric:** grad_x_weight (Gradient × Weight)
- **Mode:** per_layer analysis
- **Top-K:** 100 most critical weights
- **Max Samples:** 200 evaluation texts
- **Device:** CUDA-enabled GPU

## File Descriptions

### JSON Files
- **config.json:** Contains all analysis parameters, model settings, and runtime configuration
- **experiment_manifest.json:** Comprehensive metadata including timestamps, git info, and results summary
- **sensitivity_stats.json:** Statistical measures of weight sensitivity across layers

### CSV Files
- **top_weights.csv:** Tabular data of the most critical weights with their sensitivity scores and locations

### PNG Files
- **layer_comparison.png:** Comparative visualization across model layers
- **sensitivity_distribution.png:** Distribution plots showing sensitivity patterns
- **sensitivity_heatmap.png:** Heatmap visualization of weight criticality

## Usage Notes
- All files are in standard formats (JSON, CSV, PNG) for easy analysis
- The enhanced Phase 2 results provide more detailed per-layer insights
- Visualizations are publication-ready with proper formatting
