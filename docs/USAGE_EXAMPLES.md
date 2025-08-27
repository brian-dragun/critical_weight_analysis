# Critical Weight Analysis - Enhanced Phase 1 Usage Examples

## 🎯 PhD Research Workflow Examples

The enhanced Phase 1 implementation now supports the complete research workflow you specified. Here are practical examples:

### Basic Usage Examples

#### 1. Simple Gradient-Based Analysis
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode global
```

#### 2. Hutchinson Diagonal Estimator (Advanced Curvature)
```bash
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric hutchinson_diag \
    --topk 100 \
    --mode global \
    --seeds 42
```

#### 3. Non-Gradient Magnitude Analysis
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 500 \
    --mode per_layer \
    --save-plots
```

#### 4. Activation-Weighted Magnitude
```bash
python phase1_runner_enhanced.py \
    --model EleutherAI/pythia-410m \
    --metric act_mag \
    --topk 200 \
    --mode global
```

### Advanced Research Workflows

#### 5. Complete Perturbation Study with Controls
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode global \
    --perturb sign_flip \
    --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --save-plots
```

#### 6. Gaussian Noise Perturbation Analysis
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_squared \
    --topk 100 \
    --perturb gauss_noise \
    --perturb-scale 0.1 \
    --controls random_k,bottom_k
```

#### 7. Stability Analysis Across Seeds
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --save-plots
```

### Full PhD Research Workflow (Your Target)

#### 8. Complete LLaMA Analysis as Specified
```bash
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric hutchinson_diag \
    --topk 100 --mode global \
    --perturb sign_flip --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --save-plots \
    --out-dir outputs/llama31_8b_hutch_diag_k100 \
    --verbose
```

### Comparative Studies

#### 9. Compare Multiple Metrics
```bash
# Run different metrics separately and compare results
for metric in grad_x_weight grad_squared hutchinson_diag magnitude; do
    python phase1_runner_enhanced.py \
        --model gpt2 \
        --metric $metric \
        --topk 100 \
        --mode global \
        --controls random_k,bottom_k \
        --out-dir outputs/comparison_${metric} \
        --save-plots
done
```

#### 10. K-value Sensitivity Study
```bash
# Test different K values
for k in 50 100 200 500; do
    python phase1_runner_enhanced.py \
        --model gpt2 \
        --metric grad_x_weight \
        --topk $k \
        --mode global \
        --perturb sign_flip \
        --controls random_k,bottom_k \
        --out-dir outputs/k_study_${k}
done
```

## 📊 Output Structure

Each run creates a comprehensive output directory:

```
outputs/llama31_8b_hutch_diag_k100/
├── experiment_manifest.json          # Complete reproducibility info
├── config.json                       # Runtime configuration
├── sensitivity_stats.json            # Statistical summaries
├── top_weights.csv                   # Top-K weight selections
├── control_baselines.json            # Random/bottom-K controls
├── perturbation_results.json         # Perturbation experiment results
├── stability_results.json            # Jaccard stability analysis
└── plots/                            # Comprehensive visualizations
    ├── hutchinson_diag_k100_sensitivity_distribution.png
    ├── hutchinson_diag_k100_layer_comparison.png
    ├── hutchinson_diag_k100_sensitivity_heatmap.png
    ├── hutchinson_diag_k100_perturbation_effects.png
    └── hutchinson_diag_k100_stability_analysis.png
```

## 🔬 Research Features Available

### Sensitivity Metrics
- ✅ `grad_x_weight` - Gradient × weight
- ✅ `grad_squared` - Gradient squared  
- ✅ `hessian_diag` - Simple diagonal Hessian
- ✅ `hutchinson_diag` - Hutchinson diagonal estimator
- ✅ `magnitude` - Simple weight magnitude
- ✅ `act_mag` - Activation-weighted magnitude

### Perturbation Methods
- ✅ `zero` - Zero out weights
- ✅ `sign_flip` - Flip weight signs
- ✅ `gauss_noise` - Add Gaussian noise (with --perturb-scale)
- ✅ `bit_flip` - Bit flip simulation (with --perturb-prob)

### Control Baselines
- ✅ `random_k` - Random weight selection
- ✅ `bottom_k` - Least sensitive weights

### Ranking Modes
- ✅ `per_layer` - Top-K within each layer
- ✅ `global` - Top-K across all layers

### Analysis Features
- ✅ Multi-seed stability analysis
- ✅ Jaccard overlap computation
- ✅ Enhanced evaluation metrics (perplexity, NLL, token accuracy)
- ✅ Comprehensive visualization suite
- ✅ Full experiment manifest logging

## 🚀 Quick Start for PhD Research

For immediate PhD research use, start with this command:

```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode global \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --save-plots \
    --verbose
```

This provides a complete analysis with:
- Gradient-based sensitivity
- Global top-100 weight ranking
- Sign flip perturbation testing
- Random and bottom-K control baselines
- Multi-seed stability analysis
- Comprehensive visualizations
- Full reproducibility tracking

The implementation is now ready for rigorous PhD-level critical weight analysis research!
