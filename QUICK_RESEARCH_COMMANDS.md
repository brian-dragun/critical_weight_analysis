# Quick Research Commands for Llama & Mistral Testing

## 🚀 Immediate Testing Commands

### 1. Quick System Validation (2-3 minutes)
```bash
# Test basic functionality with GPT-2
cd /home/ubuntu/nova/critical_weight_analysis
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 50 \
    --max-samples 10 \
    --save-plots \
    --out-dir results/quick_validation/gpt2_test
```

### 2. Llama-3.1-8B Quick Analysis (5-10 minutes)
```bash
# Fast per-layer analysis with gradient-based sensitivity
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 20 \
    --save-plots \
    --out-dir results/quick_research/llama31_8b_quick
```

### 3. Mistral-7B Quick Analysis (5-10 minutes)
```bash
# Fast per-layer analysis for comparison
python phase1_runner_enhanced.py \
    --model mistralai/Mistral-7B-v0.1 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 20 \
    --save-plots \
    --out-dir results/quick_research/mistral_7b_quick
```

### 4. Perturbation Demo (3-5 minutes)
```bash
# Demonstrate perturbation effects with controls
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --device cuda \
    --max-samples 15 \
    --save-plots \
    --out-dir results/quick_research/perturbation_demo
```

## 🔬 Full Research Protocol

### Phase 1: Comprehensive Llama Analysis
```bash
# 1. Gradient-based sensitivity (per-layer)
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 500 \
    --mode per_layer \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/research/llama31_8b/grad_x_weight_per_layer

# 2. Magnitude-based sensitivity (global) - WARNING: Can take 1+ hours
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 1000 \
    --mode global \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/research/llama31_8b/magnitude_global

# 3. Perturbation experiments with stability analysis
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 300 \
    --mode per_layer \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --device cuda \
    --max-samples 40 \
    --save-plots \
    --out-dir results/research/llama31_8b/perturbation_analysis
```

### Phase 2: Comprehensive Mistral Analysis
```bash
# 1. Multi-metric comparison
for metric in "magnitude" "grad_x_weight" "grad_squared"; do
    python phase1_runner_enhanced.py \
        --model mistralai/Mistral-7B-v0.1 \
        --metric $metric \
        --topk 200 \
        --mode per_layer \
        --max-samples 30 \
        --save-plots \
        --out-dir results/research/mistral_7b/multi_metric_${metric}
done

# 2. Robustness analysis
python phase1_runner_enhanced.py \
    --model mistralai/Mistral-7B-v0.1 \
    --metric grad_x_weight \
    --topk 400 \
    --mode per_layer \
    --perturb sign_flip \
    --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/research/mistral_7b/robustness_analysis
```

### Phase 3: Cross-Model Comparison
```bash
# Standardized comparison protocol
for model in "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1"; do
    model_name=$(basename $model | sed 's/[^a-zA-Z0-9]/_/g')
    
    python phase1_runner_enhanced.py \
        --model $model \
        --metric grad_x_weight \
        --topk 500 \
        --mode per_layer \
        --perturb sign_flip \
        --controls random_k,bottom_k \
        --seeds 0,1,2 \
        --stability-check \
        --device cuda \
        --max-samples 50 \
        --save-plots \
        --out-dir results/research/cross_comparison/${model_name}_standard
done
```

## 🛠️ Automated Testing

### Run All Tests with Script
```bash
# Run the complete testing suite
cd /home/ubuntu/nova/critical_weight_analysis
./scripts/run_research_tests.sh

# Or run specific test phases
./scripts/run_research_tests.sh validation
./scripts/run_research_tests.sh llama
./scripts/run_research_tests.sh mistral
./scripts/run_research_tests.sh perturbation
```

## 📊 Expected Results Summary

### What to Look For:

1. **Sensitivity Patterns**:
   - Power-law distribution of sensitivity scores
   - Higher sensitivity in middle layers (layers 10-25)
   - Attention weights > MLP weights > LayerNorm weights

2. **Model Comparisons**:
   - Llama vs Mistral architectural differences in sensitivity patterns
   - Consistent attention mechanism importance across models
   - Different layer-wise distributions due to architectural differences

3. **Perturbation Effects**:
   - Top-K perturbations: 10x-1000x perplexity increase
   - Random-K perturbations: 2x-10x perplexity increase  
   - Bottom-K perturbations: <2x perplexity increase

4. **Stability Metrics**:
   - Jaccard overlap >0.8 for top-100 weights across seeds
   - Consistent ranking patterns across multiple runs
   - Good correlation between gradient-based and magnitude metrics

### Files Generated:

For each experiment, you'll get:
- `experiment_manifest.json` - Full reproducibility tracking
- `top_weights.csv` - Ranked weight importance scores
- `sensitivity_stats.json` - Statistical summaries
- `config.json` - Experiment configuration
- `plots/` directory with visualization files

## ⚠️ Important Notes

1. **Memory Requirements**: 
   - Llama-3.1-8B needs ~15-25GB GPU memory
   - Mistral-7B needs ~12-20GB GPU memory
   - Global ranking can take 1+ hours for large models

2. **Access Requirements**:
   - Llama models may require Hugging Face access approval
   - Mistral models are generally publicly accessible

3. **Time Estimates**:
   - Quick tests: 2-10 minutes
   - Full per-layer analysis: 10-30 minutes
   - Global ranking: 30 minutes - 2 hours
   - Complete research protocol: 2-6 hours total

4. **Troubleshooting**:
   - If CUDA OOM: Reduce `--max-samples` or `--topk`
   - If model access denied: Check Hugging Face authentication
   - If Hutchinson fails: Use `grad_x_weight` or `magnitude` instead
