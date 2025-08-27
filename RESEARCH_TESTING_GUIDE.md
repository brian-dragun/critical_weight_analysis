# Research Testing Guide: Critical Weight Analysis on Llama & Mistral Models

## Overview

This guide provides comprehensive testing protocols for your PhD research on **super-weight identification and prioritization** in Large Language Models, specifically targeting **Llama** and **Mistral** model families using the enhanced critical weight analysis system.

---

## 🎯 Research Objectives

### Primary Research Questions
1. **Do super-weights exist?** - Can we identify a small subset of weights that disproportionately affect model behavior?
2. **Are super-weights consistent?** - Do the same types of weights emerge as critical across different models?
3. **How sensitive are models?** - What's the magnitude of performance degradation when critical weights are perturbed?
4. **Do different metrics agree?** - How do gradient-based vs. non-gradient sensitivity metrics compare?

### Expected Contributions
- Empirical evidence for super-weight hypothesis in modern LLMs
- Comparative analysis of weight sensitivity across model architectures
- Foundation for Phase 2 robustness interventions (pruning, fault tolerance)

---

## 🔬 Testing Protocol

### Phase 1: Model Compatibility & Baseline Establishment

#### 1.1 Quick Compatibility Tests
```bash
# Test small models first to verify system compatibility
python phase1_runner_enhanced.py \
    --model microsoft/DialoGPT-small \
    --metric magnitude \
    --topk 10 \
    --max-samples 5 \
    --out-dir results/compatibility/dialogpt_small

python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 50 \
    --mode per_layer \
    --max-samples 10 \
    --out-dir results/compatibility/gpt2_baseline
```

**Expected Results:**
- ✅ Models load successfully on GPU
- ✅ Sensitivity computation completes without errors
- ✅ Output files generated (manifest.json, top_weights.csv, sensitivity_stats.json)
- ⏱️ Execution time: <2 minutes for small models

#### 1.2 Memory & Performance Profiling
```bash
# Test GPU memory usage with larger context
python phase1_runner_enhanced.py \
    --model microsoft/DialoGPT-medium \
    --metric grad_x_weight \
    --topk 100 \
    --max-samples 20 \
    --max-length 1024 \
    --out-dir results/profiling/dialogpt_medium_memory
```

**Expected Results:**
- 📊 GPU memory usage < 80% of available (monitor with `nvidia-smi`)
- 📈 Performance scaling linearly with context length
- 🔄 No memory leaks during extended runs

### Phase 2: Llama Model Analysis

#### 2.1 Llama-3.1-8B Comprehensive Analysis
```bash
# Test 1: Gradient-based sensitivity with per-layer ranking
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/llama31_8b/grad_x_weight_per_layer

# Test 2: Magnitude-based sensitivity with global ranking
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 1000 \
    --mode global \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/llama31_8b/magnitude_global

# Test 3: Perturbation experiments with controls
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 500 \
    --mode per_layer \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --device cuda \
    --max-samples 30 \
    --save-plots \
    --out-dir results/llama31_8b/perturbation_analysis
```

**Expected Results for Llama-3.1-8B:**
- 🧠 **Model Size**: ~8B parameters across ~32 layers
- ⏱️ **Execution Time**: 5-15 minutes per experiment (depending on topk and mode)
- 📊 **Memory Usage**: ~15-25GB GPU memory
- 🎯 **Top Weights Distribution**: 
  - Attention weights (Q,K,V projections) likely prominent
  - Feed-forward layers (up_proj, down_proj) significant
  - Layer normalization weights potentially critical
- 📈 **Sensitivity Patterns**:
  - Higher sensitivity in middle layers (layers 10-25)
  - Attention mechanism weights more sensitive than MLPs
  - Gradient-based metrics showing higher variance than magnitude

#### 2.2 Llama Model Family Comparison
```bash
# Compare different Llama sizes (if available)
for model in "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"; do
    python phase1_runner_enhanced.py \
        --model $model \
        --metric grad_x_weight \
        --topk 200 \
        --mode per_layer \
        --max-samples 30 \
        --save-plots \
        --out-dir results/llama_comparison/$(basename $model)_grad_analysis
done
```

**Expected Comparative Results:**
- 🔄 **Consistency**: Similar layer patterns across Llama versions
- 📊 **Scaling**: Larger models showing more concentrated sensitivity
- 🎯 **Architecture**: Attention patterns consistent, MLP patterns varying

### Phase 3: Mistral Model Analysis

#### 3.1 Mistral-7B Analysis
```bash
# Test 1: Standard sensitivity analysis
python phase1_runner_enhanced.py \
    --model mistralai/Mistral-7B-v0.1 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 50 \
    --save-plots \
    --out-dir results/mistral_7b/grad_x_weight_analysis

# Test 2: Multi-metric comparison
for metric in "magnitude" "grad_x_weight" "grad_squared"; do
    python phase1_runner_enhanced.py \
        --model mistralai/Mistral-7B-v0.1 \
        --metric $metric \
        --topk 200 \
        --mode per_layer \
        --max-samples 30 \
        --save-plots \
        --out-dir results/mistral_7b/multi_metric_${metric}
done

# Test 3: Perturbation robustness testing
python phase1_runner_enhanced.py \
    --model mistralai/Mistral-7B-v0.1 \
    --metric grad_x_weight \
    --topk 300 \
    --mode per_layer \
    --perturb sign_flip \
    --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --device cuda \
    --max-samples 40 \
    --save-plots \
    --out-dir results/mistral_7b/robustness_analysis
```

**Expected Results for Mistral-7B:**
- 🧠 **Architecture**: Similar to Llama but with sliding window attention
- 📊 **Sensitivity Distribution**: More uniform across layers due to architectural differences
- 🎯 **Critical Weights**: Attention mechanism still prominent, but different pattern than Llama
- 🔄 **Stability**: Should show good rank correlation across seeds (Jaccard > 0.7)

### Phase 4: Cross-Architecture Comparison

#### 4.1 Comparative Sensitivity Analysis
```bash
# Standardized comparison across models
for model in "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1"; do
    model_name=$(basename $model | sed 's/[^a-zA-Z0-9]/_/g')
    
    # Run identical analysis for fair comparison
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
        --out-dir results/cross_comparison/${model_name}_standard_analysis
done
```

#### 4.2 Statistical Analysis Script
```bash
# Create comparison analysis (you'll need to implement this)
python scripts/analyze_cross_model_results.py \
    --result-dirs results/cross_comparison/ \
    --output results/cross_comparison/statistical_analysis.json
```

---

## 📊 Expected Results & Analysis

### Key Metrics to Monitor

#### 1. **Sensitivity Distribution Analysis**
- **Histogram of sensitivity scores**: Should show power-law or log-normal distribution
- **Layer-wise patterns**: Middle layers typically more sensitive
- **Parameter type analysis**: Attention > MLP > LayerNorm (expected hierarchy)

#### 2. **Perturbation Impact Measurement**
- **Baseline Perplexity**: 
  - Llama-3.1-8B: ~5-15 on standard text
  - Mistral-7B: ~5-12 on standard text
- **Post-perturbation Perplexity**:
  - Top-K perturbation: 10x-1000x increase expected
  - Random-K perturbation: 2x-10x increase expected
  - Bottom-K perturbation: Minimal change (<2x increase)

#### 3. **Rank Stability Metrics**
- **Jaccard Overlap** (across seeds):
  - Top-100: >0.8 (high stability expected)
  - Top-500: >0.6 (moderate stability)
  - Top-1000: >0.4 (lower but significant stability)

#### 4. **Cross-Model Consistency**
- **Architecture patterns**: Similar attention weight importance
- **Layer distribution**: Comparable middle-layer sensitivity peaks
- **Metric agreement**: grad_x_weight and magnitude should correlate (r>0.5)

### Publication-Ready Figures

The system will automatically generate:

1. **`sensitivity_distribution.png`**: Histogram of weight sensitivity scores
2. **`layer_comparison.png`**: Layer-wise sensitivity analysis
3. **`sensitivity_heatmap.png`**: Parameter type × layer sensitivity matrix
4. **`perturbation_effects.png`**: Before/after perplexity comparison
5. **`stability_analysis.png`**: Rank correlation across seeds (if stability-check enabled)

---

## 🔧 Troubleshooting & Optimization

### Common Issues & Solutions

#### Memory Issues
```bash
# If OOM errors occur, reduce batch size:
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 100 \
    --max-samples 20 \
    --max-length 256 \
    --out-dir results/memory_optimized/
```

#### Long Execution Times
```bash
# For faster iteration during development:
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 50 \
    --mode per_layer \
    --max-samples 10 \
    --out-dir results/quick_test/
```

#### Gradient Computation Issues
```bash
# If Hutchinson diagonal fails, use first-order methods:
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --out-dir results/gradient_safe/
```

---

## 📋 Research Checklist

### Before Running Experiments
- [ ] Verify GPU memory available (>20GB recommended for 7B+ models)
- [ ] Check CUDA compatibility (`torch.cuda.is_available()`)
- [ ] Confirm model access (Hugging Face tokens if needed)
- [ ] Create organized output directory structure

### During Experiments
- [ ] Monitor GPU memory usage (`nvidia-smi`)
- [ ] Check for consistent baseline perplexity across runs
- [ ] Verify progress bars and logging are working
- [ ] Save experiment configurations in manifest files

### After Experiments
- [ ] Compare results across different metrics
- [ ] Analyze rank stability patterns
- [ ] Generate publication-ready visualizations
- [ ] Document unexpected findings or limitations

---

## 🎓 Research Impact & Next Steps

### Expected PhD Contributions
1. **Empirical validation** of super-weight hypothesis in modern LLMs
2. **Comparative analysis** across major transformer architectures
3. **Methodological framework** for weight sensitivity analysis
4. **Foundation for robustness research** (Phase 2)

### Potential Publication Venues
- **NeurIPS/ICML**: Core machine learning conferences
- **ICLR**: Learning representations focus
- **EMNLP/ACL**: NLP applications
- **IEEE Security**: If focusing on robustness/fault tolerance

### Phase 2 Research Directions
- **Pruning strategies** based on sensitivity analysis
- **Fault injection experiments** for hardware robustness
- **Adversarial robustness** through critical weight protection
- **Model compression** using super-weight identification

---

## 📖 Example Research Workflow

```bash
# Day 1: System validation
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5 --out-dir results/validation/

# Day 2-3: Llama analysis
python phase1_runner_enhanced.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --topk 500 --mode per_layer --save-plots --out-dir results/llama/main_analysis/

# Day 4-5: Mistral analysis  
python phase1_runner_enhanced.py --model mistralai/Mistral-7B-v0.1 --metric grad_x_weight --topk 500 --mode per_layer --save-plots --out-dir results/mistral/main_analysis/

# Day 6: Cross-model comparison
python phase1_runner_enhanced.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --topk 300 --perturb sign_flip --controls random_k,bottom_k --seeds 0,1,2 --out-dir results/comparison/llama/
python phase1_runner_enhanced.py --model mistralai/Mistral-7B-v0.1 --metric grad_x_weight --topk 300 --perturb sign_flip --controls random_k,bottom_k --seeds 0,1,2 --out-dir results/comparison/mistral/

# Day 7: Analysis and visualization
python scripts/generate_research_report.py --input-dirs results/ --output research_summary.pdf
```

This systematic approach will provide comprehensive data for your PhD research on critical weight analysis in large language models.
