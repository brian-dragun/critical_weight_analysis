# ESA Runner Usage Guide

**Version**: 2.0 (esa_runner.py)  
**Date**: 2025-08-29  
**Prerequisites**: Completed baseline testing with excellent results (PPL ~1.03, accuracy ~99%)  

---

## 🚀 **Getting Started with ESA Runner**

### **Basic Command Structure**
```bash
python esa_runner.py \
  --model MODEL_NAME \
  --metric SENSITIVITY_METRIC \
  --mode {per_layer|global} \
  --topk NUMBER_OF_WEIGHTS \
  --max-samples DATA_SAMPLES \
  --seeds SEED_LIST \
  --stability-check \
  --save-plots \
  --out-dir OUTPUT_DIRECTORY
```

### **Key Parameters**
- **`--model`**: Use your baseline-tested models (meta-llama/Llama-3.1-8B recommended)
- **`--metric`**: Sensitivity analysis method (grad_x_weight, act_mag, hutchinson_diag, etc.)
- **`--mode`**: per_layer (recommended) or global ranking
- **`--topk`**: Number of critical weights to identify (100-300 typical)
- **`--seeds`**: Use same seeds as baselines (1337,123,999) for consistency
- **`--dtype`**: Match your baseline dtype (bf16) for valid comparisons

---

## 📊 **Phase 2: Sensitivity Profiling**

### **1. Start with Your Best Model (Llama-3.1-8B)**

**Basic gradient×weight analysis (recommended first test):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --seeds 1337,123,999 \
  --stability-check \
  --dtype bf16 \
  --save-plots \
  --verbose \
  --out-dir outputs/esa/llama31_8b/gradxw_perlayer_k100
```

**Expected runtime**: 15-25 minutes  
**Success criteria**: Jaccard overlap > 0.3, heavy-tailed sensitivity distribution

### **2. Cross-Metric Validation**

**Activation magnitude (non-gradient baseline):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric act_mag \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/actmag_perlayer_k100
```

**Hutchinson diagonal (curvature-based):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric hutchinson_diag \
  --mode global \
  --topk 300 \
  --max-samples 120 \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/hutch_global_k300
```

**Gradient squared (magnitude focus):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_squared \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/gradsq_perlayer_k100
```

---

## 🧪 **Phase 3: Perturbation Testing**

### **3. Validate Critical Weight Importance**

**Sign flip perturbation (strongest test):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb sign_flip \
  --perturb-scale 1.0 \
  --controls random_k,bottom_k \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/signflip_k100
```

**Gaussian noise perturbation (robustness test):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb gauss_noise \
  --perturb-scale 0.02 \
  --controls random_k,bottom_k \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/gauss_0p02_k100
```

**Bit flip perturbation (discrete corruption):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb bit_flip \
  --perturb-prob 0.05 \
  --controls random_k,bottom_k \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/bitflip_p005_k100
```

**Zero perturbation (ablation study):**
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb zero \
  --controls random_k,bottom_k \
  --seeds 1337,123,999 \
  --stability-check \
  --save-plots \
  --out-dir outputs/esa/llama31_8b/zero_k100
```

---

## 🔍 **Understanding ESA Outputs**

### **File Structure**
```
outputs/esa/llama31_8b/gradxw_perlayer_k100/
├── experiment_manifest.json     # Experiment metadata
├── sensitivity_stats.json       # Raw sensitivity scores
├── top_weights.csv              # Ranked critical weights
├── stability_summary.json       # Cross-seed stability metrics
├── perturbation_results.json    # Performance impact data (if --perturb used)
├── control_analysis.json        # Random/bottom-k baselines (if --controls used)
└── plots/                       # Visualization files (if --save-plots used)
    ├── sensitivity_heatmap.png
    ├── weight_distribution.png
    └── stability_analysis.png
```

### **Key Metrics to Evaluate**

**Stability Metrics (stability_summary.json):**
```json
{
  "jaccard_mean": 0.42,          // ✅ Good if > 0.3
  "jaccard_std": 0.08,           // ✅ Good if < 0.15  
  "jaccard_all": [0.45, 0.39],   // Pairwise overlaps
  "num_seeds": 3,
  "topk_overlap_size": 100
}
```

**Perturbation Results (perturbation_results.json):**
```json
{
  "super_weights": {
    "ppl_change": 0.15,          // ✅ Higher is better
    "accuracy_drop": 0.08        // ✅ Higher is better
  },
  "random_k": {
    "ppl_change": 0.03,          // Should be much lower
    "accuracy_drop": 0.02        // Should be much lower
  },
  "bottom_k": {
    "ppl_change": 0.01,          // Should be lowest
    "accuracy_drop": 0.005       // Should be lowest
  }
}
```

**Success Criteria:**
- **ΔPPL(super) ≫ ΔPPL(random) ≫ ΔPPL(bottom)** = Clear super-weight validation
- **Jaccard overlap > 0.3** = Stable weight identification across seeds
- **Heavy-tailed distribution** = Clear sensitivity hierarchy

---

## 🎯 **Recommended Test Workflows**

### **Week 1: Sensitivity Discovery**
```bash
# Day 1: Quick validation test (5-10 minutes)
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --topk 50 --max-samples 50 --seeds 1337 --verbose

# Day 2-3: Full sensitivity analysis (20-30 minutes each)
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --mode per_layer --topk 100 --seeds 1337,123,999 --stability-check
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric act_mag --mode per_layer --topk 100 --seeds 1337,123,999 --stability-check

# Day 4-5: Curvature analysis (25-40 minutes)
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric hutchinson_diag --mode global --topk 300 --seeds 1337,123,999 --stability-check
```

### **Week 2: Perturbation Validation**
```bash
# Day 1-2: Core perturbation tests (30-45 minutes each)
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --perturb sign_flip --controls random_k,bottom_k --seeds 1337,123,999
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --perturb gauss_noise --perturb-scale 0.02 --controls random_k,bottom_k --seeds 1337,123,999

# Day 3-4: Extended perturbation tests (30-45 minutes each)
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --perturb bit_flip --perturb-prob 0.05 --controls random_k,bottom_k --seeds 1337,123,999
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --perturb zero --controls random_k,bottom_k --seeds 1337,123,999

# Day 5: Cross-metric perturbation validation
python esa_runner.py --model meta-llama/Llama-3.1-8B --metric hutchinson_diag --perturb sign_flip --controls random_k,bottom_k --seeds 1337,123,999
```

### **Week 3: Multi-Model Analysis**
```bash
# Test other models from your baseline suite
python esa_runner.py --model mistralai/Mistral-7B-v0.3 --metric grad_x_weight --perturb sign_flip --controls random_k,bottom_k --seeds 1337,123,999
python esa_runner.py --model microsoft/Phi-3-mini-4k-instruct --metric grad_x_weight --perturb sign_flip --controls random_k,bottom_k --seeds 1337,123,999
python esa_runner.py --model EleutherAI/pythia-1.4b --metric grad_x_weight --perturb sign_flip --controls random_k,bottom_k --seeds 1337,123,999
```

---

## 💡 **Pro Tips & Best Practices**

### **Performance Optimization**
```bash
# Quick test for development
--topk 50 --max-samples 50 --seeds 1337

# Production run for results
--topk 100 --max-samples 200 --seeds 1337,123,999 --stability-check

# Deep analysis for key experiments  
--topk 300 --max-samples 500 --seeds 1337,123,999,42,2023 --stability-check
```

### **Debugging and Monitoring**
```bash
# Add verbose logging for detailed progress
--verbose

# Save all plots for analysis
--save-plots

# Custom output directory for organization
--out-dir outputs/esa/EXPERIMENT_NAME
```

### **Architecture-Specific Tips**

**For Large Models (Llama-3.1-8B, Mixtral):**
- Use `--max-samples 200` or lower to manage compute time
- Consider `--mode per_layer` for better granularity
- Monitor GPU memory usage

**For Small Models (pythia-1.4b):**
- Can use higher `--max-samples` (500+) for more stable results
- Try both `per_layer` and `global` modes
- Faster iteration for methodology development

**For MoE Models (Mixtral):**
- Focus on expert-specific analysis
- Use `--mode global` for cross-expert comparison
- Expect longer compute times due to model size

---

## 🚨 **Troubleshooting**

### **Common Issues**

**"Invalid choice" errors:**
- Check parameter spelling: `act_mag` not `activation_magnitude`
- Use `gauss_noise` not `gaussian` for perturbations
- Verify model name matches HuggingFace format

**Memory errors:**
- Reduce `--max-samples` (try 100 or 50)
- Use `--dtype bf16` for memory efficiency
- Monitor GPU usage with `nvidia-smi`

**Low stability scores:**
- Check if model is in evaluation mode (automatically enforced)
- Verify seeds are properly set
- Consider increasing `--max-samples` for more stable metrics

**Weak perturbation effects:**
- Increase perturbation strength: `--perturb-scale 0.05` or higher
- Check baseline model performance is consistent
- Verify control baselines show expected low impact

### **Expected Runtimes (approximate)**
- **Quick test**: 5-10 minutes (topk=50, samples=50)
- **Standard analysis**: 15-25 minutes (topk=100, samples=200)  
- **Deep analysis**: 30-45 minutes (topk=300, samples=500)
- **Perturbation tests**: +50% runtime due to multiple evaluations

---

## 🎯 **Success Validation**

### **Phase 2 Success Criteria**
- ✅ **Stability**: Jaccard overlap > 0.3 across seeds
- ✅ **Distribution**: Heavy-tailed sensitivity (clear super-weights)
- ✅ **Consistency**: Similar patterns across gradient-based metrics
- ✅ **Hierarchy**: Clear layer-wise or component-wise patterns

### **Phase 3 Success Criteria**  
- ✅ **Separation**: ΔPPL(super) ≫ ΔPPL(random) ≫ ΔPPL(bottom)
- ✅ **Magnitude**: Super-weight perturbations cause ≥3x impact vs random
- ✅ **Robustness**: Results consistent across multiple perturbation types
- ✅ **Significance**: Performance degradation clearly above noise level

---

## 📈 **Next Steps After ESA Analysis**

### **Analysis Phase**
1. **Compare stability across metrics** - Which methods agree?
2. **Identify layer patterns** - Are certain layers more critical?
3. **Validate perturbation hierarchy** - Do super-weights consistently cause more damage?
4. **Cross-model comparison** - Are patterns consistent across architectures?

### **Research Extensions**
1. **Scaling studies** - How do patterns change with model size?
2. **Task-specific analysis** - Do different tasks show different patterns?
3. **Architectural studies** - MoE vs dense sensitivity differences
4. **Intervention studies** - Can you improve robustness by protecting critical weights?

---

**Ready to start your first ESA analysis?** 

Begin with this command:
```bash
python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --seeds 1337,123,999 \
  --stability-check \
  --dtype bf16 \
  --save-plots \
  --verbose \
  --out-dir outputs/esa/llama31_8b/first_analysis
```

Your excellent baseline results (1.03 perplexity, 99.4% accuracy) provide the perfect foundation for detecting even subtle sensitivity patterns! 🚀

---

*For additional help: Check `python esa_runner.py --help` or refer to the runbooks in `README_ESA_RUNBOOK*.md`*
