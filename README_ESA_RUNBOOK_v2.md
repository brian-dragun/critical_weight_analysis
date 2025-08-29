# ESA Runbook v2: Phase 1–3 Commands, Expectations, and Outcomes

This document outlines the commands for running **Extreme Sensitivity Analysis (ESA)** across **Phase 1 (Baseline)**, **Phase 2 (Sensitivity Profiling)**, and **Phase 3 (Perturbation Testing)**. ESA validates the "super-weight hypothesis" by identifying critical weights through gradient-based metrics and testing their causal importance via targeted perturbations.

**Research Goal:** Demonstrate that a small fraction of weights (typically <1%) are disproportionately important for model performance, with perturbations to these "super weights" causing significantly larger performance degradation than random or least-important weight modifications.

## CLI Choices Reference

**Valid --metric choices**: `grad_x_weight`, `grad_squared`, `hessian_diag`, `hutchinson_diag`, `magnitude`, `act_mag`  
**Valid --mode choices**: `per_layer`, `global`  
**Valid --perturb choices**: `zero`, `sign_flip`, `gauss_noise`, `bit_flip`  

## Baseline Performance Expectations (from latest reports)

| Model | Parameters | Expected PPL | Expected Accuracy |
|-------|------------|-------------|------------------|
| Llama-3.1-8B | 8.0B | 1.025 (std/ext), 1.11 (smoke) | 99.4% |
| Mistral-7B-v0.3 | 7.2B | 1.030 (std/ext), 1.14 (smoke) | 99.3% |
| Phi-3-mini | 3.8B | 1.028 (std/ext), 1.11 (smoke) | 99.3% |
| Mixtral-8x7B | 47B/13B | 1.029 (ext) | 99.5% |
| pythia-1.4b | 1.4B | 1.192 (ext), 1.20 (smoke) | 96.9% |

---

# Phase 1 — Baseline (reference performance)

**Phase Purpose:** Establish unmodified model performance as the reference point for all perturbation experiments. Critical for computing relative performance changes (ΔPPL).

## 1) Llama-3.1-8B baseline
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p1_baseline/llama31_8b
```
**Purpose:** Establish the unmodified model's **reference perplexity/loss** using the same metric as main experiments to ensure comparable sensitivity rankings.  
**Why grad_x_weight:** Matches Phase 2/3 methodology for consistent weight importance scoring.  
**Outputs:** `experiment_manifest.json`, `config.json`, baseline perplexity (Expected PPL: ~1.03–1.05 for standard/extended; ~1.11 for smoke), `top_weights.csv` for reference rankings.  
**Good result:** Stable perplexity baseline (±0.1 across runs), clear sensitivity distributions.  
**Bad result:** NaNs, unstable metrics, implausibly flat weight distributions.  
**Time estimate:** 5-10 minutes.

---

# Phase 2 — Sensitivity Profiling (rank "super weights")

**Phase Purpose:** Identify the most critical weights using multiple sensitivity metrics. The goal is to find heavy-tailed distributions where a small fraction of weights have disproportionately high sensitivity scores.

## 2) grad×weight, per-layer (Primary Method)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/gradxw_perlayer_k100
```
**Purpose:** Find **top-k weights per layer** using gradient×weight sensitivity.  
**Why this metric:** grad×weight captures both parameter magnitude and gradient influence, making it theoretically sound for identifying critical weights.  
**Why per_layer:** Ensures representation across all model components; prevents dominance by single layers.  
**Good:** Heavy-tailed distribution, clear hotspots in attention/MLP layers, consistent rankings across batches.  
**Bad:** Flat/uniform sensitivity, all top weights in embedding layers only.  
**Time estimate:** 10-15 minutes.pose:** Establish unmodified model performance as the reference point for all perturbation experiments. Critical for computing relative performance changes (ΔPPL).

## 1) Llama-3.1-8B baseline
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p1_baseline/llama31_8b
```
**Purpose:** Establish the unmodified model's **reference perplexity/loss** using the same metric as main experiments to ensure comparable sensitivity rankings.  
**Why grad_x_weight:** Matches Phase 2/3 methodology for consistent weight importance scoring.  
**Outputs:** `experiment_manifest.json`, `config.json`, baseline perplexity (Expected PPL: ~1.03–1.05 for standard/extended; ~1.11 for smoke), `top_weights.csv` for reference rankings.  
**Good result:** Stable perplexity baseline (±0.1 across runs), clear sensitivity distributions.  
**Bad result:** NaNs, unstable metrics, implausibly flat weight distributions.  
**Time estimate:** 5-10 minutes.1–3 Commands, Expectations, and Outcomes

This document outlines the commands for running **Extreme Sensitivity Analysis (ESA)** across **Phase 1 (Baseline)**, **Phase 2 (Sensitivity Profiling)**, and **Phase 3 (Perturbation Testing)**. ESA validates the "super-weight hypothesis" by identifying critical weights through gradient-based metrics and testing their causal importance via targeted perturbations.

**Research Goal:** Demonstrate that a small fraction of weights (typically <1%) are disproportionately important for model performance, with perturbations to these "super weights" causing significantly larger performance degradation than random or least-important weight modifications.

---

# Phase 1 — Baseline (reference performance)

## 1) Llama-3.1-8B baseline
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric magnitude \
  --device cuda \
  --max-samples 200 \
  --batch-size 4 \
  --out-dir outputs/p1_baseline/llama31_8b
```
**Purpose:** Establish the unmodified model’s **reference perplexity/loss**.  
**Outputs:** `experiment_manifest.json`, `config.json`, baseline metrics.  
**Good result:** Stable perplexity baseline.  
**Bad result:** NaNs, unstable metrics.

---

# Phase 2 — Sensitivity Profiling (rank “super weights”)

## 2) grad×weight, per-layer
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/gradxw_perlayer_k100
```
**Purpose:** Find **top-k weights per layer**.  
**Good:** Heavy-tailed distribution, clear hotspots.  
**Bad:** Flat/uniform sensitivity.

## 3) grad×weight, global (Validation Method)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode global \
  --topk 300 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/gradxw_global_k300
```
**Purpose:** Find absolute heaviest hitters across entire model for cross-validation.  
**Why global:** Tests if per-layer hotspots align with globally most critical weights.  
**Good:** Significant overlap (>50%) with per-layer hotspots, representation across multiple layers.  
**Bad:** All top-k concentrated in single tensor/layer (indicates bug or pathological case).  
**Time estimate:** 15-20 minutes.

## 4) Corroboration metrics (Multi-Method Validation)

**Purpose:** Validate grad×weight findings using orthogonal sensitivity measures. Strong ESA evidence requires agreement across multiple metrics.

### grad_squared (Gradient Magnitude)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_squared \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/gradsq_perlayer_k100
```
**Theory:** Weights with large gradients are actively being updated and likely important.  
**Expected overlap:** 60-80% with grad×weight top-k.  
**Time estimate:** 10-15 minutes.

### act_mag (Non-Gradient Method)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric act_mag \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/actmag_perlayer_k100
```
**Theory:** Weights that consistently produce large activations are functionally important.  
**Why important:** Provides gradient-free validation of critical weight identification.  
**Expected overlap:** 40-60% with grad×weight (lower overlap expected, validates different aspects).  
**Time estimate:** 8-12 minutes.

### hutchinson_diag (Curvature/Hessian)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric hutchinson_diag \
  --mode global \
  --topk 100 \
  --max-samples 120 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/hutchdiag_global_k100
```
**Theory:** High curvature (2nd derivative) indicates weights near critical points in loss landscape.  
**Why powerful:** Captures non-linear effects missed by 1st-order methods.  
**Computational cost:** Most expensive method due to Hessian approximation.  
**Expected overlap:** 50-70% with grad×weight if both capture true criticality.  
**Time estimate:** 20-30 minutes.

---

# Phase 3 — Perturbation Testing (causal validation)

**Phase Purpose:** Test the causal importance of identified "super weights" by perturbing them and measuring performance degradation. This is the critical validation step that distinguishes correlation from causation.

**Key Principle:** Super weights should cause significantly larger performance drops than random or bottom-k weights when perturbed. Statistical controls (random_k, bottom_k) are essential for rigorous validation.

## 5) Sign flip (Primary Causal Test)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb sign_flip --perturb-scale 1.0 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/signflip_k100
```
**Purpose:** Flip sign of super weights to test causal importance.  
**Why sign_flip:** Preserves weight magnitude while maximally disrupting gradient alignment.  
**Controls:** random_k (same number of random weights), bottom_k (least sensitive weights).  
**Good:** ΔPPL(super) ≫ ΔPPL(random) ≫ ΔPPL(bottom), stable top-k overlap across seeds (Jaccard >0.3).  
**Bad:** No significant difference from random controls, unstable rankings.  
**Time estimate:** 15-25 minutes.

## 6) Zeroing (Maximum Disruption Test)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb zero \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/zero_k100
```
**Purpose:** Hard ablation - completely remove super weights.  
**Why zeroing:** Maximum possible disruption, should show largest effects if super weights are truly critical.  
**Expected:** Largest PPL jump among all perturbation types.  
**Good:** Clear separation between super/random/bottom controls, stable effects.  
**Bad:** No measurable difference from controls, model collapse (infinite PPL).  
**Time estimate:** 15-25 minutes.

## 7) Gaussian noise (Robustness Test)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb gauss_noise --perturb-scale 0.02 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/gauss_0p02_k100
```
**Purpose:** Test noise robustness of super weights vs controls.  
**Why gauss_noise:** Models realistic weight corruption (hardware errors, quantization noise).  
**Scale choice:** 0.02 = 2% relative noise, strong enough to matter but not catastrophic.  
**Good:** Super-weight noise damage significantly exceeds random/bottom-k damage.  
**Bad:** No separation from random, excessive model degradation across all conditions.  
**Time estimate:** 15-25 minutes.

## 8) Bit flip perturbation (Discrete Test)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb bit_flip --perturb-prob 0.05 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/bitflip_p005_k100
```
**Purpose:** Test discrete perturbation effects on critical weights.  
**Why bit_flip:** Models hardware bit errors, quantization effects, discrete corruption scenarios.  
**Probability choice:** 0.05 = 5% chance per weight element, significant but localized damage.  
**Good:** Super-weight bit corruption significantly exceeds random/bottom-k damage.  
**Bad:** No separation from random, excessive model degradation across all conditions.  
**Time estimate:** 15-25 minutes.

## 9) Hutchinson + sign flip (Cross-Validation)
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric hutchinson_diag \
  --mode global \
  --topk 100 \
  --max-samples 120 \
  --perturb sign_flip --perturb-scale 1.0 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/hutch_global_signflip_k100
```
**Purpose:** Validate that curvature-based weight selection also shows causal effects.  
**Why important:** If Hutchinson and grad×weight identify different critical weights but both show strong perturbation effects, this strengthens the super-weight hypothesis.  
**Cross-validation:** Compare perturbation effects between grad×weight-selected and Hutchinson-selected weights.  
**Good:** Strong validation if results align with grad-based runs, complementary insights if different.  
**Bad:** Weak/contradictory effects, suggests method-specific artifacts rather than true weight criticality.  
**Time estimate:** 20-30 minutes.

---

# Results Interpretation & Success Criteria

## What to look for

### **Strong ESA Evidence (Supports Super-Weight Hypothesis):**
- **Large Effect Sizes:** ΔPPL(super) ≫ ΔPPL(random) ≫ ΔPPL(bottom), with effect size >2x
- **Statistical Significance:** Consistent effects across multiple seeds (p<0.05 via t-test)
- **Ranking Stability:** Non-trivial overlap across seeds (Jaccard similarity >0.3 for top-k)
- **Multi-Method Agreement:** Convergent evidence across grad×weight, grad², Hutchinson (overlap >50%)
- **Heavy-Tailed Distributions:** Clear power-law or exponential sensitivity distributions
- **Architectural Patterns:** Sensible weight locations (attention heads, key MLP neurons, not just embeddings)

### **Weak ESA Evidence (Challenges Super-Weight Hypothesis):**
- **No Control Separation:** Super weights perform no better than random controls
- **Unstable Rankings:** Low cross-seed overlap (<0.2 Jaccard), high variance in top-k selection
- **Method Disagreement:** Different metrics identify completely different critical weights
- **Flat Distributions:** Uniform or near-uniform sensitivity scores across all weights
- **Pathological Patterns:** All top weights in single layer/tensor, implausibly concentrated

### **Technical Issues (Require Investigation):**
- **Numerical Instability:** NaN values, infinite perplexity, gradient explosion
- **Implementation Bugs:** All top-k in single tensor, identical scores across weights
- **Computational Problems:** Memory errors, CUDA failures, extremely long runtimes

## Expected Timeline
- **Phase 1 (Baseline):** 5-10 minutes
- **Phase 2 (Sensitivity Profiling):** 1-2 hours total for all metrics
- **Phase 3 (Perturbation Testing):** 2-3 hours total for all perturbations
- **Total ESA Analysis:** 3-5 hours for complete Llama-3.1-8B characterization

## Publication-Ready Outputs
Each experiment generates comprehensive documentation for research papers:
- **Quantitative Results:** Statistical summaries, effect sizes, p-values
- **Visualizations:** Distribution plots, heatmaps, perturbation effect graphs
- **Reproducibility:** Complete experiment manifests with environment details
- **Controls:** Rigorous baseline comparisons for statistical validation

---

**Next Steps:** Run experiments in sequence, validate Phase 2 results before proceeding to Phase 3, cross-reference findings across multiple architectures (Mistral, GPT) for generalizability.
