ESA Runbook: Phase 1–3 Commands, Expectations, and Outcomes

This document outlines the commands for running **Extreme Sensitivity Analysis (ESA)** across **Phase 1 (Baseline)**, **Phase 2 (Sensitivity Profiling)**, and **Phase 3 (Perturbation Testing)**, including what each test is for, what outputs to expect, and what counts as good vs bad results.

---

# Phase 1 — Baseline (reference performance)

## 1) Llama-3.1-8B baseline
```bash
python scripts/run_baseline.py \
  --model meta-llama/Llama-3.1-8B \
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

## 3) grad×weight, global
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
**Purpose:** Find absolute heaviest hitters.  
**Good:** Overlap with per-layer hotspots.  
**Bad:** All top-k in one tensor due to bug.

## 4) Corroboration metrics
### grad_squared
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

### activation_magnitude
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric activation_magnitude \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p2/llama31_8b/actmag_perlayer_k100
```

### hutchinson_diag (curvature)
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

---

# Phase 3 — Perturbation Testing (causal validation)

## 5) Sign flip
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
**Purpose:** Flip sign of super weights.  
**Good:** ΔPPL ≫ controls, stable top-k overlap.  
**Bad:** No difference from random.

## 6) Zeroing
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
**Purpose:** Hard ablation.  
**Good:** Largest PPL jump.  
**Bad:** No measurable difference from controls.

## 7) Gaussian noise
```bash
python phase1_runner_enhanced.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --perturb gaussian --perturb-scale 0.02 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/gaussian_0p02_k100
```
**Purpose:** Test noise robustness.  
**Good:** Super-weight damage ≫ controls.  
**Bad:** No separation from random.

## 8) Hutchinson + sign flip
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
**Purpose:** Combine curvature ranking + perturbation.  
**Good:** Strongest validation if results align with grad-based runs.  
**Bad:** Weak/contradictory effects.

---

# What to look for
- **Good ESA signals:** ΔPPL (super ≫ random/bottom), non-trivial overlap across seeds (Jaccard > 0.3+), and metric agreement across grad × w, grad², and Hutchinson.  
- **Bad ESA signals:** No difference from controls, unstable rankings, or implausibly flat distributions.

---
