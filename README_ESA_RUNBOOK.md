ESA Runbook: Phase 1–3 Commands, Expectations, and Outcomes

This document outlines the commands for running **Extreme Sensitivity Analysis (ESA)** across **Phase 1 (Baseline)**, **Phase 2 (Sensitivity Profiling)**, and **Phase 3 (Perturbation Testing)**, including what each test is for, what outputs to expect, and what counts as good vs bad results.

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

## 1) Llama-3.1-8B baseline
```bash
python scripts/run_baseline.py \
  --model meta-llama/Llama-3.1-8B \
  --device cuda \
  --max-samples 200 \
  --batch-size 4 \
  --out-dir outputs/p1_baseline/llama31_8b
```
**Purpose:** Establish the unmodified model's **reference perplexity/loss**.  
**Outputs:** `experiment_manifest.json`, `config.json`, baseline metrics.  
**Good result:** Expected PPL (standard/extended): ~1.03–1.05; smoke: ~1.11. Token accuracy ~99% for standard/extended runs.  
**Bad result:** NaNs, unstable metrics.
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

### act_mag
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
  --perturb gauss_noise --perturb-scale 0.02 \
  --controls random_k,bottom_k \
  --seeds 0,1,2 \
  --stability-check \
  --device cuda \
  --save-plots \
  --out-dir outputs/p3/llama31_8b/gauss_0p02_k100
```
**Purpose:** Test noise robustness.  
**Good:** Super-weight damage ≫ controls.  
**Bad:** No separation from random.

## 8) Bit flip perturbation
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
**Purpose:** Test discrete perturbation effects.  
**Good:** Super-weight damage ≫ controls.  
**Bad:** No separation from random.

## 9) Hutchinson + sign flip
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
