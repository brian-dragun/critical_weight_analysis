# Combined Research + Developer + Explainer + Copilot-Generated Guide

This file consolidates all guidance for GitHub Copilot, developers, and research notes.

## 1. Copilot Instructions

# `.github/COPILOT_INSTRUCTIONS.md`

## 📌 Project Context
This repository is part of a PhD research project on **Large Language Model (LLM) robustness and sensitivity analysis**, focusing on **super-weight identification and prioritization** in transformer models.  
The code is being run on a **Lambda Labs GPU VM** (NVIDIA GH200 / A100 / H100 class hardware with CUDA 12.x, PyTorch 2.5+).  

The core Phase-1 workflow must:  
1. Compute **sensitivity metrics** (gradient-based + non-gradient heuristics).  
2. **Rank and select Top-K weights** (global or per-layer).  
3. Apply **perturbations** (zeroing, sign-flip, Gaussian noise, bit-flip).  
4. Compare against **controls** (random-K, bottom-K).  
5. Evaluate with **perplexity, NLL, token accuracy**, and later small downstream tasks.  
6. Ensure **reproducibility** (configs, fixed seeds, manifest logging).  
7. Generate **visualizations** (Δperplexity vs K, rank stability, layer histograms).  

This repository must stay focused on **Phase-1 sensitivity analysis**. Code should be modular, reproducible, and extendable to Phase-2 (robustness interventions like pruning/fault injection).  

---

## 📦 Code Style & Structure
- Language: Start with **Python 3.12** or downgrade to 
- Frameworks: **PyTorch, Hugging Face Transformers**  
- Style: **PEP8** with type hints where reasonable  
- Each metric/perturbation/control must be its own function in a dedicated module (`metrics.py`, `perturb.py`, `rank.py`).  
- Expose functionality via a **registry/dictionary** (e.g., `METRICS`, `PERTURBATIONS`) for easy extension.  
- Always include **docstrings** describing purpose, math formula, and parameters.  
- Unit-test edge cases (e.g., requires_grad=False, shared embeddings, dtype mismatches). 
- Always use GPU (CUDA) over CPU since we will have access to Lambda Labs GPU VMs
- Always write code in .py files
- Always use uv over pip to install and uninstall modules 

---

## 🚀 Required Features (Phase-1)

### Sensitivity Metrics
- ✅ Already: `grad_x_weight`, `grad_squared`  
- ➕ Add:  
  - `hutchinson_diag` (Hessian diagonal approximation using Hutchinson estimator)  
  - `magnitude` (|w|)  
  - `act_mag` (activation-weighted magnitude: E[|activation|]·|w|)  

### Ranking & Controls
- Support **global** and **per-layer Top-K**.  
- Add control selectors: `random_k`, `bottom_k`.  
- Implement rank stability: compute **Jaccard overlap** across seeds/batches.  

### Perturbations
- Perturbation types in `perturb.py`:  
  - `ZERO` (already in place)  
  - `SIGN_FLIP`  
  - `GAUSS_NOISE(scale)`  
  - `BIT_FLIP(prob)` (mask-based bit flips for fp16/int8).  

### Evaluation
- Extend `perplexity.py` to return:  
  - Perplexity  
  - Negative log-likelihood (NLL)  
  - Token accuracy (top-1)  

### Reproducibility
- All runs must save a **manifest.json** containing:  
  - Config parameters  
  - Random seeds  
  - Git SHA  
  - Python + library versions  
  - Model name/hash  
- Use Hydra/TOML configs where possible.  

### Visualization
- Minimal plots in `reports/`:  
  - Δperplexity vs K curve  
  - Layer-wise sensitivity histograms  
  - Rank-stability heatmaps  

---

## 🧑‍💻 Example Workflow

All features must support a **single CLI run** via `phase1_runner.py`:

```bash
python phase1_runner.py   model=meta-llama/Llama-3.1-8B   metric=hutchinson_diag   topk=100 mode=global   perturb=sign_flip perturb.scale=1.0   controls=random_k,bottom_k   seeds=0,1,2   out_dir=outputs/llama31_8b_hutch_diag_k100
```

---

## 🖥️ Lambda Labs Environment Notes
- VM: Lambda Labs GPU VM (e.g., NVIDIA GH200, A100, or H100).  
- CUDA: 12.x (Torch compiled with matching CUDA).  
- Mixed precision and GPU memory optimizations must be considered (`torch.autocast`, gradient checkpointing, device_map="auto").  
- Support **int8/4bit quantization** where possible (`bitsandbytes`) for larger models.  
- Default to GPU (`device = torch.device("cuda")`) but allow CPU fallback.  
- Keep performance in mind (avoid unnecessary `.cpu()` calls).  

---

## ✅ Best Practices for Copilot Suggestions
- Prefer **functional style** over long inline scripts.  
- Always integrate new methods into the existing **registry pattern** (don’t hardcode in `phase1_runner.py`).  
- Default to **device-agnostic code** (`device_map="auto"`, `.to(device)`).  
- Keep GPU safety in mind: mixed precision (`torch.autocast`), gradient checkpointing, optional int8/4bit loading.  
- Include **seed control** (`torch.manual_seed`, HF set_seed).  
- Provide **baseline & control comparisons** in outputs by default.  
- Always write **clear comments** tying code back to **research goal** (super-weight sensitivity).  

---

✅ With this file in `.github/`, GitHub Copilot will be nudged to generate **consistent, GPU-aware, research-aligned code** instead of generic boilerplate.  


## 2. Developer Quickstart Guide

# `DEV_GUIDE.md`

## Villanova PhD Research Project — LLM Sensitivity Analysis  
### Developer Quickstart (Phase 1)

---

## 📌 Project Overview
This project investigates **Large Language Model (LLM) robustness** by identifying and prioritizing **super-weights** (parameters that have outsized influence on model behavior).  

- **Phase 1 (current)**: Sensitivity analysis → weight ranking → perturbation experiments → evaluation.  
- **Phase 2 (future)**: Robustness interventions (pruning, fault injection, compression).  

All experiments are run on a **Lambda Labs GPU VM** (NVIDIA GH200/A100/H100, CUDA 12.x, PyTorch 2.5+).

---

## 🛠️ Setup
```bash
# Clone repo
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis

# Setup environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r setup/requirements.txt
```

Optional: For large models, install **bitsandbytes** for 8-bit/4-bit inference.  

---

## 🚀 Running Phase 1

### Example: Sensitivity with Hutchinson diag, perturb top-100, compare to random/bottom controls
```bash
python phase1_runner.py   model=meta-llama/Llama-3.1-8B   metric=hutchinson_diag   topk=100 mode=global   perturb=sign_flip perturb.scale=1.0   controls=random_k,bottom_k   seeds=0,1,2   out_dir=outputs/llama31_8b_hutch_diag_k100
```

### Outputs
- `outputs/.../rankings.csv` → weight scores, sorted.  
- `outputs/.../perturbation_results.json` → evaluation results.  
- `outputs/.../manifest.json` → run config, seeds, env, versions.  
- `reports/` → optional plots (Δperplexity vs K, histograms, stability).  

---

## 📦 Modules
- `sensitivity/metrics.py` → sensitivity metrics (grad×weight, grad², Hutchinson diag, magnitude, act_mag).  
- `sensitivity/rank.py` → Top-K ranking, random/bottom-K controls, overlap stability.  
- `perturb.py` → perturbation functions (zero, sign-flip, Gaussian noise, bit-flip).  
- `eval/perplexity.py` → evaluation (PPL, NLL, token accuracy).  
- `phase1_runner.py` → orchestrates a full experiment.  

---

## 🔍 Best Practices
- Always fix seeds (`--seeds=0,1,2`) for reproducibility.  
- Compare **Top-K vs Random-K vs Bottom-K** to show prioritization works.  
- Run small-scale (LLaMA-1B) first before scaling to large (LLaMA-8B+).  
- Use mixed precision (`torch.autocast`) to avoid OOM errors.  

---

## ✅ Next Steps
- Add more perturbation families.  
- Evaluate on small benchmark subsets (HellaSwag, LAMBADA).  
- Automate plotting for reports.  


## 3. Explainer Guide (Plain-English Research Notes)

# `EXPLAINER_GUIDE.md`

## Understanding Phase 1 — Plain-English Research Notes  

This guide explains what the code is doing, in simple terms, so you can describe it in your paper.  

---

### 1. **What is sensitivity analysis?**
Think of a model as billions of knobs (weights). Sensitivity analysis asks:  
➡️ *“If I nudge this knob, how much does the model’s output change?”*  

- **Gradient-based methods** (grad×weight, grad²): measure how steeply loss changes if you adjust a weight.  
- **Hutchinson diag**: estimates curvature (second derivative), showing how “fragile” a weight is.  
- **Magnitude / act_mag**: heuristic methods; sometimes big weights or highly-used ones are most important.  

This gives us a **score per weight**.

---

### 2. **Ranking and Top-K selection**
After scoring, we **sort weights** and keep the **Top-K** most “sensitive” ones.  
- **Global Top-K** → pick from entire model.  
- **Per-layer Top-K** → pick a fixed number per layer (ensures coverage).  

We then compare against **controls**:  
- Random-K (just random weights).  
- Bottom-K (least sensitive weights).  
➡️ This proves whether our method is meaningful or just noise.

---

### 3. **Perturbation experiments**
To test if these weights *really* matter, we mess with them and re-check performance.  
- Zero them out.  
- Flip their signs.  
- Add random noise.  
- Flip a random bit (simulating hardware faults).  

If performance (perplexity, accuracy) drops **more for Top-K than Random-K**, we know these weights are special.  

---

### 4. **Evaluation**
We measure model quality **before and after perturbation**.  
- **Perplexity (PPL):** how “surprised” the model is by test data. Lower = better.  
- **NLL:** average log-loss; another uncertainty measure.  
- **Token accuracy:** how often it predicts the right next token.  

ΔPPL or ΔNLL after perturbation = measure of sensitivity/robustness.  

---

### 5. **Why it matters for the PhD**
- Shows **super-weights exist** → small subset of parameters disproportionately control behavior.  
- Builds foundation for Phase 2: pruning, compression, or defending against bit-flip attacks.  
- Bridges **theory (gradients, curvature)** and **practical robustness** (fault tolerance, adversarial resilience).  

---

### 6. **How to write it in your paper**
You can frame Phase 1 as:  
- *“We designed a framework to systematically score and rank weights in LLMs. By perturbing prioritized vs. random weights, we demonstrate that a tiny fraction of weights (‘super-weights’) disproportionately affect performance.”*  
- Use plots: Δperplexity vs K, overlap stability.  
- Highlight novelty: comparing **gradient vs non-gradient families** of prioritization.  


# 4. Copilot Auto-Generated Instructions (Enhanced Workflow)

## GitHub Copilot Instructions: Critical Weight Analysis Research System

### Architecture Overview

This is a **PhD-level research system** for analyzing weight sensitivity in transformer language models. The architecture follows a **layered research pipeline**:

- **Main CLI**: `phase1_runner_enhanced.py` - Complete research workflow with 6 sensitivity metrics, 4 perturbation methods, control baselines, and stability analysis
- **Core Engine**: `src/sensitivity/` - Implements gradient-based (`grad_x_weight`, `grad_squared`, `hutchinson_diag`) and non-gradient (`magnitude`, `act_mag`) sensitivity metrics
- **Evaluation Layer**: `src/eval/` - Perplexity, NLL, token accuracy measurement with perturbation tracking
- **Reproducibility System**: `src/utils/manifest.py` - Full experiment logging with git state, environment, and configuration tracking

### Critical Developer Workflows

#### Environment Setup (uv-based)
```bash
# Modern package management - always use uv for this project
source .venv/bin/activate
uv pip install -r setup/requirements.txt
```

#### Research Execution Pattern
```bash
# The canonical PhD research command structure:
python phase1_runner_enhanced.py     --model [model]     --metric [grad_x_weight|hutchinson_diag|magnitude]     --topk 100 --mode [per_layer|global]     --perturb [sign_flip|gauss_noise|bit_flip]     --controls random_k,bottom_k     --seeds 0,1,2     --stability-check     --out-dir outputs/experiment_name
```

#### Data Sources
- **Default**: Built-in 10 ML sentences in `phase1_runner_enhanced.py:221-232`
- **Research**: Use `--data-file src/data/dev_small.txt` (52 ML/AI sentences)
- **Custom**: Any text file with one sentence per line

### Project-Specific Patterns

#### Sensitivity Metric Implementation
All metrics in `src/sensitivity/metrics.py` follow this pattern:
```python
def compute_X_sensitivity(model, param, param_name, input_ids, attention_mask, **kwargs):
    # Returns: torch.Tensor of same shape as param
```

#### Weight Ranking System (`src/sensitivity/rank.py`)
- **Per-layer mode**: Top-K within each layer independently
- **Global mode**: Top-K across entire model (flattened)
- Always preserves `(layer_name, param_name, flat_index)` tuple tracking

#### Perturbation Architecture (`src/sensitivity/perturb.py`)
```python
class WeightPerturber:
    # Applies perturbations while tracking original state
    # Methods: zero, sign_flip, gauss_noise, bit_flip
    # Always includes restore() for cleanup
```

#### Output Structure Convention
```
outputs/
└── [timestamp]/experiment_name/
    ├── manifest.json          # Full reproducibility tracking
    ├── results_summary.json   # Key metrics and rankings
    ├── sensitivity_scores.pt  # Raw tensor data
    └── plots/                 # Publication-ready visualizations
```

### Integration Points

#### Model Loading Pattern
```python
# Always use src/models/loader.py for consistency:
model, tokenizer = load_model_and_tokenizer(
    model_name, device=device, torch_dtype=torch.float32
)
```

#### Experiment Manifest Usage
```python
from src.utils.manifest import ExperimentManifest
manifest = ExperimentManifest("experiment_name", output_dir)
manifest.log_config(args)
manifest.log_results(results_dict)
manifest.save()  # Creates manifest.json
```

#### GPU Memory Management
- **Critical**: Always use `torch.cuda.empty_cache()` after large operations
- **Pattern**: Gradients computed on-demand, not accumulated
- **Hutchinson**: Uses random vectors, requires more memory than other metrics

### Testing & Debugging

#### Model Compatibility Testing
```bash
python model_compatibility_tester.py --model gpt2 --quick-test
```

#### Quick Development Testing
```bash
# Fast iteration cycle:
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5
```

#### Common Debug Points
1. **CUDA OOM**: Reduce `--topk` or `--max-samples`, check model size vs GPU memory
2. **Metric NaN**: Usually gradient computation on frozen parameters - check `requires_grad`
3. **Perturbation fails**: Verify weight tensor shapes match sensitivity tensor shapes

### Research-Specific Conventions

#### Metric Selection Strategy
- **Hutchinson diagonal**: Best for curvature analysis, memory-intensive
- **grad_x_weight**: Standard baseline, good balance of speed/accuracy  
- **magnitude**: Non-gradient baseline, fastest for large models

#### Statistical Validation
- **Always use controls**: `--controls random_k,bottom_k` for proper baselines
- **Stability checking**: `--seeds 0,1,2 --stability-check` computes Jaccard overlap
- **Multiple evaluations**: Each perturbation measured on fresh data batches

#### Visualization System
`src/utils/visualize.py` provides publication-ready plots:
- `plot_sensitivity_distribution()` - Core sensitivity analysis visualization
- `plot_perturbation_effects()` - Before/after performance comparison
- `plot_stability_analysis()` - Cross-seed stability metrics

### Code Navigation Shortcuts

- **Main research workflow**: `phase1_runner_enhanced.py:run_sensitivity_analysis()`
- **Hutchinson implementation**: `src/sensitivity/metrics.py:_compute_hutchinson_diagonal()`
- **Perturbation core**: `src/sensitivity/perturb.py:WeightPerturber.apply_perturbation()`
- **Ranking algorithms**: `src/sensitivity/rank.py:select_top_k_weights()`
- **Experiment tracking**: `src/utils/manifest.py:ExperimentManifest`
