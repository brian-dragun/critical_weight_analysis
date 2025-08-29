# GitHub Copilot Instructions: Baseline Tests for ESA Research

## Context
This repo implements Extreme Sensitivity Analysis (ESA) on transformer models.
We need **baseline evaluations** before perturbations to ensure results are
valid and reproducible.

## Task for Copilot
Update `baseline_runner.py` (or create if missing) so that it can run and log
the following baseline tests for each Hugging Face model:

### 1. Smoke Test
- Dataset: small slice (1k–2k tokens) from WikiText-2 validation
- Metrics: loss, perplexity, token_accuracy
- Logs: tokenizer coverage, context length used, dtype, eval throughput

### 2. Standard Baseline
- Datasets: WikiText-103 validation + C4 validation (or OpenWebText2)
- Context lengths: 1k, 4k, (and longer if model supports)
- Metrics:
  - loss, perplexity, token_accuracy
  - calibration: Expected Calibration Error (ECE), Brier score
  - stability: variance over 3 random seeds
- Logs/artifacts:
  - logprob histograms
  - entropy distribution per position
  - top-k mass (e.g., mass@5)
  - activation stats (mean/std per layer)
  - gradient norms snapshot (1 backward pass, no update)
- Store results in `manifest.json` and `results_summary.json`

### 3. Extended Robustness Baseline
- Eval suites: HellaSwag, PIQA, BoolQ, Winogrande, ARC-E/C (zero-shot)
- For instruct models: 200-prompt mini set (greedy decode)
- Long-context test: "needle-in-a-haystack" retrieval at 8k/16k/32k
- For MoE models (e.g., Mixtral):
  - log expert usage entropy and per-expert token share
  - rerun with top-k routing perturbed (k→k-1) as control
- Optional: compare bf16 vs fp16, run weight-only int8 (bitsandbytes)

### 4. Reproducibility
- Freeze seeds: {1337, 123, 999} for RNG, CUDA, dataloader
- Log tokenizer hash, EOS/PAD IDs
- Batch size should target constant effective tokens (e.g., 128k tokens/step)

## Deliverables
Copilot should:
- Implement CLI options for `--baseline smoke|standard|extended`
- Save outputs into `outputs/baselines/{model_name}/{baseline_type}/`
- Record all metrics/artifacts into the experiment manifest

## Model Priority List for Testing
### Phase 1: Core Dense Models
1. meta-llama/Llama-3.1-8B
2. mistralai/Mistral-7B-v0.3
3. microsoft/Phi-3-mini-3.8B-instruct
4. EleutherAI/pythia-1.4b

### Phase 2: Scale Validation
5. meta-llama/Llama-3.1-70B
6. EleutherAI/pythia-410m
7. EleutherAI/pythia-6.9b
8. google/gemma-2-9b

### Phase 3: Architecture Diversity
9. mistralai/Mixtral-8x7B-v0.1
10. Qwen/Qwen2.5-14B
11. allenai/OLMo-2-1124-13B
12. TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Implementation Notes
- Use `datasets` library for data loading
- Implement numerically stable cross-entropy and accuracy
- Save artifacts as .npy files with clear naming
- Use bf16 where supported, fall back to fp16
- Implement proper error handling for OOM scenarios
- Log comprehensive metadata for reproducibility
