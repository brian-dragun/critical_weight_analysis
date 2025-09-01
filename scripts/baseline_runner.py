#!/usr/bin/env python3
# baseline_runner.py
"""
Baseline runner for Extreme Sensitivity Analysis (ESA)

Usage examples:
  # Smoke
  python scripts/baseline_runner.py \
    --model meta-llama/Llama-3.1-8B \
    --baseline smoke --datasets wikitext-2-raw-v1 \
    --context-lens 1024 --seeds 1337 123 999 \
    --dtype bf16 --batch-tokens 131072

  # Standard (multi-dataset, multi-context)
  python scripts/baseline_runner.py \
    --model mistralai/Mistral-7B-v0.3 \
    --baseline standard --datasets wikitext-103 c4 \
    --context-lens 1024 4096 \
    --seeds 1337 123 999 --dtype bf16 --batch-tokens 131072

  # Extended (adds zero-shot + long-context; MoE logging if present)
  python scripts/baseline_runner.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --baseline extended --datasets wikitext-103 openwebtext \
    --context-lens 1024 4096 32768 --long-context 8192 16384 32768 \
    --seeds 1337 123 999 --dtype bf16 --batch-tokens 131072 --eval-suites hellaswag piqa boolq arc_e arc_c winogrande
"""

from __future__ import annotations
import argparse, json, math, os, random, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# HF imports (assumes installed in your env)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ======== Config & Manifest ========

BASELINES = ("smoke", "standard", "extended")

@dataclass
class RunConfig:
    model_id: str
    revision: Optional[str]
    baseline: str
    datasets: List[str]
    context_lens: List[int]
    long_context: List[int]
    seeds: List[int]
    batch_tokens: int
    dtype: str
    eval_suites: List[str]
    output_dir: str

@dataclass
class RunManifest:
    # core identifiers
    model_id: str
    revision: Optional[str]
    dtype: str
    tokenizer_hash: Optional[str]
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]
    vocab_size: Optional[int]

    # run params
    baseline: str
    datasets: List[str]
    context_lens: List[int]
    long_context: List[int]
    seeds: List[int]
    batch_tokens: int
    eval_suites: List[str]
    date_utc: str
    git_sha: Optional[str] = None
    commit_dirty_flag: Optional[bool] = None

    # results (filled incrementally)
    results: Dict[str, Dict] = None
    throughput_tok_per_s: Dict[str, float] = None
    peak_vram_gb: float = None

# ======== Utils ========

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_dtype(dtype_str: str):
    if dtype_str.lower() == "bf16":
        return torch.bfloat16
    if dtype_str.lower() == "fp16":
        return torch.float16
    return torch.float32

def peak_vram_gb():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ======== Metrics (stubs; Copilot can flesh out) ========

def compute_basic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    logits: [B, T, V], labels: [B, T]
    Returns: loss, ppl, token_accuracy
    """
    # Reshape for cross-entropy: logits [B*T, V], labels [B*T]
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)
    labels_flat = labels.view(-1)
    
    # Compute cross-entropy loss (handles ignore_index=-100 automatically)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fn(logits_flat, labels_flat)
    
    # Compute token accuracy (excluding ignored tokens)
    with torch.no_grad():
        preds = torch.argmax(logits_flat, dim=-1)
        valid_mask = (labels_flat != -100)
        if valid_mask.sum() > 0:
            accuracy = (preds == labels_flat)[valid_mask].float().mean()
        else:
            accuracy = torch.tensor(0.0)
    
    # Compute perplexity
    perplexity = torch.exp(loss) if not torch.isnan(loss) else torch.tensor(float('inf'))
    
    return {
        "loss": float(loss.item()),
        "perplexity": float(perplexity.item()),
        "token_accuracy": float(accuracy.item())
    }

def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, n_bins: int = 20) -> float:
    """
    probs: [N] predicted confidence, correct: [N] 0/1
    """
    if len(probs) == 0:
        return 0.0
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)

def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    """
    Brier score: mean squared difference between predicted probabilities and actual outcomes
    """
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - correct) ** 2))

# ======== Data loading (stubs; Copilot should wire HF datasets) ========

def iter_dataset_tokens(dataset_name: str, tokenizer: AutoTokenizer, max_tokens: int, context_len: int):
    """
    Yields batches of tokenized chunks of length `context_len`.
    Simple implementation using dummy text for testing.
    """
    # For now, create dummy text to test the pipeline
    # TODO: Replace with actual dataset loading
    
    dummy_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for testing purposes.
    Machine learning and artificial intelligence are transforming the world as we know it.
    Natural language processing enables computers to understand and generate human language.
    Deep learning models like transformers have revolutionized many AI applications.
    """ * 100  # Repeat to get enough tokens
    
    # Tokenize the dummy text
    tokens = tokenizer.encode(dummy_text, add_special_tokens=False)
    
    # Create chunks of context_len + 1 (for input + target)
    chunk_size = context_len + 1
    num_tokens_yielded = 0
    
    for i in range(0, len(tokens) - chunk_size, chunk_size):
        if num_tokens_yielded >= max_tokens:
            break
            
        chunk = tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:
            # Convert to tensor and add batch dimension
            batch = torch.tensor([chunk], dtype=torch.long)  # [1, context_len + 1]
            yield batch
            num_tokens_yielded += context_len

# ======== Evaluation passes ========

def eval_one_pass(model, tokenizer, context_len: int, max_tokens: int, batch_tokens: int, compute_calibration: bool = False):
    """
    Streams through `max_tokens` tokens in chunks of `context_len`.
    Maintains effective tokens/step ≈ batch_tokens by adjusting micro-batch size.
    Returns metrics, and optionally calibration arrays.
    """
    device = model.device
    vocab_size = model.config.vocab_size
    tokens_processed = 0
    start = time.time()

    # Artifacts (Copilot should fill)
    logprob_hist = []
    entropy_hist = []
    topk_mass_hist = []

    all_conf = []
    all_correct = []

    # Dynamic micro-batch: floor(batch_tokens / context_len)
    micro_bsz = max(1, batch_tokens // context_len)

    # Initialize metrics accumulator
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # TODO: Copilot: real dataloader
    for batch in iter_dataset_tokens("DUMMY", tokenizer, max_tokens, context_len):
        # batch: [B, T]
        inputs = batch[:, :-1].to(device, non_blocking=True)
        labels = batch[:, 1:].to(device, non_blocking=True)

        with torch.no_grad():
            out = model(input_ids=inputs, use_cache=False)
            logits = out.logits  # [B, T, V]

        # Basic metrics
        batch_metrics = compute_basic_metrics(logits, labels)
        
        # Accumulate metrics (handle NaN values gracefully)
        if not math.isnan(batch_metrics.get("loss", float("nan"))):
            total_loss += batch_metrics["loss"]
        if not math.isnan(batch_metrics.get("token_accuracy", float("nan"))):
            total_correct += batch_metrics["token_accuracy"] * inputs.numel()
        total_tokens += inputs.numel()
        num_batches += 1

        if compute_calibration:
            # TODO: Copilot: compute softmax, confidences and correctness
            pass

        tokens_processed += inputs.numel()
        if tokens_processed >= max_tokens:
            break

    elapsed = max(1e-6, time.time() - start)
    tps = tokens_processed / elapsed

    # Compute final metrics
    avg_loss = total_loss / max(1, num_batches) if num_batches > 0 else float("nan")
    avg_accuracy = total_correct / max(1, total_tokens) if total_tokens > 0 else float("nan")
    perplexity = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")

    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": avg_accuracy,
        "tokens_evaluated": tokens_processed,
        "tok_per_sec": tps,
    }

    calibration = {}
    if compute_calibration and len(all_conf) > 0:
        conf = np.array(all_conf)
        corr = np.array(all_correct)
        calibration = {
            "ece": float(expected_calibration_error(conf, corr, n_bins=20)),
            "brier": float(brier_score(conf, corr)),
        }

    artifacts = {
        "logprob_hist_path": None,   # TODO: Copilot: save npy
        "entropy_hist_path": None,   # TODO: Copilot
        "topk_mass_hist_path": None, # TODO: Copilot
    }
    return results, calibration, artifacts, tps

# ======== MoE diagnostics (optional) ========

def maybe_log_moe_stats(model, out_dir: Path):
    """
    If the model is MoE (e.g., Mixtral), collect gate stats:
    - expert usage entropy
    - per-expert token share
    - gate margin histogram
    """
    # TODO: Copilot: detect Mixtral router modules and hook forward passes
    return None

# ======== CLI / Main ========

def parse_args():
    p = argparse.ArgumentParser(description="ESA Baseline Runner")
    p.add_argument("--model", required=True, help="HF model id (e.g., meta-llama/Llama-3.1-8B)")
    p.add_argument("--revision", default=None, help="Specific git commit or tag on HF hub")
    p.add_argument("--baseline", choices=BASELINES, default="standard")
    p.add_argument("--datasets", nargs="+", required=True, help="Datasets (HF ids). For smoke: use wikitext-2-raw-v1")
    p.add_argument("--context-lens", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--long-context", nargs="+", type=int, default=[], help="Only used in extended baseline")
    p.add_argument("--seeds", nargs="+", type=int, default=[1337, 123, 999])
    p.add_argument("--batch-tokens", type=int, default=131072, help="Target effective tokens/step")
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--eval-suites", nargs="+", default=[], help="Zero-shot suites for extended: hellaswag piqa boolq arc_e arc_c winogrande")
    p.add_argument("--output-dir", default="outputs/baselines")
    return p.parse_args()

def load_model_and_tokenizer(model_id: str, revision: Optional[str], dtype: str):
    torch_dtype = to_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # Prefer EOS over PAD if PAD missing
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return model, tok

def build_manifest(cfg: RunConfig, tok) -> RunManifest:
    return RunManifest(
        model_id=cfg.model_id,
        revision=cfg.revision,
        dtype=cfg.dtype,
        tokenizer_hash=getattr(tok, "init_kwargs", {}).get("hash", None),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        vocab_size=tok.vocab_size,
        baseline=cfg.baseline,
        datasets=cfg.datasets,
        context_lens=cfg.context_lens,
        long_context=cfg.long_context,
        seeds=cfg.seeds,
        batch_tokens=cfg.batch_tokens,
        eval_suites=cfg.eval_suites,
        date_utc=now_utc_iso(),
        results={},
        throughput_tok_per_s={},
        peak_vram_gb=0.0,
    )

def main():
    args = parse_args()
    cfg = RunConfig(
        model_id=args.model,
        revision=args.revision,
        baseline=args.baseline,
        datasets=args.datasets,
        context_lens=args.context_lens,
        long_context=args.long_context,
        seeds=args.seeds,
        batch_tokens=args.batch_tokens,
        dtype=args.dtype,
        eval_suites=args.eval_suites,
        output_dir=args.output_dir,
    )

    out_root = Path(cfg.output_dir) / cfg.model_id.replace("/", "__") / cfg.baseline
    ensure_dir(out_root)

    model, tok = load_model_and_tokenizer(cfg.model_id, cfg.revision, cfg.dtype)
    manifest = build_manifest(cfg, tok)

    # ===== baseline routing =====
    # Smoke: single small dataset + single context len; no calibration, minimal tokens
    if cfg.baseline == "smoke":
        ds = cfg.datasets[0]
        ctx = cfg.context_lens[0]
        total_tokens = 2000  # ~1k–2k tokens
        seed = cfg.seeds[0]

        set_global_seed(seed)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        results, calibration, artifacts, tps = eval_one_pass(
            model, tok, context_len=ctx, max_tokens=total_tokens,
            batch_tokens=cfg.batch_tokens, compute_calibration=False
        )

        manifest.results[f"{ds}@{ctx}"] = {
            "seed": seed, "metrics": results, "calibration": calibration, "artifacts": artifacts
        }
        manifest.throughput_tok_per_s[f"{ds}@{ctx}"] = tps
        manifest.peak_vram_gb = max(manifest.peak_vram_gb or 0.0, peak_vram_gb())

    # Standard: loop datasets × context_lens × seeds; add calibration & grad snapshot
    elif cfg.baseline == "standard":
        for ds in cfg.datasets:
            for ctx in cfg.context_lens:
                for seed in cfg.seeds:
                    set_global_seed(seed)
                    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

                    # Heuristic: evaluate up to ~1–5M tokens total per combo (tune as needed)
                    total_tokens = 500_000

                    results, calibration, artifacts, tps = eval_one_pass(
                        model, tok, context_len=ctx, max_tokens=total_tokens,
                        batch_tokens=cfg.batch_tokens, compute_calibration=True
                    )

                    # TODO: Copilot: one backward() on a micro-batch to snapshot grad norms per layer and save to npy
                    # grad_norms_path = save_grad_snapshot(model, tok, ctx, out_root)

                    key = f"{ds}@{ctx}@seed{seed}"
                    manifest.results[key] = {
                        "metrics": results,
                        "calibration": calibration,
                        "artifacts": artifacts,
                        # "grad_norms_path": grad_norms_path,
                    }
                    manifest.throughput_tok_per_s[key] = tps
                    manifest.peak_vram_gb = max(manifest.peak_vram_gb or 0.0, peak_vram_gb())

    # Extended: add zero-shot tasks, long-context, and (if MoE) router stats
    elif cfg.baseline == "extended":
        # 1) First run standard baseline evaluations for all datasets/contexts/seeds
        for ds in cfg.datasets:
            for ctx in cfg.context_lens:
                for seed in cfg.seeds:
                    set_global_seed(seed)
                    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

                    # Standard evaluation (same as standard baseline)
                    total_tokens = 500_000

                    results, calibration, artifacts, tps = eval_one_pass(
                        model, tok, context_len=ctx, max_tokens=total_tokens,
                        batch_tokens=cfg.batch_tokens, compute_calibration=True
                    )

                    key = f"{ds}@{ctx}@seed{seed}"
                    manifest.results[key] = {
                        "metrics": results,
                        "calibration": calibration,
                        "artifacts": artifacts,
                    }
                    manifest.throughput_tok_per_s[key] = tps
                    manifest.peak_vram_gb = max(manifest.peak_vram_gb or 0.0, peak_vram_gb())

        # 2) Long-context diagnostic
        for L in cfg.long_context:
            # TODO: Copilot: implement needle-in-a-haystack @ length L and record retrieval accuracy
            pass

        # 3) Zero-shot suites (HellaSwag, PIQA, etc.)
        if cfg.eval_suites:
            # TODO: Copilot: wire small eval loaders + exact-match/accuracy
            pass

        # 4) MoE diagnostics (if applicable)
        moe_stats = maybe_log_moe_stats(model, out_root)
        if moe_stats:
            manifest.results["moe_stats"] = moe_stats

    # ===== save manifests & summaries =====
    manifest_path = out_root / "manifest.json"
    save_json(asdict(manifest), manifest_path)

    # Lightweight summary for dashboards
    summary = {
        "model_id": cfg.model_id,
        "baseline": cfg.baseline,
        "best_ppl": _best_metric(manifest, "perplexity", minimize=True),
        "best_acc": _best_metric(manifest, "token_accuracy", minimize=False),
        "peak_vram_gb": manifest.peak_vram_gb,
        "avg_tok_per_sec": _avg_tps(manifest),
        "created": now_utc_iso(),
    }
    save_json(summary, out_root / "results_summary.json")

    print(f"[OK] Baseline complete. Manifest: {manifest_path}")

def _best_metric(manifest: RunManifest, name: str, minimize: bool) -> Optional[float]:
    vals = []
    for k, v in (manifest.results or {}).items():
        m = (v.get("metrics") or {}).get(name)
        if m is not None and not math.isnan(m):
            vals.append(m)
    if not vals: return None
    return (min(vals) if minimize else max(vals))

def _avg_tps(manifest: RunManifest) -> Optional[float]:
    vals = list((manifest.throughput_tok_per_s or {}).values())
    return float(sum(vals)/len(vals)) if vals else None

if __name__ == "__main__":
    main()
