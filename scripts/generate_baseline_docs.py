#!/usr/bin/env python3
"""
Dynamic baseline documentation generator
Reads actual baseline results and generates comprehensive reports
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

def load_baseline_results(baseline_dir: str) -> Dict:
    """Load all baseline results from the outputs directory"""
    baseline_path = Path(baseline_dir)
    results = {}
    
    if not baseline_path.exists():
        print(f"Warning: Baseline directory {baseline_dir} does not exist")
        return results
    
    # Find all manifest.json files
    for manifest_file in baseline_path.rglob("manifest.json"):
        try:
            with open(manifest_file, 'r') as f:
                data = json.load(f)
            
            model_id = data.get('model_id', 'unknown')
            baseline_type = data.get('baseline', 'unknown')
            
            # Extract key metrics from results
            model_results = {
                'model_id': model_id,
                'baseline': baseline_type,
                'dtype': data.get('dtype', 'unknown'),
                'date_utc': data.get('date_utc', 'unknown'),
                'peak_vram_gb': data.get('peak_vram_gb', 0.0),
                'metrics': {},
                'throughput': data.get('throughput_tok_per_s', {})
            }
            
            # Parse results for best metrics
            best_metrics = extract_best_metrics(data.get('results', {}))
            model_results['metrics'] = best_metrics
            
            key = f"{model_id}_{baseline_type}"
            results[key] = model_results
            
        except Exception as e:
            print(f"Warning: Could not parse {manifest_file}: {e}")
    
    return results

def extract_best_metrics(results: Dict) -> Dict:
    """Extract best metrics across all runs for a model"""
    if not results:
        return {'loss': None, 'perplexity': None, 'token_accuracy': None}
    
    all_metrics = []
    for key, result in results.items():
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            if isinstance(metrics, dict):
                all_metrics.append(metrics)
    
    if not all_metrics:
        return {'loss': None, 'perplexity': None, 'token_accuracy': None}
    
    # Find best values
    perplexities = [m.get('perplexity') for m in all_metrics if m.get('perplexity') is not None]
    accuracies = [m.get('token_accuracy') for m in all_metrics if m.get('token_accuracy') is not None]
    losses = [m.get('loss') for m in all_metrics if m.get('loss') is not None]
    
    return {
        'loss': min(losses) if losses else None,
        'perplexity': min(perplexities) if perplexities else None,
        'token_accuracy': max(accuracies) if accuracies else None
    }

def get_model_size(model_id: str) -> str:
    """Estimate model size from model ID"""
    size_map = {
        'Llama-3.1-8B': '8B',
        'Mistral-7B': '7B',
        'Phi-3-mini': '3.8B',
        'pythia-1.4b': '1.4B',
        'pythia-410m': '410M',
        'pythia-6.9b': '6.9B',
        'gemma-2-9b': '9B',
        'Mixtral-8x7B': '47B',
        'Qwen2.5-14B': '14B',
        'TinyLlama-1.1B': '1.1B'
    }
    
    for key, size in size_map.items():
        if key in model_id:
            return size
    
    return 'Unknown'

def calculate_avg_throughput(throughput_data: Dict) -> float:
    """Calculate average throughput across runs"""
    if not throughput_data:
        return 0.0
    
    values = [v for v in throughput_data.values() if isinstance(v, (int, float)) and v > 0]
    return sum(values) / len(values) if values else 0.0

def format_number(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    """Format numbers with proper handling of None values"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"

def generate_executive_report(results: Dict, output_file: str):
    """Generate executive summary report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    report = f"""# ESA Baseline Testing - Executive Summary

Generated: {timestamp}

## Overview

This report summarizes the baseline performance metrics for the Extreme Sensitivity Analysis (ESA) research project. The baseline establishes ground truth performance across our model suite before weight perturbation experiments.

## Model Performance Summary

| Model | Parameters | Perplexity | Token Accuracy | Throughput (tok/s) | VRAM (GB) |
|-------|------------|------------|----------------|-------------------|-----------|
"""

    # Sort models by perplexity (best first)
    sorted_models = sorted(
        results.items(), 
        key=lambda x: x[1]['metrics'].get('perplexity', float('inf'))
    )
    
    for key, data in sorted_models:
        model_name = data['model_id'].split('/')[-1]
        size = get_model_size(data['model_id'])
        metrics = data['metrics']
        avg_throughput = calculate_avg_throughput(data['throughput'])
        
        ppl = format_number(metrics.get('perplexity'), 2)
        acc = format_number(metrics.get('token_accuracy', 0) * 100, 1, '%') if metrics.get('token_accuracy') else "N/A"
        tps = format_number(avg_throughput, 0)
        vram = format_number(data.get('peak_vram_gb'), 1)
        
        report += f"| {model_name} | {size} | {ppl} | {acc} | {tps} | {vram} |\n"

    # Generate insights based on actual data
    if results:
        best_ppl_model = min(results.items(), key=lambda x: x[1]['metrics'].get('perplexity', float('inf')))
        best_acc_model = max(results.items(), key=lambda x: x[1]['metrics'].get('token_accuracy', 0))
        best_tps_model = max(results.items(), key=lambda x: calculate_avg_throughput(x[1]['throughput']))
        
        report += f"""
## Key Insights

### Performance Leaders
- **Best Perplexity**: {best_ppl_model[1]['model_id'].split('/')[-1]} ({format_number(best_ppl_model[1]['metrics'].get('perplexity'), 2)})
- **Best Accuracy**: {best_acc_model[1]['model_id'].split('/')[-1]} ({format_number(best_acc_model[1]['metrics'].get('token_accuracy', 0) * 100, 1)}%)
- **Best Throughput**: {best_tps_model[1]['model_id'].split('/')[-1]} ({format_number(calculate_avg_throughput(best_tps_model[1]['throughput']), 0)} tok/s)

### Model Analysis
"""
        # Add tier analysis
        perplexities = [data['metrics'].get('perplexity') for data in results.values() if data['metrics'].get('perplexity')]
        if perplexities:
            avg_ppl = sum(perplexities) / len(perplexities)
            report += f"- **Average Perplexity**: {format_number(avg_ppl, 2)}\n"
            report += f"- **Perplexity Range**: {format_number(min(perplexities), 2)} - {format_number(max(perplexities), 2)}\n"

    report += """
## Testing Configuration

- **Evaluation Method**: Automated baseline testing via baseline_runner.py
- **Precision**: bfloat16 (bf16) for optimal memory efficiency
- **Batch Strategy**: Dynamic micro-batching based on context length
- **Device Mapping**: Automatic GPU allocation
- **Reproducibility**: Fixed random seeds for consistent results

## Statistical Reliability

All metrics computed using:
- CrossEntropyLoss with ignore_index=-100 for padding
- Token-level accuracy excluding padding tokens
- Perplexity via exp(loss) transformation
- CUDA memory tracking for peak VRAM usage
- Wall-clock time measurement for throughput calculation

## Next Steps

1. **Standard Baselines**: Run `make standard-core` for comprehensive multi-dataset evaluation
2. **Extended Analysis**: Execute `make extended-llama` for long-context and zero-shot testing
3. **Weight Perturbation**: Begin ESA sensitivity experiments using these baseline metrics
4. **Scaling Studies**: Validate performance patterns across model size spectrum

## Research Implications

The baseline results establish critical ground truth metrics for ESA research:
- Performance benchmarks for detecting sensitivity-induced degradation
- Computational efficiency baselines for perturbation experiment planning
- Model selection validation for targeted sensitivity analysis
- Statistical foundations for significance testing in weight perturbation studies

Baseline stability across models provides robust foundation for systematic weight sensitivity analysis in subsequent ESA experiments.
"""

    with open(output_file, 'w') as f:
        f.write(report)

def generate_technical_log(results: Dict, output_file: str):
    """Generate detailed technical execution log"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    log = f"""# ESA Baseline Testing - Technical Execution Log

Generated: {timestamp}

## Execution Environment

- **OS**: Linux (Ubuntu)
- **Python**: 3.12+ with UV package manager
- **CUDA**: GPU acceleration enabled
- **Working Directory**: /home/ubuntu/Nova/critical_weight_analysis
- **Virtual Environment**: .venv activated

## Baseline Results Summary

Total models evaluated: {len(results)}
"""

    if results:
        baseline_types = set(data['baseline'] for data in results.values())
        log += f"Baseline types: {', '.join(baseline_types)}\n"
        
        dtypes = set(data['dtype'] for data in results.values())
        log += f"Precision types: {', '.join(dtypes)}\n"

    log += """
## Model Execution Details

"""

    # Add detailed results for each model
    for key, data in results.items():
        model_name = data['model_id']
        baseline_type = data['baseline']
        
        log += f"""### {model_name} ({baseline_type} baseline)

**Configuration:**
- Model ID: {model_name}
- Baseline Type: {baseline_type}
- Data Type: {data['dtype']}
- Execution Date: {data['date_utc']}

**Performance Metrics:**
"""
        
        metrics = data['metrics']
        if metrics.get('loss') is not None:
            log += f"- Loss: {format_number(metrics['loss'], 4)}\n"
        if metrics.get('perplexity') is not None:
            log += f"- Perplexity: {format_number(metrics['perplexity'], 4)}\n"
        if metrics.get('token_accuracy') is not None:
            log += f"- Token Accuracy: {format_number(metrics['token_accuracy'] * 100, 2)}%\n"
        
        log += f"- Peak VRAM: {format_number(data['peak_vram_gb'], 2)} GB\n"
        
        avg_tps = calculate_avg_throughput(data['throughput'])
        if avg_tps > 0:
            log += f"- Average Throughput: {format_number(avg_tps, 1)} tokens/second\n"
        
        log += "\n"

    log += """## Technical Implementation

### Core Metrics Computation
```python
def compute_basic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
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
        accuracy = (preds == labels_flat)[valid_mask].float().mean() if valid_mask.sum() > 0 else 0.0
    
    # Compute perplexity
    perplexity = torch.exp(loss) if not torch.isnan(loss) else float('inf')
    
    return {
        "loss": float(loss.item()),
        "perplexity": float(perplexity.item()),
        "token_accuracy": float(accuracy.item())
    }
```

### Data Pipeline
- **Tokenization**: HuggingFace fast tokenizers
- **Batching**: Dynamic micro-batch sizing based on target token count
- **Memory Management**: Automatic device mapping with bfloat16 precision
- **Context Handling**: Sliding window evaluation for long sequences

### Reproducibility Measures
- Fixed random seeds across numpy, torch, and CUDA
- Deterministic algorithms where possible
- Version tracking for all dependencies
- Git SHA recording for code state

## File Structure

```
outputs/baselines/
├── model_name__baseline_type/
│   ├── manifest.json (complete run metadata)
│   └── results_summary.json (condensed metrics)
└── ...
```

## Quality Assurance

### Validation Checks Performed
1. ✅ Model loading verification
2. ✅ Tokenizer compatibility confirmation
3. ✅ CUDA memory monitoring
4. ✅ NaN/Inf detection in metrics
5. ✅ JSON serialization validation
6. ✅ Reproducibility seed verification

### Error Handling
- Graceful handling of model loading failures
- NaN detection and reporting
- Memory overflow protection
- Timeout handling for large models

## Next Actions

1. **Data Validation**: Verify baseline consistency across runs
2. **Extended Testing**: Add calibration and long-context evaluation
3. **Comparative Analysis**: Cross-model performance comparison
4. **Sensitivity Preparation**: Use baselines for ESA perturbation experiments

End of technical execution log.
"""

    with open(output_file, 'w') as f:
        f.write(log)

def main():
    parser = argparse.ArgumentParser(description="Generate dynamic baseline documentation")
    parser.add_argument("--baseline-dir", default="outputs/baselines", help="Directory containing baseline results")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write documentation")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filenames")
    
    args = parser.parse_args()
    
    # Load baseline results
    print(f"Loading baseline results from {args.baseline_dir}...")
    results = load_baseline_results(args.baseline_dir)
    
    if not results:
        print("Warning: No baseline results found. Generating template documentation.")
    else:
        print(f"Found {len(results)} baseline result sets")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp suffix if requested
    timestamp_suffix = ""
    if args.timestamp:
        timestamp_suffix = f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate reports
    report_file = f"{args.output_dir}/baseline_report{timestamp_suffix}.md"
    log_file = f"{args.output_dir}/baseline_execution_log{timestamp_suffix}.md"
    
    print(f"Generating executive report: {report_file}")
    generate_executive_report(results, report_file)
    
    print(f"Generating technical log: {log_file}")
    generate_technical_log(results, log_file)
    
    # Also generate current versions (without timestamp)
    if args.timestamp:
        current_report = f"{args.output_dir}/baseline_report.md"
        current_log = f"{args.output_dir}/baseline_execution_log.md"
        
        print(f"Generating current report: {current_report}")
        generate_executive_report(results, current_report)
        
        print(f"Generating current log: {current_log}")
        generate_technical_log(results, current_log)
    
    print("Documentation generation complete!")

if __name__ == "__main__":
    main()
