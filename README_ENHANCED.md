# Enhanced Critical Weight Analysis System

A comprehensive PhD research framework for analyzing critical weights in Large Language Models (LLMs) with advanced sensitivity metrics, perturbation methods, and evaluation capabilities.

## 🎯 Overview

This enhanced system provides state-of-the-art tools for understanding which model weights are most critical for LLM performance. It combines gradient-based and non-gradient sensitivity metrics with sophisticated analysis techniques to provide comprehensive insights into model behavior.

## 🚀 Enhanced Features (Latest Release)

### 1. Advanced Weight Analysis 📊
- **Weight Clustering**: Groups weights by sensitivity patterns using K-means/DBSCAN
- **Correlation Analysis**: Discovers relationships between layer sensitivities
- **Statistical Profiling**: Comprehensive statistics for sensitivity distributions

### 2. Temporal Stability Analysis ⏱️
- **Cross-Condition Tracking**: Monitor sensitivity changes across different conditions
- **Ranking Stability**: Measure consistency of critical weight rankings
- **Drift Detection**: Identify when models' critical regions shift over time

### 3. Architecture-Aware Analysis 🏗️
- **Component Classification**: Analyze embeddings, attention, feed-forward layers separately
- **Depth Pattern Analysis**: Understand sensitivity trends across model layers
- **Attention Component Comparison**: Compare query, key, value weight importance

### 4. Advanced Perturbation Methods ⚡
- **Progressive Perturbation**: Gradually increase perturbation severity
- **Clustered Perturbation**: Target weights based on sensitivity clusters
- **Targeted Perturbation**: Focus on specific architectural components

### 5. Downstream Task Evaluation 🎯
- **HellaSwag Evaluation**: Commonsense reasoning assessment
- **LAMBADA Evaluation**: Language modeling evaluation
- **Integrated Performance Tracking**: Connect sensitivity to real task performance

### 6. Enhanced Visualization Suite 📈
- **Publication-Ready Plots**: Professional visualizations for research papers
- **Interactive Analysis**: Multiple plot types for comprehensive understanding
- **Automated Report Generation**: Complete analysis summaries with plots

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Modern GPU (recommended for large models)

### Quick Setup with UV (Recommended)
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/critical_weight_analysis.git
cd critical_weight_analysis

# Install dependencies
uv add torch transformers datasets matplotlib seaborn scikit-learn scipy pandas tqdm
```

### Traditional Setup
```bash
pip install torch transformers datasets matplotlib seaborn scikit-learn scipy pandas tqdm
```

## 🎮 Quick Start

### Basic Usage
```bash
# Run comprehensive enhanced analysis
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --weight-analysis \
    --architecture-analysis \
    --temporal-stability \
    --downstream-tasks \
    --advanced-perturbations \
    --save-plots \
    --out-dir results/gpt2_enhanced
```

### Example Workflows
```bash
# Run pre-built workflow examples
python examples/enhanced_workflows.py

# Or execute specific workflow scripts
bash examples/run_enhanced_workflows.sh
```

## 📋 Command Line Options

### Enhanced Analysis Flags
- `--weight-analysis`: Enable weight clustering and correlation analysis
- `--architecture-analysis`: Analyze different architectural components
- `--temporal-stability`: Track sensitivity stability across conditions
- `--downstream-tasks`: Evaluate on HellaSwag and LAMBADA tasks
- `--advanced-perturbations`: Apply sophisticated perturbation methods

### Core Analysis Options
- `--model`: Model name or path (default: gpt2)
- `--metric`: Sensitivity metric (grad_x_weight, hutchinson_diag, magnitude, etc.)
- `--topk`: Number of top weights to analyze (default: 100)
- `--mode`: Ranking mode (global, layerwise, head_wise)

### Experimental Options
- `--perturb`: Perturbation method (sign_flip, magnitude_scale, gaussian_noise)
- `--perturb-scale`: Perturbation intensity
- `--controls`: Control baselines (random_k, bottom_k, all_k)
- `--seeds`: Multiple seeds for stability analysis

## 📊 Output Structure

```
results/
├── enhanced_analysis.json      # All enhanced analysis results
├── sensitivity_stats.json     # Basic sensitivity statistics
├── top_weights.csv            # Critical weight rankings
├── perturbation_results.json  # Perturbation experiment results
├── config.json                # Experiment configuration
├── manifest.json              # Complete file manifest
└── plots/                     # All generated visualizations
    ├── enhanced_weight_clustering.png
    ├── enhanced_architecture_analysis.png
    ├── enhanced_temporal_stability.png
    ├── enhanced_advanced_perturbations.png
    └── enhanced_downstream_tasks.png
```

## 🔬 Advanced Usage Examples

### 1. Large Model Analysis
```bash
# Analyze Llama with Hutchinson diagonal estimator
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric hutchinson_diag \
    --topk 500 \
    --mode global \
    --weight-analysis \
    --architecture-analysis \
    --out-dir results/llama_analysis
```

### 2. Multi-Condition Stability Analysis
```bash
# Analyze sensitivity stability across multiple conditions
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --temporal-stability \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --out-dir results/stability_analysis
```

### 3. Comprehensive Perturbation Study
```bash
# Run all perturbation methods with downstream evaluation
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 200 \
    --advanced-perturbations \
    --downstream-tasks \
    --perturb sign_flip \
    --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --out-dir results/perturbation_study
```

## 🧪 Testing and Validation

### Run All Tests
```bash
# Execute comprehensive test suite
bash tests/run_tests.sh
```

### Quick Validation
```bash
# Validate all enhanced features quickly
python validate_enhanced_features.py
```

### Individual Component Tests
```bash
# Test specific components
python -m pytest tests/test_integration.py::TestWeightAnalyzer -v
python -m pytest tests/test_integration.py::TestTemporalStabilityAnalyzer -v
python -m pytest tests/test_integration.py::TestArchitectureAnalyzer -v
```

## 📖 Research Applications

### Supported Research Questions

1. **Which weights are most critical for model performance?**
   - Use gradient-based metrics (grad_x_weight, hessian_diag)
   - Apply perturbation experiments to validate findings

2. **How do critical weights vary across model architectures?**
   - Use architecture-analysis to compare components
   - Analyze depth patterns across layers

3. **Are critical weight patterns stable across different inputs?**
   - Use temporal-stability analysis
   - Track rankings across multiple conditions

4. **How do critical weights impact downstream tasks?**
   - Use downstream-tasks evaluation
   - Connect sensitivity scores to task performance

5. **What perturbation strategies are most effective?**
   - Use advanced-perturbations with multiple methods
   - Compare progressive vs. clustered approaches

### Example Research Workflows

See `examples/enhanced_workflows.py` for complete implementations of:
- Basic sensitivity analysis
- Architecture comparison study
- Temporal stability investigation
- Perturbation effectiveness analysis
- Comprehensive model analysis

## 🔧 Customization and Extension

### Adding New Sensitivity Metrics
```python
# Extend src/sensitivity/metrics.py
def my_custom_metric(model, inputs, **kwargs):
    # Your metric implementation
    return sensitivity_scores
```

### Adding New Perturbation Methods
```python
# Extend src/sensitivity/advanced_perturb.py
class MyPerturbationMethod:
    def apply_perturbation(self, model, sensitivity_dict):
        # Your perturbation implementation
        return results
```

### Adding New Analysis Modules
```python
# Create new module in src/analysis/
class MyAnalyzer:
    def analyze(self, sensitivity_dict):
        # Your analysis implementation
        return results
```

## 📈 Performance Optimization

### For Large Models
- Use `--metric hutchinson_diag` for memory efficiency
- Set appropriate `--topk` values (100-1000)
- Use `--mode layerwise` for layer-specific analysis

### For Speed
- Use smaller `--topk` values for initial exploration
- Skip expensive analyses (`--no-downstream-tasks`)
- Use `--device cuda` for GPU acceleration

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/critical_weight_analysis.git
cd critical_weight_analysis

# Install development dependencies
uv add --dev pytest pytest-cov black flake8 mypy

# Run tests before committing
bash tests/run_tests.sh
```

### Code Structure
```
src/
├── analysis/           # Enhanced analysis modules
├── eval/              # Evaluation and downstream tasks
├── models/            # Model loading and utilities
├── sensitivity/       # Core sensitivity computation
└── utils/            # Visualization and result management
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@misc{enhanced_critical_weight_analysis,
  title={Enhanced Critical Weight Analysis for Large Language Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/critical_weight_analysis}
}
```

## 🎉 What's New in This Release

### Version 2.0 - Enhanced Analysis Suite

- ✅ **Advanced Weight Clustering**: Understand weight sensitivity patterns
- ✅ **Temporal Stability Tracking**: Monitor consistency across conditions
- ✅ **Architecture-Aware Analysis**: Component-specific insights
- ✅ **Advanced Perturbation Methods**: Sophisticated intervention strategies
- ✅ **Downstream Task Integration**: Real-world performance validation
- ✅ **Publication-Ready Visualizations**: Professional plots and reports
- ✅ **Comprehensive Test Suite**: Validated and reliable codebase
- ✅ **Example Workflows**: Ready-to-use research templates

### Previous Versions
- **Version 1.0**: Basic sensitivity analysis and perturbation experiments

## 🆘 Support

- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for research questions
- **Documentation**: Check the `docs/` directory for detailed guides

---

**Ready to discover what makes your models tick? Start with the enhanced critical weight analysis system today!** 🚀
