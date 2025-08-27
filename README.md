# Critical Weight Analysis: Enhanced PhD Research System

A comprehensive research framework for analyzing weight sensitivity in transformer language models through gradient-based and non-gradient metrics, advanced perturbation experiments, and rigorous evaluation methodologies.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Research Ready](https://img.shields.io/badge/Research-Ready-purple.svg)](https://github.com/brian-dragun/critical_weight_analysis)

## 🎯 Research Overview

This system provides state-of-the-art tools for understanding which model weights are most critical for Large Language Model (LLM) performance. It combines gradient-based and non-gradient sensitivity metrics with sophisticated analysis techniques to provide comprehensive insights into transformer model behavior.

### **Phase 1 Complete Implementation:**
1. **Multi-Modal Sensitivity Discovery**: Gradient-based (grad×weight, grad², Hutchinson diagonal) and non-gradient (magnitude, activation-weighted) metrics
2. **Advanced Perturbation Testing**: Sign flip, Gaussian noise, bit flip, and zeroing experiments  
3. **Rigorous Control Baselines**: Random-K and bottom-K weight selection for statistical validation
4. **Stability Analysis**: Jaccard overlap computation across seeds and data batches
5. **Comprehensive Evaluation**: Perplexity, NLL, token accuracy with detailed performance tracking
6. **Full Reproducibility**: Experiment manifest logging with environment, git state, and configuration tracking

### **Key Capabilities:**
- ⚡ **Advanced Metrics**: Hutchinson diagonal estimator for curvature analysis
- 🎯 **Flexible Ranking**: Per-layer and global Top-K weight selection  
- 🔬 **Rigorous Controls**: Random and bottom-K baselines for statistical validation
- 📊 **Multi-Perturbation**: Sign flip, Gaussian noise, bit flip experiments
- 🔗 **Stability Testing**: Cross-seed and cross-batch Jaccard similarity analysis
- 📈 **Publication-Ready**: Comprehensive visualization suite and statistical analysis

## 🚀 Enhanced Features

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
- PyTorch 2.0+ with CUDA support
- Modern GPU (recommended for large models)
- 16GB+ GPU memory for 7B+ models

### Quick Setup with UV (Recommended)
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone repository
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -r setup/requirements.txt
```

### Traditional Setup
```bash
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis
python -m venv .venv
source .venv/bin/activate
pip install -r setup/requirements.txt
```

## 🎮 Quick Start

### Basic Usage
```bash
# Run comprehensive enhanced analysis
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --save-plots \
    --out-dir results/gpt2_analysis

# Advanced research workflow
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric hutchinson_diag \
    --topk 100 --mode global \
    --perturb sign_flip --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --save-plots \
    --out-dir outputs/llama31_8b_hutch_diag_k100
```

### Example Workflows

#### Quick System Validation (2-3 minutes)
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 50 \
    --max-samples 10 \
    --save-plots \
    --out-dir results/quick_validation
```

#### Comprehensive Model Analysis (10-30 minutes)
```bash
# Gradient-based analysis with perturbations
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
    --max-samples 50 \
    --save-plots \
    --out-dir results/llama31_comprehensive
```

#### Cross-Model Comparison Study
```bash
# Compare Llama vs Mistral architectures
for model in "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1"; do
    model_name=$(basename $model | sed 's/[^a-zA-Z0-9]/_/g')
    python phase1_runner_enhanced.py \
        --model $model \
        --metric grad_x_weight \
        --topk 300 \
        --mode per_layer \
        --perturb sign_flip \
        --controls random_k,bottom_k \
        --seeds 0,1,2 \
        --stability-check \
        --save-plots \
        --out-dir results/comparison/${model_name}
done
```

## 🔧 Command Line Options

### Core Parameters
```bash
--model MODEL_NAME          # HuggingFace model identifier
--metric METRIC              # Sensitivity metric: grad_x_weight, magnitude, hutchinson_diag, etc.
--topk K                     # Number of top weights to select
--mode MODE                  # Ranking mode: per_layer or global
--device DEVICE              # Device: cuda, cpu, or auto
```

### Advanced Options
```bash
--max-samples N              # Maximum evaluation samples
--max-length L               # Maximum sequence length
--perturb METHOD             # Perturbation method: sign_flip, gauss_noise, bit_flip, zero
--perturb-scale SCALE        # Perturbation intensity (default: 1.0)
--perturb-prob PROB          # Perturbation probability for bit_flip (default: 0.1)
--controls METHODS           # Control baselines: random_k,bottom_k
--seeds SEEDS                # Random seeds: 0,1,2
--stability-check            # Enable stability analysis
--save-plots                 # Generate visualization plots
--out-dir DIR                # Output directory
```

### Data and Evaluation
```bash
--data-file FILE             # Custom evaluation text file
--downstream-tasks           # Run downstream task evaluation
--task-samples N             # Samples for downstream tasks
--weight-analysis            # Enable advanced weight analysis
```

## 🏗️ Project Structure

```
critical_weight_analysis/
├── 📁 src/                          # Core source code
│   ├── sensitivity/                 # Sensitivity analysis modules
│   │   ├── metrics.py              # Sensitivity computation (6 metrics)
│   │   ├── rank.py                 # Weight ranking algorithms
│   │   └── perturb.py              # Perturbation methods
│   ├── eval/                       # Evaluation modules
│   │   ├── perplexity.py           # Core evaluation metrics
│   │   └── downstream.py           # Downstream task evaluation
│   ├── models/                     # Model handling
│   │   └── loader.py               # Unified model loading
│   ├── utils/                      # Utilities and visualization
│   │   ├── manifest.py             # Experiment tracking
│   │   ├── visualize.py            # Standard plotting
│   │   └── enhanced_visualize.py   # Advanced visualizations
│   └── data/                       # Evaluation datasets
├── 📁 scripts/                     # Automation scripts
│   ├── run_research_tests.sh       # Automated testing suite
│   └── generate_research_report.py # Results aggregation
├── 📁 setup/                       # Installation and configuration
│   ├── requirements.txt            # Python dependencies
│   └── install_uv.sh              # UV package manager setup
├── 📄 phase1_runner_enhanced.py    # Main research CLI
├── 📚 RESEARCH_TESTING_GUIDE.md    # Comprehensive research protocol
├── 📚 QUICK_RESEARCH_COMMANDS.md   # Quick reference commands
└── 📚 README.md                    # This file
```

## 🧪 Automated Testing

The system includes comprehensive automated testing for research validation:

```bash
# Run all tests (system validation + model analysis)
./scripts/run_research_tests.sh

# Run specific test phases
./scripts/run_research_tests.sh validation    # System compatibility
./scripts/run_research_tests.sh llama        # Llama model analysis
./scripts/run_research_tests.sh mistral      # Mistral model analysis
./scripts/run_research_tests.sh perturbation # Perturbation experiments
```

### Generated Test Outputs:
- **Validation Results**: System compatibility and basic functionality
- **Model Analysis**: Comprehensive sensitivity analysis across architectures
- **Perturbation Studies**: Performance degradation under weight modifications
- **Statistical Reports**: Automated summaries and visualizations

## 📊 Research Report Generation

Automatically aggregate and analyze all experimental results:

```bash
# Generate comprehensive research report
python scripts/generate_research_report.py \
    --input-dirs results/ \
    --output research_summary

# Outputs:
# 📄 research_summary.md - Complete analysis report
# 📊 research_summary_plots/ - Publication-ready visualizations
```

The report includes:
- **Executive Summary**: Overview of experiments and findings
- **Cross-Model Analysis**: Patterns across different architectures
- **Statistical Summaries**: Sensitivity distributions and trends
- **Visualization Suite**: Professional plots for publication

## 🔬 Research Applications

### Supported Models
- **GPT Family**: GPT-2, GPT-3.5, GPT-4 (via API)
- **Llama Series**: Llama-2-7B, Llama-3.1-8B, Code-Llama variants
- **Mistral Models**: Mistral-7B-v0.1, Mistral-7B-Instruct
- **Other Transformers**: Any HuggingFace compatible model

### Sensitivity Metrics
1. **grad_x_weight**: Gradient × weight magnitude
2. **grad_squared**: Squared gradient magnitude  
3. **hutchinson_diag**: Hessian diagonal approximation
4. **magnitude**: Absolute weight magnitude
5. **act_mag**: Activation-weighted magnitude
6. **random**: Random baseline for control

### Perturbation Methods
1. **sign_flip**: Flip weight signs
2. **gauss_noise**: Add Gaussian noise
3. **bit_flip**: Simulate hardware bit flips
4. **zero**: Set weights to zero

## 🎓 Academic Usage

### Expected Research Contributions
1. **Empirical Validation**: Super-weight hypothesis in modern LLMs
2. **Comparative Analysis**: Cross-architecture sensitivity patterns
3. **Methodological Framework**: Standardized sensitivity analysis protocols
4. **Robustness Foundations**: Basis for fault tolerance and compression research

### Publication-Ready Outputs
- **Statistical Analysis**: Rigorous experimental design with controls
- **Reproducibility**: Complete environment and configuration tracking
- **Visualization Suite**: Professional plots suitable for conference papers
- **Cross-Model Studies**: Comparative analysis across major architectures

### Typical Research Workflow
```bash
# Week 1: System validation and baseline establishment
./scripts/run_research_tests.sh validation

# Week 2-3: Comprehensive model analysis
python phase1_runner_enhanced.py --model meta-llama/Llama-3.1-8B --metric grad_x_weight --topk 500 --mode per_layer --save-plots --out-dir results/llama/analysis/
python phase1_runner_enhanced.py --model mistralai/Mistral-7B-v0.1 --metric grad_x_weight --topk 500 --mode per_layer --save-plots --out-dir results/mistral/analysis/

# Week 4: Cross-model comparison and perturbation studies
# [Multiple experiments with different configurations]

# Week 5: Report generation and analysis
python scripts/generate_research_report.py --input-dirs results/ --output final_research_summary
```

## 🛠️ Troubleshooting

### Common Issues

#### CUDA/GPU Problems
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If CUDA unavailable, install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Memory Issues
```bash
# Reduce memory usage for large models
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 100 \
    --max-samples 20 \
    --max-length 256 \
    --out-dir results/memory_optimized/
```

#### Model Access Issues
```bash
# For Llama models, ensure HuggingFace access
huggingface-cli login
# Follow prompts to enter your token
```

### Performance Optimization
- **Per-layer mode**: Faster than global ranking for large models
- **Magnitude metric**: Fastest option, no gradient computation required
- **Reduced samples**: Use `--max-samples 20` for faster iteration
- **GPU optimization**: Ensure CUDA-enabled PyTorch installation

## 📚 Documentation

- **[Research Testing Guide](RESEARCH_TESTING_GUIDE.md)**: Comprehensive research protocols
- **[Quick Commands](QUICK_RESEARCH_COMMANDS.md)**: Ready-to-run command reference
- **[Copilot Instructions](.github/copilot-instructions.md)**: Development guidelines

## 🤝 Contributing

This is a PhD research project focused on critical weight analysis in large language models. Contributions are welcome through:

1. **Issues**: Report bugs or suggest improvements
2. **Pull Requests**: Submit enhancements or fixes
3. **Research Collaboration**: Contact for academic partnerships

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Brian Dragun**  
PhD Student, Villanova University  
Critical Weight Analysis Research

- **GitHub**: [@brian-dragun](https://github.com/brian-dragun)
- **Email**: [research contact through GitHub]
- **Repository**: [critical_weight_analysis](https://github.com/brian-dragun/critical_weight_analysis)

## 📈 Citations

If you use this research framework in your work, please cite:

```bibtex
@software{dragun2025critical,
  title={Critical Weight Analysis: Enhanced PhD Research System},
  author={Dragun, Brian},
  year={2025},
  url={https://github.com/brian-dragun/critical_weight_analysis},
  note={PhD Research Framework for LLM Weight Sensitivity Analysis}
}
```

---

**🚀 Ready to start your research?** Check out the [Quick Start](#-quick-start) section or run the automated testing suite with `./scripts/run_research_tests.sh`!
