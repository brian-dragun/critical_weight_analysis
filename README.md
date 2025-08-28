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

## 📦 Installation & Setup

### System Requirements

#### Hardware Requirements
- **GPU**: Modern NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for Llama-7B+ models)
- **RAM**: 16GB+ system memory (32GB+ recommended for large models)
- **Storage**: 50GB+ free space (models and cache can be large)
- **CUDA**: CUDA 11.8+ or 12.x compatible GPU drivers

#### Software Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.8+ (3.11 or 3.12 recommended)
- **CUDA Drivers**: Compatible with PyTorch CUDA version
- **Git**: For repository management

### 🚀 Lambda Labs VM Setup (Recommended - 5 minutes)

If you're using Lambda Labs GPU cloud, use our optimized setup:

```bash
# Quick setup for new Lambda Labs VM
wget https://raw.githubusercontent.com/brian-dragun/critical_weight_analysis/master/setup/quick_vm_setup.sh
chmod +x quick_vm_setup.sh
./quick_vm_setup.sh

# Authenticate with HuggingFace (required for Llama models)
huggingface-cli login

# Test your setup
cd ~/nova/critical_weight_analysis
python scripts/quick_test.py
```

**What this does:**
- Updates system packages and installs build tools
- Installs UV package manager (10x faster than pip)
- Clones the repository to `~/nova/critical_weight_analysis`
- Sets up Python 3.12 virtual environment
- Installs PyTorch with CUDA support (tries 12.4, falls back to 12.1)
- Installs all research dependencies including GPU monitoring tools
- Configures shared cache directories in `/data/cache/`
- Sets up environment variables automatically
- Creates utility scripts for monitoring and testing

### 🔧 Manual Setup (Full Control)

#### Step 1: Install UV Package Manager (Recommended)
```bash
# Install UV (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
uv --version
```

#### Step 2: Clone Repository
```bash
# Create project directory
mkdir -p ~/nova
cd ~/nova

# Clone the repository
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis
```

#### Step 3: Create Virtual Environment
```bash
# With UV (recommended - much faster)
uv venv .venv --python 3.12
source .venv/bin/activate

# Or with standard Python
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

#### Step 4: Install PyTorch with CUDA
```bash
# For CUDA 12.x (most common)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 (if you have older drivers)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (not recommended for research)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 5: Install Research Dependencies
```bash
# Install all dependencies from requirements file
uv pip install -r setup/requirements.txt

# Or install manually step by step:

# Core ML libraries
uv pip install transformers>=4.55.0 datasets>=3.1.0 accelerate>=1.2.0
uv pip install huggingface_hub>=0.26.0 tokenizers>=0.20.0 safetensors>=0.4.0

# Data science stack
uv pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 scipy>=1.11.0

# GPU monitoring and system utilities
uv pip install gputil>=1.4.0 psutil>=5.9.0 pynvml>=11.5.0

# Research tracking tools
uv pip install wandb>=0.17.0 tensorboard>=2.15.0

# Development tools
uv pip install jupyter>=1.0 ipython>=7.0 pytest>=6.0
```

#### Step 6: Install Project in Development Mode
```bash
# Install the critical_weight_analysis package
uv pip install -e .
```

#### Step 7: Setup HuggingFace Authentication
```bash
# Required for accessing Llama models
huggingface-cli login

# Get your token from: https://huggingface.co/settings/tokens
# You may need to request access to Llama models specifically
```

#### Step 8: Configure Environment Variables
```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
export HF_HOME=/data/cache/hf
export HF_HUB_CACHE=/data/cache/hf/hub
export TRANSFORMERS_CACHE=/data/cache/hf/transformers
export DATASETS_CACHE=/data/cache/hf/datasets
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip
export PATH="$HOME/.local/bin:$PATH"

# For GPU optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Apply changes
source ~/.bashrc
```

### 🔍 Verify Installation

#### Basic System Check
```bash
# Test Python and packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test GPU access
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('No GPU available')
"
```

#### Comprehensive System Validation
```bash
# Run built-in diagnostics
python scripts/check_gpu.py
python scripts/quick_test.py

# Monitor system resources
python scripts/vm_monitor.py
```

#### Test Model Loading
```bash
# Quick model test (should complete in ~30 seconds)
python -c "
from src.models.loader import load_model
print('Testing model loading...')
model, tokenizer = load_model('gpt2', device='cuda')
print(f'✅ Model loaded: {type(model).__name__}')
print(f'✅ Device: {next(model.parameters()).device}')
"
```

### 🛠️ Alternative Setup Methods

#### Using Conda/Mamba
```bash
# Create conda environment
conda create -n critical_weights python=3.12
conda activate critical_weights

# Install PyTorch via conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies via pip
pip install -r setup/requirements.txt
pip install -e .
```

#### Using Docker (Advanced)
```bash
# For NVIDIA GPU support
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  -v /data/cache:/data/cache \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Then inside container:
cd /workspace
pip install -r setup/requirements.txt
pip install -e .
```

### 🚨 Troubleshooting Common Issues

#### CUDA Issues
```bash
# Check CUDA compatibility
nvidia-smi  # Should show driver version
nvcc --version  # Should show CUDA toolkit version

# Reinstall PyTorch if CUDA version mismatch
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Memory Issues
```bash
# Check available memory
free -h  # System memory
nvidia-smi  # GPU memory

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce model size for testing
python phase1_runner_enhanced.py --model gpt2 --max-samples 10
```

#### HuggingFace Access Issues
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login

# Test authentication
huggingface-cli whoami

# For Llama models, ensure you have access
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf
```

#### Package Conflicts
```bash
# Clean environment and reinstall
deactivate
rm -rf .venv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r setup/requirements.txt
```

### 📊 Performance Optimization

#### GPU Memory Optimization
```bash
# Set memory fraction for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use gradient checkpointing for large models
python phase1_runner_enhanced.py --model ... --gradient-checkpointing
```

#### Cache Configuration
```bash
# Create cache directories
sudo mkdir -p /data/cache/{hf,pip,torch}
sudo chown -R $USER:$USER /data/cache

# Verify cache setup
ls -la /data/cache/
```

### 🔄 Setup Automation Scripts

Your repository includes several automation scripts:

- **`setup/setup.sh`**: Complete automated setup for Linux/macOS
- **`setup/quick_vm_setup.sh`**: Rapid setup for new cloud VMs
- **`scripts/check_gpu.py`**: GPU diagnostics and validation
- **`scripts/quick_test.py`**: Comprehensive functionality testing
- **`scripts/vm_monitor.py`**: Real-time resource monitoring

Run the full automated setup:
```bash
chmod +x setup/setup.sh
./setup/setup.sh
```

## 🎮 Quick Start

### 🚀 Immediate Testing (2-3 minutes)

#### System Validation
```bash
# Test basic functionality with GPT-2
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 50 \
    --max-samples 10 \
    --save-plots \
    --out-dir results/quick_validation/gpt2_test
```

#### Llama-3.1-8B Quick Analysis (5-10 minutes)
```bash
# Fast per-layer analysis with gradient-based sensitivity
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 20 \
    --save-plots \
    --out-dir results/quick_research/llama31_8b_quick
```

#### Mistral-7B Quick Analysis (5-10 minutes)
```bash
# Fast per-layer analysis for comparison
python phase1_runner_enhanced.py \
    --model mistralai/Mistral-7B-v0.1 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --device cuda \
    --max-samples 20 \
    --save-plots \
    --out-dir results/quick_research/mistral_7b_quick
```

### 🔬 Complete Research Workflows

#### Advanced Perturbation Study
```bash
# Demonstrate perturbation effects with controls
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --device cuda \
    --max-samples 15 \
    --save-plots \
    --out-dir results/quick_research/perturbation_demo
```

#### Complete PhD Research Protocol
```bash
# Full Llama analysis with all features
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric hutchinson_diag \
    --topk 100 --mode global \
    --perturb sign_flip --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --save-plots \
    --out-dir outputs/llama31_8b_hutch_diag_k100 \
    --verbose
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

## � Output Structure

Each experiment creates a comprehensive output directory with all necessary files for reproducibility:

```
outputs/llama31_8b_hutch_diag_k100/
├── experiment_manifest.json          # Complete reproducibility info
├── config.json                       # Runtime configuration
├── sensitivity_stats.json            # Statistical summaries
├── top_weights.csv                   # Top-K weight selections
├── control_baselines.json            # Random/bottom-K controls
├── perturbation_results.json         # Perturbation experiment results
├── stability_results.json            # Jaccard stability analysis
└── plots/                            # Comprehensive visualizations
    ├── hutchinson_diag_k100_sensitivity_distribution.png
    ├── hutchinson_diag_k100_layer_comparison.png
    ├── hutchinson_diag_k100_sensitivity_heatmap.png
    ├── hutchinson_diag_k100_perturbation_effects.png
    └── hutchinson_diag_k100_stability_analysis.png
```

### Key Output Files:
- **`experiment_manifest.json`**: Complete environment tracking, git state, configuration
- **`top_weights.csv`**: Ranked weight importance scores with layer/parameter details
- **`sensitivity_stats.json`**: Statistical summaries of sensitivity distributions
- **`perturbation_results.json`**: Before/after performance metrics from perturbation tests
- **`plots/`**: Publication-ready visualizations for research papers

## �🔧 Command Line Options

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
│   ├── generate_research_report.py # Results aggregation
│   ├── vm_monitor.py               # Real-time resource monitoring
│   └── verify_restoration.py       # VM restoration validation
├── 📁 docs/                        # All documentation & guides
│   ├── RESEARCH_TESTING_GUIDE.md   # Comprehensive PhD research protocols
│   ├── QUICK_RESEARCH_COMMANDS.md  # Ready-to-run command reference
│   ├── USAGE_EXAMPLES.md           # Detailed examples and workflows
│   ├── STRUCTURE.md                # Project organization details
│   ├── INTEGRATION_GUIDE.md        # Integration with other projects
│   ├── MODEL_GUIDE.md              # Model compatibility guide
│   ├── LLAMA_RESEARCH_GUIDE.md     # LLaMA-specific research workflows
│   ├── MIGRATION_SUMMARY.md        # Shell → Python migration guide
│   ├── GITIGNORE_GUIDE.md          # Git configuration guide
│   ├── README_ORIGINAL_BACKUP.md   # Original README backup
│   └── research_summary.md         # Generated research reports
├── 📁 setup/                       # Installation and configuration
│   ├── requirements.txt            # Python dependencies
│   ├── setup.sh                    # Complete automated setup
│   └── quick_vm_setup.sh           # Rapid VM restoration script
├── � outputs/                     # ALL experiment results
├── �📄 phase1_runner_enhanced.py    # Main research CLI
└── � README.md                    # This comprehensive guide
```

### Key Directories:
- **`src/`**: Core analysis algorithms and utilities
- **`docs/`**: Complete documentation suite for research workflows
- **`scripts/`**: Automation tools for testing and report generation
- **`outputs/`**: All experimental results with date organization
- **`setup/`**: Installation and dependency management

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
./scripts/run_research_tests.sh multi-metric # Multi-metric comparison
```

### What the Automated Suite Does:

1. **Prerequisites Check**: Validates CUDA availability and GPU memory
2. **System Validation**: Tests basic functionality with small models
3. **Model Analysis**: Comprehensive sensitivity analysis across architectures  
4. **Perturbation Studies**: Performance degradation under weight modifications
5. **Statistical Reports**: Automated summaries and visualizations

### Expected Test Results:
- ✅ **Compatibility**: Models load successfully on GPU
- ✅ **Sensitivity Computation**: Completes without errors  
- ✅ **Output Generation**: Creates manifest.json, top_weights.csv, sensitivity_stats.json
- ⏱️ **Performance**: <2 minutes for small models, 5-15 minutes for large models
- 📊 **Memory Usage**: <80% of available GPU memory

### Generated Test Outputs:
- **Validation Results**: System compatibility and basic functionality
- **Model Analysis**: Comprehensive sensitivity analysis across architectures
- **Perturbation Studies**: Performance degradation under weight modifications
- **Statistical Reports**: Automated summaries and visualizations

## � VM Restoration & Backup

### Complete Environment Restoration

If you delete your VM and need to restore your research environment on a fresh Lambda Labs instance:

#### **🚀 Quick Restoration (5 minutes)**
```bash
# Download and run automated setup
wget https://raw.githubusercontent.com/brian-dragun/critical_weight_analysis/master/setup/quick_vm_setup.sh
chmod +x quick_vm_setup.sh
./quick_vm_setup.sh

# Authenticate with HuggingFace (required for Llama models)
huggingface-cli login

# Verify restoration worked
cd ~/nova/critical_weight_analysis
python scripts/verify_restoration.py
```

**What this restores:**
- ✅ Complete Python environment with all dependencies
- ✅ PyTorch with CUDA support (tries 12.4, falls back to 12.1)
- ✅ All research packages and GPU monitoring tools
- ✅ Project structure and automation scripts
- ✅ Environment variables and cache configuration
- ✅ Ready-to-run research system

#### **🔍 Manual Restoration (15 minutes)**
For complete control, follow the detailed manual setup in the [Installation](#-installation--setup) section above.

#### **✅ Verify Restoration**
```bash
# Comprehensive restoration verification
python scripts/verify_restoration.py

# Expected output: "🎉 VM restoration successful! Ready for research."
```

**Verification checks:**
- Package imports (transformers, torch, datasets, etc.)
- PyTorch and CUDA functionality
- GPU computation and memory access
- HuggingFace authentication
- Model loading (GPT-2 on GPU)
- Analysis pipeline (perplexity computation)
- All project files and scripts

#### **📋 Current Environment Snapshot**
Your working configuration (as of August 28, 2025):
- **PyTorch**: 2.5.1 with CUDA 12.4 support
- **GPU**: NVIDIA GH200 480GB working perfectly
- **HuggingFace**: Authenticated as bdragun
- **All Tests**: Passing (6/6) with analysis pipeline working
- **Key Features**: GPU computation, model loading, research workflows

### Backup Procedures

#### **Daily Backup Commands**
```bash
# Save current package versions
pip freeze > my_environment_backup.txt

# Commit and push code changes
git add . && git commit -m "Daily backup $(date)" && git push

# Save system state snapshot
python scripts/vm_monitor.py save backup_$(date +%Y%m%d).json
```

#### **Critical Configuration Backup**
```bash
# Save HuggingFace token (if needed)
cp ~/.cache/huggingface/token ~/hf_token_backup.txt

# Save environment configuration
cp ~/.bashrc ~/.bashrc.backup

# Export current environment
conda env export > environment.yml  # If using conda
# OR
pip freeze > requirements_backup.txt  # If using pip/uv
```

### Time Estimates
- **Quick restoration**: 5 minutes total
- **Manual restoration**: 15 minutes total
- **Verification**: 30 seconds
- **Ready for research**: Immediately after verification

### Restoration Validation
After restoration, you should be able to run:
```bash
# Basic validation
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5

# Advanced research
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --max-samples 20 \
    --save-plots
```

📚 **For detailed restoration procedures, see:** [`docs/VM_RESTORATION_CHECKLIST.md`](docs/VM_RESTORATION_CHECKLIST.md)

## �📊 Research Report Generation

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

### Quick Diagnostics
```bash
# Run comprehensive system check
python scripts/check_gpu.py
python scripts/quick_test.py
python scripts/vm_monitor.py
```

### Common Issues

#### CUDA/GPU Problems
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch CUDA version:', torch.version.cuda)"
nvidia-smi  # Check driver version

# If CUDA unavailable, reinstall PyTorch with correct CUDA version
uv pip uninstall torch torchvision torchaudio
# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8  
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Check system resources
python scripts/vm_monitor.py

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce memory usage for large models
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 100 \
    --max-samples 10 \
    --max-length 256 \
    --out-dir results/memory_optimized/
```

#### Model Access Issues
```bash
# Re-authenticate with HuggingFace
huggingface-cli logout
huggingface-cli login

# Test authentication
huggingface-cli whoami

# For Llama models, ensure you have access permission
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B
```

#### Package Installation Issues
```bash
# Clean environment and reinstall
deactivate
rm -rf .venv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r setup/requirements.txt
uv pip install -e .

# Alternative: Use conda instead of pip
conda create -n critical_weights python=3.12
conda activate critical_weights
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r setup/requirements.txt
```

#### Gradient Computation Issues
```bash
# If Hutchinson diagonal fails, use first-order methods
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --out-dir results/gradient_safe/

# Or use non-gradient methods
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --out-dir results/magnitude_only/
```

#### Long Execution Times
```bash
# For faster iteration during development
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric magnitude \
    --topk 100 \
    --max-samples 10 \
    --out-dir results/quick_test/

# Use per-layer mode (faster than global for large models)
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --mode per_layer \
    --topk 100 \
    --max-samples 20
```

#### Environment Variable Issues
```bash
# Ensure environment variables are set
echo $HF_HOME
echo $TRANSFORMERS_CACHE
echo $TORCH_HOME

# If not set, add to ~/.bashrc:
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf/transformers
export TORCH_HOME=/data/cache/torch
export PATH="$HOME/.local/bin:$PATH"
EOF

source ~/.bashrc
```

### Performance Optimization Tips
- **Per-layer mode**: Faster than global ranking for large models
- **Magnitude metric**: Fastest option, no gradient computation required  
- **Reduced samples**: Use `--max-samples 20` for faster iteration
- **GPU optimization**: Ensure CUDA-enabled PyTorch installation
- **Cache directories**: Use shared cache in `/data/cache/` to save bandwidth

### Setup Validation Commands
```bash
# Complete system validation
python scripts/check_gpu.py       # GPU diagnostics
python scripts/quick_test.py      # Functionality test
python scripts/vm_monitor.py      # Resource monitoring

# Test model loading
python -c "from src.models.loader import load_model; model, tokenizer = load_model('gpt2'); print('✅ Model loading works')"

# Test analysis pipeline
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5
```

### Time & Memory Estimates
- **Quick tests**: 2-10 minutes
- **Full per-layer analysis**: 10-30 minutes  
- **Global ranking**: 30 minutes - 2 hours
- **Llama-3.1-8B**: ~15-25GB GPU memory required
- **Mistral-7B**: ~12-20GB GPU memory required
- **Setup time**: 5 minutes (automated) to 15 minutes (manual)

### Getting Help
1. **Check logs**: Look in the generated output directories for error details
2. **Run diagnostics**: Use `python scripts/check_gpu.py` and `python scripts/quick_test.py`
3. **Monitor resources**: Use `python scripts/vm_monitor.py` to see system utilization
4. **Consult documentation**: See `docs/LAMBDA_LABS_SETUP.md` for detailed setup procedures
5. **Validate environment**: Ensure all dependencies are correctly installed

## 📚 Documentation

For more detailed information, see the comprehensive guides in the `docs/` folder:

### 📖 Quick References
- **[Quick Commands Reference](docs/QUICK_RESEARCH_COMMANDS.md)**: Ready-to-run command examples
- **[Usage Examples](docs/USAGE_EXAMPLES.md)**: Detailed examples and workflows
- **[Documentation Index](docs/INDEX.md)**: Complete guide to all documentation

### 🔬 Research Protocols  
- **[Research Testing Guide](docs/RESEARCH_TESTING_GUIDE.md)**: Complete PhD research protocols for Llama & Mistral
- **[Research Summary](docs/research_summary.md)**: Generated research reports and findings

### 🛠️ Setup & Configuration
- **[Lambda Labs Setup Guide](docs/LAMBDA_LABS_SETUP.md)**: Complete VM setup procedures and optimization
- **[VM Restoration Checklist](docs/VM_RESTORATION_CHECKLIST.md)**: Complete environment restoration procedures
- **[Project Structure](docs/STRUCTURE.md)**: Codebase organization and architecture
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)**: Integration with other projects
- **[Model Guide](docs/MODEL_GUIDE.md)**: Model compatibility and requirements

### 🚀 Advanced Topics
- **Llama Research Protocol**: Step-by-step analysis procedures in Research Testing Guide
- **Multi-Model Comparison**: Cross-architecture studies and best practices
- **Statistical Analysis**: Significance testing and validation methodologies
- **Publication Preparation**: Research paper figure generation and data export

### 💡 Development Resources
- **Setup Scripts**: Automated installation in `setup/` directory
- **Monitoring Tools**: Resource monitoring scripts in `scripts/` directory
- **Testing Suite**: Comprehensive validation tools and automated testing
- **Environment Configuration**: Cache setup, GPU optimization, and dependency management

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
