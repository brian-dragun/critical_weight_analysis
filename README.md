# Critical Weight Analysis: Enhanced PhD Research Implementation

A comprehensive research system for analyzing weight sensitivity in transformer language models through gradient-based and non-gradient metrics, advanced perturbation experiments, and rigorous evaluation methodologies.

## 🎯 Enhanced Research Goals

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
- � **Rigorous Controls**: Random and bottom-K baselines for statistical validation
- 📊 **Multi-Perturbation**: Sign flip, Gaussian noise, bit flip experiments
- 🔗 **Stability Testing**: Cross-seed and cross-batch Jaccard similarity analysis
- 📈 **Publication-Ready**: Comprehensive visualization suite and statistical analysis

## 🚀 Enhanced PhD Research Workflow

### **Target Workflow (Now Implemented):**
```bash
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

### **Quick Start Examples:**
```bash
# Basic gradient analysis
python phase1_runner_enhanced.py --model gpt2 --metric grad_x_weight --topk 100

# Non-gradient magnitude analysis  
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 100 --save-plots

# Advanced perturbation study
python phase1_runner_enhanced.py --model gpt2 --metric grad_x_weight --topk 100 \
    --perturb sign_flip --controls random_k,bottom_k --stability-check
```

📋 **For comprehensive usage examples:** [`docs/USAGE_EXAMPLES.md`](docs/USAGE_EXAMPLES.md)

## 🏗️ Complete Project Structure

**📁 For detailed folder organization, see:** [`docs/STRUCTURE.md`](docs/STRUCTURE.md)

```
critical_weight_analysis/
├── 📋 docs/                           # All documentation & guides (THIS FOLDER)
│   ├── README.md                      # This comprehensive guide
│   ├── MIGRATION_SUMMARY.md           # Shell → Python migration details
│   ├── INTEGRATION_GUIDE.md           # How to connect with other research projects
│   ├── LLAMA_RESEARCH_GUIDE.md        # LLaMA-specific workflows
│   ├── MODEL_GUIDE.md                 # Model compatibility guide
│   ├── STRUCTURE.md                   # Complete folder organization guide
│   └── GITIGNORE_GUIDE.md             # Git configuration
├── 🔧 setup/                          # Setup & configuration files
│   ├── setup.sh                      # Environment setup script
│   ├── setup_llama.sh                # LLaMA-specific setup
│   ├── requirements.txt               # Python dependencies
│   └── pyproject.toml                 # Python package configuration
├── 📊 outputs/                        # ALL experiment results (consolidated)
│   ├── 2025-08-26/                   # Results organized by date
│   ├── 2025-08-27/                   # Recent experiment results
│   └── [DATE]/                       # Future results auto-organized
├── 🔬 src/                            # Core source code
│   ├── data/                         # Evaluation datasets
│   ├── models/                       # Model loading utilities
│   ├── sensitivity/                  # Sensitivity analysis algorithms
│   ├── eval/                         # Evaluation metrics
│   └── utils/                        # Common utilities
├── 📓 notebooks/                      # Jupyter research notebooks
├── 🧪 scripts/                        # Utility scripts
├── 🧪 test/                           # Test files
├── 📦 deprecated/                     # Legacy files (archived)
├── 🐍 **MAIN PYTHON MODULES**         # Core interface (in root)
│   ├── phase1_runner.py              # Core critical weight analysis
│   ├── model_compatibility_tester.py  # Model testing & validation
│   ├── integration_validator.py       # Integration testing
│   ├── llama_research_runner.py      # LLaMA-optimized workflows
│   └── research_bridge.py            # Research project integration
└── � README.md                      # Main project documentation (this file)
```

## � VM Environment Setup (Lambda Labs / Cloud GPU)

### **🔥 Quick VM Setup Guide with UV**

This section helps you recreate the entire development environment on a fresh Lambda Labs instance or similar cloud GPU provider using `uv` for blazing-fast package management.

#### **1. Initial VM Configuration:**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y git curl wget build-essential software-properties-common

# Install Python 3.12 (if not available)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python-is-python3

# Verify GPU setup
nvidia-smi
```

#### **2. Install UV (Ultra-Fast Python Package Manager):**

```bash
# Install uv (much faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell or add to PATH
source $HOME/.local/bin/env

# Verify uv installation
uv --version
```

#### **3. Clone Repository & Setup Project with UV:**

```bash
# Create working directory
mkdir -p /home/ubuntu/nova
cd /home/ubuntu/nova

# Clone the project repository
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis

# Create virtual environment with uv (much faster)
uv venv .venv --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv (10x faster than pip)
# Using latest PyTorch 2.8.0 with CUDA 12.8 (latest supported)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers datasets accelerate
uv pip install numpy pandas matplotlib seaborn jupyter
uv pip install huggingface_hub tokenizers
uv pip install tqdm psutil GPUtil

# Install development tools
uv pip install pytest black isort mypy

# Verify installation with GPU check
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Test core functionality
python scripts/quick_test.py
```

#### **4. CUDA Version Compatibility Check:**

```bash
# Check system CUDA version first
nvidia-smi

# The README uses CUDA 12.8, but you can adjust based on your system:
# For CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121  
# For CUDA 12.6: --index-url https://download.pytorch.org/whl/cu126
# For CUDA 12.8: --index-url https://download.pytorch.org/whl/cu128 (latest)
# For CPU only:  --index-url https://download.pytorch.org/whl/cpu

# Most Lambda Labs instances have CUDA 12.8, so cu128 is recommended
```

#### **5. Alternative: Use UV with pyproject.toml (Recommended for Reproducibility):**

```bash
# After cloning the repository
cd /home/ubuntu/nova/critical_weight_analysis

# Install project with uv (reads pyproject.toml automatically)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Or install from requirements if available
uv pip install -r setup/requirements.txt

# Verify installation
python scripts/quick_test.py
```

#### **6. GPU & CUDA Verification:**

```bash
# Comprehensive GPU check
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run project GPU diagnostic
python scripts/check_gpu.py
```

#### **7. Performance Benefits of UV:**

```bash
# Speed comparison (for reference):
# pip install torch transformers     # ~2-5 minutes
# uv pip install torch transformers  # ~30-60 seconds

# UV also provides better dependency resolution and caching
uv pip list  # View installed packages
uv cache clean  # Clean package cache if needed
```

#### **8. Generate Requirements for Future Deployments:**

```bash
# After installing all packages, export exact versions
uv pip freeze > requirements.txt

# Or create a more organized requirements structure
uv pip freeze | grep -E "torch|transformers|huggingface" > requirements-ml.txt
uv pip freeze | grep -E "numpy|pandas|matplotlib|seaborn" > requirements-data.txt
uv pip freeze | grep -E "pytest|black|isort|mypy" > requirements-dev.txt

# For next VM setup, just run:
# uv pip install -r requirements.txt
```

## 🔗 GitHub & HuggingFace Integration Setup

### **📦 GitHub Repository Setup**

#### **Option 1: Quick Setup Script (Recommended)**
```bash
# From the nova directory
cd /home/ubuntu/nova
./setup_github.sh
```

#### **Option 2: Manual GitHub Setup**
```bash
# Configure git (replace with your details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create SSH key for GitHub
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# Copy this key to GitHub SSH settings

# In your project directory
cd /home/ubuntu/nova/critical_weight_analysis

# Add remote (if not already added)
git remote add origin https://github.com/yourusername/critical_weight_analysis.git

# Create initial commit
git add .
git commit -m "Initial commit: Critical Weight Analysis Platform"
git push -u origin main
```

### **🤗 HuggingFace Hub Setup**

#### **Option 1: Automated Setup Script (Recommended)**
```bash
# From the nova directory
cd /home/ubuntu/nova
./setup_huggingface.sh
```

#### **Option 2: Manual HuggingFace Setup**
```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Login to HuggingFace (you'll need your token)
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens

# Create dataset repository
huggingface-cli repo create critical-weight-analysis-results --type dataset

# Upload results (after running experiments)
huggingface-cli upload critical-weight-analysis-results outputs/ --repo-type dataset
```

### **🚀 Complete Setup Verification**

```bash
# Test complete pipeline
python phase1_runner.py \
  --model gpt2 \
  --metric grad_x_weight \
  --topk 10 \
  --eval-limit 5 \
  --verbose

# Check outputs were created
ls -la outputs/

# Verify git status
git status

# Test HuggingFace connection
python -c "from huggingface_hub import HfApi; api = HfApi(); print('HF Connection:', api.whoami())"
```

## 🛠️ New Python Analysis Modules

### **🆕 Recently Added Python Tools** (Replaces Shell Scripts)

The project now includes powerful Python modules that replace legacy shell scripts with enhanced functionality:

#### **1. Model Compatibility Tester** (`model_compatibility_tester.py`)
Test model compatibility across different architectures and sizes:

```bash
# Test all model categories
python model_compatibility_tester.py --all

# Quick development testing  
python model_compatibility_tester.py --category small --quick

# Test specific model family
python model_compatibility_tester.py --category pythia --timeout 600

# Test single model
python model_compatibility_tester.py --model gpt2

# Export results for CI/automation
python model_compatibility_tester.py --category gpt2 --output ci_results/
```

**Features:**
- ✅ **Structured Categories**: GPT-2, Pythia, OPT, Code models, Small models
- 📊 **JSON Export**: Machine-readable results for automation
- 🚀 **Quick Mode**: Minimal parameters for CI/testing  
- 📈 **Progress Tracking**: Real-time status updates
- 🔧 **Error Classification**: Timeout vs. error vs. success analysis

#### **2. Integration Validator** (`integration_validator.py`)
Validates critical weight discoveries through perturbation testing:

```bash
# Auto-validate using latest critical weight results
python integration_validator.py --auto

# Use specific results directory
python integration_validator.py --from-results my_results/critical_analysis_20240827_143210/

# Test specific critical layers manually
python integration_validator.py --layer "gpt_neox.layers.2.mlp.dense_4h_to_h"

# Test multiple layers with custom LLM project path
python integration_validator.py --layers "gpt_neox.layers.2.mlp.dense_4h_to_h" "gpt_neox.layers.4.mlp.dense_4h_to_h" --llm-project "/path/to/llm_research_project"
```

**Features:**
- 🔍 **Auto-Discovery**: Automatically finds latest critical weight results
- 📈 **Correlation Analysis**: Validates sensitivity vs. perturbation impact
- 📋 **Comprehensive Reports**: Markdown reports with statistical analysis
- 🎯 **Flexible Testing**: Single layer, multiple layers, or auto-recommended
- 🔗 **Pandas Integration**: Advanced CSV handling and data analysis

#### **3. LLaMA Research Runner** (`llama_research_runner.py`)
Specialized workflow for LLaMA model analysis with memory optimization:

```bash
# Quick LLaMA 2 7B analysis (discovery only)
python llama_research_runner.py --model llama2-7b --discovery-only

# Full analysis with both discovery and validation
python llama_research_runner.py --model meta-llama/Llama-2-7b-hf --full-analysis

# Quick test for development
python llama_research_runner.py --model llama2-7b --quick

# Chat model analysis
python llama_research_runner.py --model llama2-7b-chat --discovery-only

# Code-specialized LLaMA analysis
python llama_research_runner.py --model code-llama-7b --discovery-only
```

**Features:**
- 🦙 **Model Shortcuts**: Easy names (llama2-7b, llama3-8b, code-llama-7b)
- 💾 **Memory Optimization**: Automatic parameter adjustment based on model size
- 📊 **Progress Tracking**: Real-time updates with time estimates
- 📄 **Specialized Reports**: LLaMA-specific insights and recommendations
- 🔧 **Multi-Mode**: Discovery-only, validation-only, full-analysis, quick

### **Migration from Shell Scripts**

The new Python modules replace the following legacy shell scripts:

| Old Shell Script | New Python Module | Status |
|------------------|-------------------|---------|
| `test_models.sh` | `model_compatibility_tester.py` | ✅ **Migrated** |
| `test_integration.sh` | `integration_validator.py` | ✅ **Migrated** |
| `llama_research.sh` | `llama_research_runner.py` | ✅ **Migrated** |

**Benefits of Migration:**
- 🔧 **Better Maintainability**: Structured code with classes and error handling
- 📊 **Enhanced Features**: JSON export, correlation analysis, progress tracking
- 🌍 **Cross-Platform**: Works on Windows, macOS, Linux
- 🔗 **Integration Ready**: Can be imported as modules in other scripts
- 📈 **Research Features**: Statistical analysis, automated reporting, memory optimization

For detailed migration information, see: [`MIGRATION_SUMMARY.md`](MIGRATION_SUMMARY.md)

### **📋 Environment Variables Setup**

Create a `.env` file for persistent configuration:

```bash
# Create environment file
cat > /home/ubuntu/nova/critical_weight_analysis/.env << 'EOF'
# HuggingFace Configuration
HF_TOKEN=your_huggingface_token_here
HF_HOME=/home/ubuntu/.cache/huggingface

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Project Configuration
PYTHONPATH=/home/ubuntu/nova/critical_weight_analysis/src
PROJECT_ROOT=/home/ubuntu/nova/critical_weight_analysis
EOF

# Load environment variables
source .env
```

### **🔄 Daily Workflow Commands (with UV)**

```bash
# Start of session
cd /home/ubuntu/nova/critical_weight_analysis
source .venv/bin/activate
source .env

# If you need to install new packages (super fast with UV)
uv pip install new-package-name

# Update requirements after adding packages
uv pip freeze > requirements.txt

# Run experiments
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100

# Commit results (including updated requirements)
git add outputs/ requirements.txt
git commit -m "Add analysis results for gpt2 and update requirements"
git push

# Share on HuggingFace
huggingface-cli upload critical-weight-analysis-results outputs/critical_analysis_* --repo-type dataset
```

### **⚡ Super Fast VM Recreation with UV**

```bash
# Complete setup in one command block (after system setup)
cd /home/ubuntu/nova && \
git clone https://github.com/brian-dragun/critical_weight_analysis.git && \
cd critical_weight_analysis && \
uv venv .venv --python 3.12 && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
python scripts/quick_test.py && \
echo "✅ Ready for research in ~2 minutes!"
```

## �🚀 Installation & Setup

### **Prerequisites:**
- NVIDIA GPU with CUDA support (tested on GH200 480GB)
- Python 3.12+
- Linux environment (Ubuntu recommended)

### **Complete Setup:**
```bash
# 1. Navigate to project directory
cd /home/ubuntu/nova/critical_weight_analysis

# 2. Run automated setup (installs everything)
./setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Verify installation
python scripts/quick_test.py
```

## 🔬 Usage Guide

### **🎯 Primary Interface: CLI Runner**

The `phase1_runner.py` script is your main interface for comprehensive analysis:

```bash
# Basic sensitivity analysis
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100

# Multi-metric comprehensive analysis
python phase1_runner.py \
  --model EleutherAI/pythia-2.8b \
  --metric grad_x_weight grad_squared \
  --topk 50 100 500 \
  --eval-limit 100 \
  --output my_results/

# Fast prototyping (quick results)
python phase1_runner.py \
  --model gpt2 \
  --metric grad_x_weight \
  --topk 10 \
  --eval-limit 10 \
  --fast-perturbation \
  --verbose
```

## 🤖 Model Compatibility Guide

### **✅ Fully Tested & Optimized Models:**

#### **GPT-2 Family** (OpenAI)
```bash
python phase1_runner.py --model gpt2              # 124M parameters
python phase1_runner.py --model gpt2-medium       # 355M parameters  
python phase1_runner.py --model gpt2-large        # 774M parameters
python phase1_runner.py --model gpt2-xl           # 1.5B parameters
```

#### **Pythia Series** (EleutherAI) - Recommended for Research
```bash
python phase1_runner.py --model EleutherAI/pythia-70m    # 70M parameters
python phase1_runner.py --model EleutherAI/pythia-160m   # 160M parameters
python phase1_runner.py --model EleutherAI/pythia-410m   # 410M parameters
python phase1_runner.py --model EleutherAI/pythia-1b     # 1B parameters
python phase1_runner.py --model EleutherAI/pythia-1.4b   # 1.4B parameters
python phase1_runner.py --model EleutherAI/pythia-2.8b   # 2.8B parameters
python phase1_runner.py --model EleutherAI/pythia-6.9b   # 6.9B parameters ⚠️
python phase1_runner.py --model EleutherAI/pythia-12b    # 12B parameters ⚠️
```

### **� Research-Grade Models:**

#### **LLaMA 2 Series** (Meta) - State-of-the-Art
```bash
# Requires HF token and approval
python phase1_runner.py --model meta-llama/Llama-2-7b-hf   # 7B parameters
python phase1_runner.py --model meta-llama/Llama-2-13b-hf  # 13B parameters ⚠️
python phase1_runner.py --model meta-llama/Llama-2-70b-hf  # 70B parameters ⚠️⚠️
```

#### **Mistral Series** (Mistral AI)
```bash
python phase1_runner.py --model mistralai/Mistral-7B-v0.1          # 7B parameters
python phase1_runner.py --model mistralai/Mistral-7B-Instruct-v0.1 # 7B parameters (instruct)
python phase1_runner.py --model mistralai/Mixtral-8x7B-v0.1        # 8x7B MoE ⚠️⚠️
```

#### **Code-Specialized Models**
```bash
python phase1_runner.py --model Salesforce/codegen-350M-mono   # 350M parameters (Python)
python phase1_runner.py --model Salesforce/codegen-2B-mono     # 2B parameters (Python)
python phase1_runner.py --model bigcode/starcoder              # 15B parameters (Multi-lang) ⚠️
python phase1_runner.py --model WizardLM/WizardCoder-15B-V1.0  # 15B parameters ⚠️
```

### **🧪 Experimental & Specialized Models:**

#### **OPT Series** (Meta/Facebook)
```bash
python phase1_runner.py --model facebook/opt-125m    # 125M parameters
python phase1_runner.py --model facebook/opt-350m    # 350M parameters
python phase1_runner.py --model facebook/opt-1.3b    # 1.3B parameters
python phase1_runner.py --model facebook/opt-2.7b    # 2.7B parameters
python phase1_runner.py --model facebook/opt-6.7b    # 6.7B parameters ⚠️
python phase1_runner.py --model facebook/opt-13b     # 13B parameters ⚠️
```

#### **EleutherAI Research Models**
```bash
python phase1_runner.py --model EleutherAI/gpt-j-6b     # 6B parameters ⚠️
python phase1_runner.py --model EleutherAI/gpt-neox-20b # 20B parameters ⚠️⚠️
```

#### **Smaller Research Models** (Fast Prototyping)
```bash
python phase1_runner.py --model distilgpt2                    # 82M parameters
python phase1_runner.py --model microsoft/DialoGPT-small      # 117M parameters
python phase1_runner.py --model roneneldan/TinyStories-1M     # 1M parameters (tiny)
python phase1_runner.py --model roneneldan/TinyStories-8M     # 8M parameters (tiny)
```

### **⚠️ Memory Requirements & Recommendations:**

#### **Model Size Guide:**
| Model Size | GPU Memory | Recommended Settings | Analysis Time |
|------------|------------|---------------------|---------------|
| **< 500M** | 8GB+ | `--eval-limit 100` | 1-5 minutes |
| **500M-2B** | 16GB+ | `--eval-limit 50` | 5-15 minutes |
| **2B-7B** | 24GB+ | `--eval-limit 30` | 15-30 minutes |
| **7B-15B** | 40GB+ | `--eval-limit 20 --fast-perturbation` | 30-60 minutes |
| **15B+** | 80GB+ | `--eval-limit 10 --no-perturbation` | 1+ hours |

#### **Performance Optimization by Model Size:**

**Small Models (< 1B)** - Full Analysis:
```bash
python phase1_runner.py \
  --model EleutherAI/pythia-410m \
  --metric grad_x_weight grad_squared \
  --topk 100 500 1000 \
  --eval-limit 100
```

**Medium Models (1B-7B)** - Balanced Analysis:
```bash
python phase1_runner.py \
  --model EleutherAI/pythia-2.8b \
  --metric grad_x_weight \
  --topk 50 100 500 \
  --eval-limit 50 \
  --fast-perturbation
```

**Large Models (7B+)** - Sensitivity Only:
```bash
python phase1_runner.py \
  --model meta-llama/Llama-2-7b-hf \
  --metric grad_x_weight \
  --topk 100 \
  --eval-limit 20 \
  --no-perturbation
```

### **🔧 Model-Specific Notes:**

#### **LLaMA Models:**
- Require Hugging Face account and model approval
- Set `HF_TOKEN` environment variable for access
- Use `--eval-limit 20` for 7B+ models

#### **Pythia Models:**
- **Best for research**: Consistent training data across sizes
- **Recommended**: Start with `pythia-410m` for prototyping
- **Production**: Use `pythia-2.8b` for comprehensive analysis

#### **Code Models:**
- May have different tokenization patterns
- Consider domain-specific evaluation texts
- Often more sensitive to certain weight perturbations

#### **Instruction-Tuned Models:**
- May show different sensitivity patterns than base models
- Compare with base versions for instruction-tuning analysis
- Use appropriate prompting in evaluation texts

### **🚀 Multi-Model Comparison Workflows:**

#### **Cross-Size Analysis** (Same Architecture):
```bash
# Compare sensitivity across Pythia model sizes
for model in pythia-70m pythia-410m pythia-1.4b pythia-2.8b; do
  python phase1_runner.py \
    --model EleutherAI/$model \
    --metric grad_x_weight \
    --topk 100 \
    --output comparison_$model/
done
```

#### **Cross-Architecture Analysis**:
```bash
# Compare different model families
python phase1_runner.py --model gpt2 --topk 100 --output gpt2_analysis/
python phase1_runner.py --model EleutherAI/pythia-410m --topk 100 --output pythia_analysis/
python phase1_runner.py --model facebook/opt-350m --topk 100 --output opt_analysis/
```

#### **Code vs Language Models**:
```bash
# Compare code-specialized vs general language models
python phase1_runner.py --model gpt2 --topk 100 --output general_model/
python phase1_runner.py --model Salesforce/codegen-350M-mono --topk 100 --output code_model/
```

### **🎯 Research Applications by Model Type:**

#### **Scaling Laws Research:**
Use Pythia series (70M → 12B) to study how critical weight patterns change with scale

#### **Architecture Comparison:**
Compare GPT-2, OPT, LLaMA to understand architectural sensitivity differences

#### **Domain Specialization:**
Analyze code models vs general models to understand specialization effects

#### **Instruction Tuning Impact:**
Compare base models with their instruction-tuned variants

### **⚠️ Important Considerations:**

#### **License Requirements:**
- **LLaMA**: Custom license, requires approval
- **Pythia/GPT-J/GPT-NeoX**: Apache 2.0 (research-friendly)
- **GPT-2**: MIT (most permissive)
- **OPT**: Custom license (check terms)

#### **Computational Limits:**
- **70B+ models**: May require model sharding or quantization
- **Memory optimization**: Use `--no-perturbation` for discovery-only analysis
- **Time limits**: Large models can take hours for full analysis

#### **Research Ethics:**
- Always cite model creators and training data sources
- Be aware of potential biases in sensitivity patterns
- Consider computational carbon footprint for large-scale studies

#### **Model Settings:**
- `--model`: Model name from Hugging Face Hub or local path
  - **GPT Family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
  - **Pythia Series**: `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`, `EleutherAI/pythia-1b`, `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-2.8b`, `EleutherAI/pythia-6.9b`, `EleutherAI/pythia-12b`
  - **LLaMA Models**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`
  - **Mistral**: `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`
  - **Code Models**: `Salesforce/codegen-350M-mono`, `Salesforce/codegen-2B-mono`, `bigcode/starcoder`
  - **Other Popular**: `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-6.7b`, `EleutherAI/gpt-j-6b`, `EleutherAI/gpt-neox-20b`
- `--device`: Device selection (`cuda`, `cpu`, or `auto`)

#### **Analysis Settings:**
- `--metric`: Sensitivity metrics (`grad_x_weight`, `grad_squared`)
- `--topk`: Top-K values for ranking (e.g., `10 50 100`)
- `--perturbation-ratios`: Fraction of top-K to mask (e.g., `0.1 0.5 1.0`)

#### **Data Settings:**
- `--eval-limit`: Number of evaluation texts (default: 100)
- `--data-file`: Custom evaluation text file

#### **Performance Settings:**
- `--fast-perturbation`: Use text subset for faster perturbation analysis
- `--no-perturbation`: Skip perturbation (sensitivity analysis only)
- `--verbose`: Detailed progress output

#### **Output Settings:**
- `--output`: Results directory (default: `outputs`)
- `--experiment-id`: Custom experiment identifier

### **📈 Understanding the Output:**

The Critical Weight Analysis system provides detailed console output and generates multiple files for comprehensive analysis. Here's what each section means:

#### **📊 Console Output Breakdown:**

##### **🔬 Analysis Header & Configuration**
```
🔬 Critical Weight Analysis - Phase 1 Runner
============================================================
📅 Started: 2025-08-26 22:11:18
🤖 Model: EleutherAI/pythia-2.8b
🔧 Device: cuda
📊 Metrics: grad_x_weight
🏆 Top-K values: [50]
📚 Evaluation limit: 30
💾 Output directory: outputs/critical_analysis_20250826_221118
============================================================
```
**Explanation:**
- **Started**: Timestamp for experiment tracking and reproducibility
- **Model**: HuggingFace model identifier being analyzed
- **Device**: Whether using CPU or GPU (cuda) for computation
- **Metrics**: Which sensitivity metric is being computed (`grad_x_weight`, `grad_squared`)
- **Top-K values**: How many most critical weights to identify and test
- **Evaluation limit**: Number of text samples used for perplexity evaluation
- **Output directory**: Where all results are saved (timestamped for uniqueness)

##### **📚 Data Loading Phase**
```
📚 Loading Evaluation Data
✅ Loaded 30 evaluation texts
```
**Explanation:**
- Loads evaluation dataset from `src/data/dev_small.txt`
- **30 texts**: Subset for faster analysis (adjustable with `--eval-limit`)
- These texts are used for both baseline perplexity and post-perturbation evaluation

##### **🤖 Model Loading Phase**
```
🤖 Loading Model  
✅ Model: GPTNeoXForCausalLM
✅ Parameters: 2,775,208,960
✅ Device: cuda:0
```
**Explanation:**
- **GPTNeoXForCausalLM**: PyTorch model class (varies by model family)
- **2.77B parameters**: Total trainable weights in the model
- **cuda:0**: Specific GPU device used (important for multi-GPU systems)

##### **📏 Baseline Evaluation**
```
📏 Baseline Evaluation
✅ Baseline perplexity: 20.84
```
**Explanation:**
- **Baseline perplexity**: Model's original performance on evaluation texts
- **Lower is better**: ~20.84 is good for Pythia-2.8b on this dataset
- This serves as the reference point for measuring perturbation impact

##### **🧮 Sensitivity Analysis Phase**
```
🧮 Computing Sensitivity Metrics
----------------------------------------
🔄 Computing grad_x_weight sensitivity...
  ✅ Layers: 130
  📊 Weights: 837,888
  📈 Mean: 5.43e-05
  📉 Std: 1.91e-04  
  🔝 Max: 1.67e+00
  ⏱️ Time: 3.5s
```
**Explanation:**
- **grad_x_weight**: Gradient × weight magnitude sensitivity metric
- **130 layers**: Total model layers analyzed (attention + MLP + embeddings)
- **837,888 weights**: Subset of weights analyzed (not all 2.77B for efficiency)
- **Statistics**:
  - **Mean (5.43e-05)**: Average sensitivity across all weights
  - **Std (1.91e-04)**: Standard deviation shows sensitivity distribution spread
  - **Max (1.67)**: Highest sensitivity value found (indicates critical weight)
- **Time (3.5s)**: Computation time for the sensitivity analysis

##### **🏆 Critical Weight Ranking Phase**
```
🏆 Ranking Critical Weights
----------------------------------------
🔄 Ranking weights for grad_x_weight...
    🔄 Finding top-50 weights from 130 layers...
    ✅ Selected top-50 weights
  Top-50: 50 weights across 9 layers (11.7s)
```
**Explanation:**
- **Global ranking**: Sorts all 837,888 weights by sensitivity value
- **Top-50**: Identifies the 50 most critical weights across the entire model
- **9 layers**: These critical weights are concentrated in 9 out of 130 layers
- **11.7s**: Time for ranking and selection process
- **Key insight**: Critical weights cluster in specific layers (not uniformly distributed)

##### **🎯 Perturbation Experiments Phase**
```
🎯 Perturbation Analysis
----------------------------------------
📊 Running 5 perturbation experiments...
  grad_x_weight Top-50 10%: PPL 20.8 (+0.0)
  grad_x_weight Top-50 25%: PPL 20.8 (+0.0)
  [... more results ...]
✅ Completed 5/5 experiments
```
**Explanation:**
- **5 experiments**: Different masking ratios tested (10%, 25%, 50%, 75%, 100%)
- **Masking**: Setting critical weights to zero to test impact
- **PPL 20.8 (+0.0)**: Perplexity after perturbation (change from baseline)
- **+0.0 change**: Indicates these weights may not be as critical as expected
- **Research insight**: High gradient×weight doesn't always predict performance impact

##### **💾 File Export Phase**
```
💾 Exporting Results
----------------------------------------
  ✅ Summary: experiment_summary.json
  ✅ Perturbation: perturbation_results.csv (5 rows)
  ✅ Top-50 grad_x_weight: top_50_weights_grad_x_weight.csv
```
**Explanation:**
- **experiment_summary.json**: High-level results for quick analysis
- **perturbation_results.csv**: Detailed data for each masking experiment
- **top_50_weights_grad_x_weight.csv**: Ranked list of most critical weights
- **5 rows**: One row per masking ratio tested

##### **🔬 Final Summary**
```
🔬 Analysis Complete - Summary
============================================================
🤖 Model: EleutherAI/pythia-2.8b
📏 Baseline PPL: 20.84
🎯 Perturbation experiments: 5
📈 Max PPL increase: 0.00
🏆 Most Impactful Perturbation:
  grad_x_weight Top-50 at 10% masking
  PPL: 20.8 → 20.8 (+0.0)
🎉 Analysis complete!
```
**Explanation:**
- **Baseline PPL**: Reference performance
- **Max PPL increase (0.00)**: Largest performance degradation observed
- **Most Impactful**: Which experiment caused the biggest change
- **Research conclusion**: In this case, masking top-50 grad×weight weights had minimal impact

#### **🔍 Interpreting Results:**

##### **Case 1: Significant Impact (PPL increase > 1.0)**
```
grad_x_weight Top-50 25%: PPL 20.8 → 22.3 (+1.5)
```
**Interpretation**: Found truly critical weights! Gradient×weight metric successfully identified important parameters.

##### **Case 2: Minimal Impact (PPL increase < 0.1)** 
```
grad_x_weight Top-50 25%: PPL 20.8 → 20.9 (+0.1)
```
**Interpretation**: Metric may not capture true importance, or model is robust to these perturbations.

##### **Case 3: No Impact (PPL increase = 0.0)**
```
grad_x_weight Top-50 25%: PPL 20.8 → 20.8 (+0.0)
```
**Interpretation**: Either the weights aren't critical, or the masking didn't affect the specific evaluation texts used.

#### **📁 Generated Files Explained:**

1. **`experiment_summary.json`** - Quick overview for automated analysis:
```json
{
  "model_name": "EleutherAI/pythia-2.8b",
  "baseline_perplexity": 20.84,
  "num_parameters": 2775208960,
  "total_weights_analyzed": 837888,
  "layers_analyzed": 130,
  "experiments_completed": 5,
  "max_ppl_increase": 0.00,
  "most_impactful_experiment": "grad_x_weight Top-50 at 10%",
  "completion_time": "2025-08-26 22:12:45"
}
```

2. **`top_K_weights_METRIC.csv`** - Ranked critical weights for further analysis:
```csv
rank,layer_name,weight_indices,sensitivity_value,layer_type
1,"gpt_neox.embed_out.weight","(15, 530)",1.6660115718841553,"embedding"
2,"gpt_neox.embed_out.weight","(13, 521)",1.5076959133148193,"embedding"
3,"gpt_neox.layers.2.mlp.dense_4h_to_h.weight","(1793, 9501)",1.06950843334198,"mlp"
```
**Columns explained:**
- **rank**: Global ranking (1 = most critical)
- **layer_name**: Exact PyTorch parameter name for targeting
- **weight_indices**: Specific tensor coordinates
- **sensitivity_value**: Computed gradient×weight magnitude
- **layer_type**: Category (embedding, attention, mlp) for analysis

3. **`perturbation_results.csv`** - Experimental data for publication:
```csv
metric,topk,mask_ratio,weights_masked,baseline_ppl,perturbed_ppl,ppl_increase,ppl_ratio
grad_x_weight,50,0.1,5,20.84,20.84,0.00,1.00
grad_x_weight,50,0.25,12,20.84,20.84,0.00,1.00
grad_x_weight,50,0.5,25,20.84,20.84,0.00,1.00
```
**Research applications:**
- **mask_ratio**: Percentage of top-K weights zeroed out
- **weights_masked**: Absolute number of weights affected
- **ppl_ratio**: Relative performance degradation (1.0 = no change)
- **Statistical analysis**: Use for significance testing and correlation studies

#### **🚨 Logging & Troubleshooting:**

##### **📝 Console Output is Automatically Logged:**

**✅ YES - All console output is saved for analysis!**

The system automatically captures **all console output** to persistent log files:

- **Log Location**: `outputs/critical_analysis_TIMESTAMP/analysis.log`
- **Content**: Complete session including timestamps, progress bars, statistics, warnings, and errors
- **Format**: Structured with timestamps and log levels for easy analysis
- **Console + File**: Output appears both on console AND in the log file simultaneously

**🔍 Log File Example:**
```
2024-01-15 14:32:10,123 - INFO - 🔍 Logging enabled - saving to: outputs/critical_analysis_20240115_143210/analysis.log
2024-01-15 14:32:15,456 - INFO - Loading evaluation data
2024-01-15 14:32:16,789 - INFO - Loaded 50 evaluation texts
2024-01-15 14:32:20,012 - INFO - Loading model: gpt2
2024-01-15 14:32:35,345 - INFO - Model loaded: 124M parameters on cuda:0
```

**📊 What Gets Logged:**
- **Setup**: Environment configuration, device detection, memory allocation
- **Data Loading**: Number of evaluation texts, tokenization progress
- **Model Loading**: Parameter counts, device placement, memory usage
- **Sensitivity Analysis**: Layer processing, weight counts, computation progress
- **Rankings**: Top-K selection algorithms, statistical summaries
- **Exports**: File creation confirmations, CSV/JSON generation
- **Errors**: Full stack traces, CUDA memory issues, model loading failures
- **Performance**: Timing information, memory usage, GPU utilization

**🔧 Using Log Files for Research:**
```bash
# View latest log
tail -f outputs/critical_analysis_*/analysis.log

# Search for specific patterns
grep "INFO.*Loaded" outputs/critical_analysis_*/analysis.log
grep "ERROR\|WARNING" outputs/critical_analysis_*/analysis.log

# Extract timing information
grep "Processing.*layers" outputs/critical_analysis_*/analysis.log

# Compare multiple runs
diff outputs/critical_analysis_20240115_*/analysis.log outputs/critical_analysis_20240116_*/analysis.log
```

**💡 Log File Benefits:**
- **Reproducibility**: Full record of experimental parameters and outputs
- **Debugging**: Detailed error messages with context and timestamps
- **Performance Analysis**: Memory usage and timing patterns across experiments
- **Research Documentation**: Permanent record of model behavior and sensitivity patterns

##### **Common Output Patterns:**

**✅ Successful Analysis:**
- All phases complete with ✅ checkmarks
- Baseline perplexity computed successfully
- Top-K weights found and ranked
- Files exported without errors

**⚠️ Warning Signs:**
```
📊 Weights: 50,000 (reduced due to memory constraints)
⚠️ Some layers skipped due to size
```
**Action**: Increase `--eval-limit` or use smaller model for full analysis

**❌ Error Indicators:**
```
❌ CUDA out of memory
❌ Model loading failed
❌ No weights found above threshold
```
**Action**: Check GPU memory, model name, or adjust sensitivity thresholds

1. **`experiment_summary.json`** - High-level results overview:
```json
{
  "model_name": "EleutherAI/pythia-2.8b",
  "baseline_perplexity": 20.84,
  "num_parameters": 2775208960,
  "experiments_completed": 5,
  "max_ppl_increase": 0.00
}
```

2. **`top_K_weights_METRIC.csv`** - Ranked critical weights:
```csv
rank,layer,indices,sensitivity
1,embed_out.weight,"(15, 530)",1.6660115718841553
2,embed_out.weight,"(13, 521)",1.5076959133148193
3,gpt_neox.layers.2.mlp.dense_4h_to_h.weight,"(1793, 9501)",1.06950843334198
```

3. **`perturbation_results.csv`** - Experimental results:
```csv
metric,topk,mask_ratio,weights_masked,baseline_ppl,perturbed_ppl,ppl_increase,ppl_ratio
grad_x_weight,50,0.1,5,20.84,20.84,0.00,1.00
grad_x_weight,50,0.25,12,20.84,20.84,0.00,1.00
```

### **� Research Notebook Interface:**

For interactive analysis and visualization:

```bash
# Start Jupyter Lab
jupyter lab notebooks/

# Open: sensitivity_analysis_research.ipynb
# Complete pipeline with plots and detailed analysis
```

## 🔗 Integration with Existing Research

### **Connection to llm_research_project:**

If you have an existing `llm_research_project`, this system perfectly complements it:

1. **Discovery Phase**: Use Critical Weight Analysis to find most sensitive weights
2. **Validation Phase**: Use targeted perturbation experiments on discovered weights
3. **Comparison Phase**: Cross-validate findings between global and local analysis

### **Integration Workflow:**

```bash
# Step 1: Discover critical weights
cd /home/ubuntu/nova/critical_weight_analysis
python phase1_runner.py --model EleutherAI/pythia-2.8b --topk 100 --output integration_results/

# Step 2: Test critical layers in original project  
cd /home/ubuntu/nova/llm_research_project
TARGET_LAYER="gpt_neox.layers.2.mlp.dense_4h_to_h" \
accelerate launch scripts/run_topk_perturb.py

# Step 3: Compare results
./test_integration.sh
```

### **Reading Integration Results:**

The `test_integration.sh` script will:
1. Test the most critical layers found by the analysis
2. Generate logs comparing perturbation impacts
3. Validate if high sensitivity correlates with high perturbation impact

Check these key files:
- `logs/critical_layer2_mlp.log` - Results for layer 2 MLP
- `logs/critical_layer4_mlp.log` - Results for layer 4 MLP  
- `logs/critical_embed_out.log` - Results for embedding layer

## 📊 Example Research Workflows

### **Workflow 1: Single Model Deep Dive**
```bash
# Comprehensive analysis of one model
python phase1_runner.py \
  --model gpt2 \
  --metric grad_x_weight grad_squared \
  --topk 10 50 100 500 \
  --eval-limit 100 \
  --output gpt2_comprehensive/
```

### **Workflow 2: Cross-Model Comparison**
```bash
# Compare sensitivity across model sizes
python phase1_runner.py --model gpt2 --topk 100 --output gpt2_results/
python phase1_runner.py --model EleutherAI/pythia-410m --topk 100 --output pythia410m_results/
python phase1_runner.py --model EleutherAI/pythia-2.8b --topk 100 --output pythia2.8b_results/
```

### **Workflow 3: Method Validation**
```bash
# Test if sensitivity metrics predict perturbation impact
python phase1_runner.py --model pythia-2.8b --metric grad_x_weight --topk 50
# Then run test_integration.sh to validate with targeted experiments
```

```

## 💾 VM Backup & Data Preservation

### **🔄 Before Terminating Lambda Labs Instance**

**Save your work and configurations:**

```bash
# 1. Commit all current work
cd /home/ubuntu/nova/critical_weight_analysis
git add .
git commit -m "Save work before VM termination - $(date)"
git push

# 2. Backup environment configuration
cp .env ~/backup_env_$(date +%Y%m%d_%H%M%S)
cp -r outputs/ ~/backup_outputs_$(date +%Y%m%d_%H%M%S)/

# 3. Export installed packages list with UV (much faster)
uv pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements with UV freeze"
git push

# 4. Upload latest results to HuggingFace
huggingface-cli upload critical-weight-analysis-results outputs/ --repo-type dataset

# 5. Create tarball of entire project (optional)
cd /home/ubuntu
tar -czf nova_backup_$(date +%Y%m%d_%H%M%S).tar.gz nova/
```

### **🚀 Restoring on New VM Instance**

**When you spin up a new Lambda Labs instance:**

```bash
# 1. Follow VM Environment Setup (from section above)
# ... (complete steps 1-3 from VM Environment Setup)

# 2. Restore your exact environment
cd /home/ubuntu/nova/critical_weight_analysis
cp ~/backup_env_* .env  # if you saved .env file
source .env

# 3. Download previous results from HuggingFace (optional)
huggingface-cli download critical-weight-analysis-results --repo-type dataset --local-dir restored_outputs/

# 4. Verify everything works
python scripts/quick_test.py
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 10 --eval-limit 5 --verbose
```

### **📦 Essential Files to Never Lose**

These files contain your work and should always be backed up:

```bash
# Code and configuration
├── src/                    # Your core code
├── phase1_runner.py        # Main CLI interface  
├── .env                    # Environment variables
├── .gitignore             # Git exclusions
├── pyproject.toml         # Package configuration
└── requirements_backup.txt # Exact package versions

# Results and experiments  
├── outputs/               # All experimental results
├── notebooks/             # Research notebooks
└── logs/                  # Execution logs

# HuggingFace cache (optional - can be re-downloaded)
~/.cache/huggingface/      # Model cache (large files)
```

### **⚡ Quick Recovery Commands**

```bash
# One-command complete restore with UV (after VM setup)
cd /home/ubuntu/nova/critical_weight_analysis && \
uv venv .venv --python 3.12 && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
huggingface-cli download critical-weight-analysis-results --repo-type dataset --local-dir outputs/ && \
python scripts/quick_test.py

# One-command backup before termination (with UV freeze)
uv pip freeze > requirements.txt && \
git add . && \
git commit -m "Auto-backup $(date)" && \
git push && \
huggingface-cli upload critical-weight-analysis-results outputs/ --repo-type dataset && \
echo "✅ Backup complete! Safe to terminate VM."
```

## 🔧 Troubleshooting

### **Common Issues:**

1. **Import Errors**: Run `./setup.sh` to ensure all dependencies are installed
2. **CUDA Errors**: Check GPU availability with `python scripts/check_gpu.py`
3. **Memory Issues**: Reduce `--eval-limit` or use `--fast-perturbation`
4. **Slow Performance**: The ranking process is optimized but can take 10-15s for large models

### **Performance Optimization:**

- **Fast Prototyping**: Use `--eval-limit 10 --fast-perturbation`
- **Production Analysis**: Use `--eval-limit 100` or higher
- **Memory Efficiency**: Process is optimized for memory but may need 40GB+ for largest models

### **Getting Help:**

1. Check `scripts/quick_test.py` for basic functionality
2. Review `INTEGRATION_GUIDE.md` for research workflow details
3. Examine example outputs in timestamped result directories

## 🎯 Research Applications

### **Publication-Ready Analysis:**
This system enables comprehensive LLM sensitivity research suitable for academic publication:

1. **"Gradient-Based Weight Sensitivity Predicts Perturbation Impact in Large Language Models"**
2. **"Cross-Model Analysis of Critical Weight Patterns in Transformer Architectures"**  
3. **"Validation of Global Sensitivity Metrics Through Targeted Perturbation Experiments"**

### **Research Questions Addressed:**
- Which weights are most critical for language model performance?
- Do gradient-based sensitivity metrics predict actual perturbation impact?
- How do critical weight patterns differ across model sizes and architectures?
- Can global sensitivity analysis guide targeted robustness testing?

### **Experimental Design:**
The system provides a complete experimental framework from discovery → validation → analysis, enabling rigorous scientific investigation of transformer model robustness and weight importance.

---

## 📝 Quick Reference Commands

```bash
# Setup
./setup.sh && source .venv/bin/activate

# Basic analysis  
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100

# Comprehensive analysis
python phase1_runner.py --model EleutherAI/pythia-2.8b --metric grad_x_weight grad_squared --topk 50 100 500 --eval-limit 100

# Fast prototyping
python phase1_runner.py --model gpt2 --topk 10 --eval-limit 5 --fast-perturbation --verbose

# Integration testing
./test_integration.sh

# Jupyter notebook
jupyter lab notebooks/sensitivity_analysis_research.ipynb
```

🎉 **Your complete research system for LLM weight sensitivity analysis is ready!** 🔬✨
