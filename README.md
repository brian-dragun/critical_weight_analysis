# Critical Weight Analysis: LLM Sensitivity & Super-Weight Prioritization

A comprehensive research system for analyzing weight sensitivity in transformer language models through gradient-based metrics, critical weight identification, and systematic perturbation experiments.

## 🎯 Project Overview

### **Research Goals:**
1. **Global Weight Discovery**: Identify the most sensitive weights across entire transformer models
2. **Sensitivity Metrics**: Implement gradient×weight and gradient² sensitivity measures  
3. **Systematic Perturbation**: Test impact of masking top-K critical weights
4. **Cross-Model Analysis**: Compare sensitivity patterns across model sizes and architectures
5. **Integration Testing**: Validate discoveries through targeted perturbation experiments

### **Key Capabilities:**
- ⚡ **Fast Analysis**: Process 100K+ weights in seconds with progress tracking
- 🎯 **Top-K Ranking**: Global ranking of most critical weights across all layers
- 📊 **Automated Experiments**: Complete pipeline from sensitivity → ranking → perturbation → export
- 🔗 **Research Integration**: Bridges with existing perturbation research workflows
- 📈 **Professional Output**: CSV exports, JSON summaries, and comprehensive logging

## 🏗️ Complete Project Structure

```
critical_weight_analysis/
├── README.md                           # This comprehensive guide
├── INTEGRATION_GUIDE.md               # How to connect with other research projects
├── pyproject.toml                     # Python package configuration
├── setup.sh                          # Environment setup script
├── phase1_runner.py                   # Main CLI runner (primary interface)
├── research_bridge.py                 # Integration with llm_research_project
├── test_integration.sh                # Integration testing script
├── src/
│   ├── data/
│   │   └── dev_small.txt              # Evaluation text dataset
│   ├── models/
│   │   └── loader.py                  # Model loading with device management
│   ├── sensitivity/
│   │   ├── metrics.py                 # Gradient-based sensitivity computation
│   │   ├── rank.py                    # Top-K weight ranking algorithms
│   │   └── mask.py                    # Weight masking and intervention tools
│   ├── eval/
│   │   └── perplexity.py              # Individual-text perplexity evaluation
│   └── utils/                         # Logging, seeding, device utilities
├── notebooks/
│   └── sensitivity_analysis_research.ipynb  # Complete research notebook
├── outputs/                           # All experiment results
│   ├── critical_analysis_YYYYMMDD_HHMMSS/   # Timestamped result directories
│   │   ├── experiment_summary.json    # High-level results summary
│   │   ├── config.json                # Experiment configuration
│   │   ├── perturbation_results.csv   # Detailed perturbation data
│   │   ├── top_K_weights_METRIC.csv   # Ranked critical weights
│   │   └── sensitivity_statistics.json # Statistical summaries
│   ├── logs/                          # Execution logs
│   └── figs/                          # Generated visualizations
└── scripts/                           # Utility scripts
    ├── check_gpu.py                   # GPU diagnostics
    └── quick_test.py                  # Functionality validation
```

## � VM Environment Setup (Lambda Labs / Cloud GPU)

### **🔥 Quick VM Setup Guide**

This section helps you recreate the entire development environment on a fresh Lambda Labs instance or similar cloud GPU provider.

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

#### **2. Clone Repository & Setup Project:**

```bash
# Create working directory
mkdir -p /home/ubuntu/nova
cd /home/ubuntu/nova

# Clone the project repository
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis

# Run automated environment setup
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python scripts/quick_test.py
```

#### **3. GPU & CUDA Verification:**

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Run GPU diagnostic script
python scripts/check_gpu.py
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

### **🔄 Daily Workflow Commands**

```bash
# Start of session
cd /home/ubuntu/nova/critical_weight_analysis
source .venv/bin/activate
source .env

# Run experiments
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 100

# Commit results
git add outputs/
git commit -m "Add analysis results for gpt2"
git push

# Share on HuggingFace
huggingface-cli upload critical-weight-analysis-results outputs/critical_analysis_* --repo-type dataset
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

#### **Console Output:**
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

📚 Loading Evaluation Data
✅ Loaded 30 evaluation texts

🤖 Loading Model  
✅ Model: GPTNeoXForCausalLM
✅ Parameters: 2,775,208,960
✅ Device: cuda:0

📏 Baseline Evaluation
✅ Baseline perplexity: 20.84

🧮 Computing Sensitivity Metrics
----------------------------------------
🔄 Computing grad_x_weight sensitivity...
  ✅ Layers: 130
  📊 Weights: 837,888
  📈 Mean: 5.43e-05
  📉 Std: 1.91e-04  
  🔝 Max: 1.67e+00
  ⏱️ Time: 3.5s

🏆 Ranking Critical Weights
----------------------------------------
🔄 Ranking weights for grad_x_weight...
    🔄 Finding top-50 weights from 130 layers...
    ✅ Selected top-50 weights
  Top-50: 50 weights across 9 layers (11.7s)

🎯 Perturbation Analysis
----------------------------------------
📊 Running 5 perturbation experiments...
  grad_x_weight Top-50 10%: PPL 20.8 (+0.0)
  grad_x_weight Top-50 25%: PPL 20.8 (+0.0)
  [... more results ...]
✅ Completed 5/5 experiments

💾 Exporting Results
----------------------------------------
  ✅ Summary: experiment_summary.json
  ✅ Perturbation: perturbation_results.csv (5 rows)
  ✅ Top-50 grad_x_weight: top_50_weights_grad_x_weight.csv
  
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

#### **Key Files Generated:**

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

# 3. Export installed packages list
pip freeze > requirements_backup.txt
git add requirements_backup.txt
git commit -m "Update requirements backup"
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
# One-command complete restore (after VM setup)
cd /home/ubuntu/nova/critical_weight_analysis && \
./setup.sh && \
source .venv/bin/activate && \
huggingface-cli download critical-weight-analysis-results --repo-type dataset --local-dir outputs/ && \
python scripts/quick_test.py

# One-command backup before termination
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
