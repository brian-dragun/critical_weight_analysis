# 🚀 Lambda Labs VM Setup Guide for Critical Weight Analysis

## Quick Setup (New VM - 5 minutes)

When you spin up a fresh Lambda Labs VM:

```bash
# 1. Download and run quick setup
wget https://raw.githubusercontent.com/brian-dragun/critical_weight_analysis/master/setup/quick_vm_setup.sh
chmod +x quick_vm_setup.sh
./quick_vm_setup.sh

# 2. Authenticate with HuggingFace (for Llama models)
huggingface-cli login

# 3. Test your setup
cd ~/nova/critical_weight_analysis
python scripts/quick_test.py
```

## Manual Setup (Full Control)

### 1. System Preparation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential software-properties-common
```

### 2. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Clone Project
```bash
mkdir -p ~/nova
cd ~/nova
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis
```

### 4. Run Enhanced Setup
```bash
chmod +x setup/setup.sh
./setup/setup.sh
```

## 📊 Monitoring Your VM

### Real-time Resource Monitoring
```bash
# Single snapshot
python scripts/vm_monitor.py

# Continuous monitoring (30-second intervals)
python scripts/vm_monitor.py monitor

# Save snapshot to file
python scripts/vm_monitor.py save my_vm_state.json
```

### GPU Diagnostics
```bash
# Quick GPU check
python scripts/check_gpu.py

# NVIDIA monitoring
nvidia-smi
watch -n 1 nvidia-smi  # Continuous monitoring
```

## 🔧 Configuration Details

### Your Current Setup Features:
- ✅ **Python 3.12** with UV package manager (10x faster than pip)
- ✅ **PyTorch 2.8.0** with CUDA 12.4/12.1 fallback
- ✅ **Shared cache directories** in `/data/cache/` 
- ✅ **Research tools**: wandb, tensorboard, GPU monitoring
- ✅ **HuggingFace integration** for Llama model access
- ✅ **Automated environment setup** in ~/.bashrc

### Cache Directories:
```
/data/cache/
├── hf/           # HuggingFace models and datasets
├── pip/          # Package cache
└── torch/        # PyTorch model cache
```

### Environment Variables (Auto-set):
```bash
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf/transformers
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip
```

## 🧪 Testing Your Setup

### Quick Validation (2 minutes)
```bash
# System test
python scripts/quick_test.py

# Quick research test
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 10 \
    --max-samples 5
```

### Full Research Validation (5-10 minutes)
```bash
# Llama analysis test
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --max-samples 20 \
    --save-plots
```

### Automated Test Suite
```bash
# Run comprehensive testing
./scripts/run_research_tests.sh validation
./scripts/run_research_tests.sh llama
```

## 🔍 Troubleshooting

### Common Issues:

#### CUDA Version Mismatch
```bash
# Check CUDA version
nvidia-smi
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

# If mismatch, reinstall PyTorch
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Memory Issues
```bash
# Check memory usage
python scripts/vm_monitor.py

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in experiments
python phase1_runner_enhanced.py --model ... --max-samples 10
```

#### HuggingFace Access
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login

# Test access
huggingface-cli whoami
```

## 💾 Backup & Restore

### Daily Backup Commands
```bash
# Save current state
python scripts/vm_monitor.py save daily_backup_$(date +%Y%m%d).json

# Commit code changes
git add . && git commit -m "Daily backup $(date)" && git push

# Export package list
uv pip freeze > requirements_backup.txt
```

### Restore on New VM
```bash
# Quick restore with our script
./setup/quick_vm_setup.sh

# Or manual restore
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis
./setup/setup.sh
huggingface-cli login
```

## 🎯 Research Workflow

### Start a Research Session
```bash
# 1. Check resources
python scripts/vm_monitor.py

# 2. Test system
python scripts/quick_test.py

# 3. Run experiment
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 500 \
    --mode per_layer \
    --save-plots \
    --out-dir results/$(date +%Y%m%d_%H%M%S)

# 4. Monitor progress (in another terminal)
python scripts/vm_monitor.py monitor 60
```

### End Session Checklist
```bash
# 1. Save results
git add outputs/ results/
git commit -m "Research session $(date +%Y%m%d)"
git push

# 2. Generate report
python scripts/generate_research_report.py

# 3. Check final state
python scripts/vm_monitor.py save session_end.json
```

---

**Your Lambda Labs VM is optimized for critical weight analysis research! 🚀**

Need help? Check the [documentation](docs/INDEX.md) or run `python scripts/quick_test.py` to validate your setup.
