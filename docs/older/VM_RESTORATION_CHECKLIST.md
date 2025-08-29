# 🔄 VM Restoration Checklist

## Current VM State Snapshot (August 28, 2025)

Your current working configuration:
- ✅ PyTorch 2.5.1 with CUDA 12.4 support
- ✅ HuggingFace authenticated as: bdragun
- ✅ All tests passing with GPU support
- ✅ Git repository up to date with remote
- ✅ Virtual environment with critical_weight_analysis package installed

## 🚀 Complete VM Restoration Procedure

### Option 1: Quick Restoration (5 minutes)

```bash
# 1. Download and run the automated setup
wget https://raw.githubusercontent.com/brian-dragun/critical_weight_analysis/master/setup/quick_vm_setup.sh
chmod +x quick_vm_setup.sh
./quick_vm_setup.sh

# 2. Authenticate HuggingFace (required for Llama models)
huggingface-cli login
# Enter your token when prompted

# 3. Validate setup
cd ~/nova/critical_weight_analysis
python scripts/quick_test.py

# 4. Test GPU and model loading
python -c "
import torch
from src.models.loader import load_model
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
model, tokenizer = load_model('gpt2', device='cuda')
print('✅ System ready for research')
"
```

### Option 2: Manual Step-by-Step (10-15 minutes)

```bash
# 1. System preparation
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential software-properties-common

# 2. Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. Clone repository
mkdir -p ~/nova
cd ~/nova
git clone https://github.com/brian-dragun/critical_weight_analysis.git
cd critical_weight_analysis

# 4. Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# 5. Install PyTorch with CUDA 12.4 support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 6. Install all dependencies
uv pip install -r setup/requirements.txt

# 7. Install project in development mode
uv pip install -e .

# 8. Setup HuggingFace authentication
huggingface-cli login

# 9. Configure environment variables (add to ~/.bashrc)
cat >> ~/.bashrc <<'EOF'
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export TORCH_HOME=~/.cache/torch
export PATH="$HOME/.local/bin:$PATH"
EOF
source ~/.bashrc

# 10. Validate installation
python scripts/quick_test.py
```

## 🔍 Validation Checklist

After restoration, verify these items work:

### ✅ Basic System Check
```bash
cd ~/nova/critical_weight_analysis

# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU access
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### ✅ Package Imports
```bash
python -c "
import transformers
import datasets
import accelerate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ All packages imported successfully')
"
```

### ✅ HuggingFace Authentication
```bash
huggingface-cli whoami
# Should show: user: bdragun
```

### ✅ Model Loading Test
```bash
python -c "
from src.models.loader import load_model
model, tokenizer = load_model('gpt2', device='cuda')
print(f'✅ Model loaded on: {next(model.parameters()).device}')
"
```

### ✅ Analysis Pipeline Test
```bash
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 10 \
    --max-samples 5 \
    --verbose
```

### ✅ Automated Testing Suite
```bash
python scripts/check_gpu.py
python scripts/quick_test.py
```

## 🔧 Expected Results

After successful restoration, you should see:

1. **PyTorch Version**: 2.5.1 or newer with CUDA 12.4 support
2. **GPU Detection**: CUDA available: True, GPU count: 1
3. **HuggingFace**: Authenticated as bdragun
4. **Model Loading**: GPT-2 loads successfully on GPU
5. **All Tests**: Pass with "🎉 All tests passed! Ready for research."

## 🚨 Troubleshooting

### If PyTorch CUDA doesn't work:
```bash
# Check CUDA version compatibility
nvidia-smi

# Reinstall PyTorch with correct CUDA version
uv pip uninstall torch torchvision torchaudio
# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### If HuggingFace authentication fails:
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login
# Use your token from: https://huggingface.co/settings/tokens
```

### If model loading fails:
```bash
# Test with smaller model first
python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModel.from_pretrained('gpt2')
print('✅ Basic model loading works')
"
```

## 📋 Critical Files to Preserve

If you want to backup specific configurations:

```bash
# Save current package versions
pip freeze > my_environment_backup.txt

# Save HuggingFace token (if needed)
cat ~/.cache/huggingface/token

# Save custom configurations
cp ~/.bashrc ~/.bashrc.backup
```

## 🎯 Research-Ready Commands

Once restored, you can immediately run:

```bash
# Quick validation
python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5

# Llama research
python phase1_runner_enhanced.py \
    --model meta-llama/Llama-3.1-8B \
    --metric grad_x_weight \
    --topk 100 \
    --mode per_layer \
    --max-samples 20 \
    --save-plots

# Automated testing
./scripts/run_research_tests.sh validation
```

---

**Total Restoration Time**: 5-15 minutes depending on method chosen
**Success Criteria**: All validation checks pass and research commands work
