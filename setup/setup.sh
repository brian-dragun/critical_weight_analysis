#!/usr/bin/env bash
# Fresh Lambda Labs setup for Critical Weight Analysis project
# - Python 3.12 with uv package manager
# - CUDA GPU wheels for PyTorch (cu124)
# - Caches under /data/cache (shared with other projects)
set -euo pipefail

# ---------- Config (edit if needed) ----------
PYVER=3.12                   # change to 3.11 for max compatibility
PROJECT_DIR=~/nova/critical_weight_analysis
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124   # CUDA 12.x wheels
# --------------------------------------------

echo "===> Setting up Critical Weight Analysis project"
echo "Project directory: $PROJECT_DIR"

# Check if uv is already installed (from previous setup)
if ! command -v uv >/dev/null 2>&1; then
  echo "===> Installing uv package manager"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
  echo "===> uv already installed, using existing"
  export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure Python runtime is available
echo "===> Checking Python runtime: $PYVER"
uv python install "$PYVER" || echo "Python $PYVER already installed"

echo "===> Creating project environment"
cd "$PROJECT_DIR"

# Clean any existing environment
deactivate 2>/dev/null || true
rm -rf .venv

# Create new virtual environment
uv venv --python "$PYVER" .venv

# Activate environment
source .venv/bin/activate
export PIP_USER=0   # prevents user install conflicts

echo "===> Setting up cache directories (shared with other projects)"
# Cache directories should already exist from llm_research_project setup
if [ ! -d "/data/cache" ]; then
  sudo mkdir -p /data/cache/{pip,hf,torch}
  sudo chown -R "$USER":"$USER" /data/cache
fi

# Set environment variables (should already be in ~/.bashrc)
export HF_HOME=/data/cache/hf
export HF_HUB_CACHE=/data/cache/hf/hub
export TRANSFORMERS_CACHE=/data/cache/hf/transformers
export DATASETS_CACHE=/data/cache/hf/datasets
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip

echo "===> Installing PyTorch with CUDA support"
# Remove any existing PyTorch installations
uv pip uninstall -y torch torchvision torchaudio || true
uv pip install --upgrade pip wheel setuptools

# Install CUDA-enabled PyTorch
if ! uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"; then
  ARCH="$(uname -m)"
  if [ "$ARCH" = "aarch64" ]; then
    echo "----"
    echo "aarch64 wheel install failed. On GH200/Grace, use NVIDIA's PyTorch container:"
    echo "docker run --gpus all --rm -it \\"
    echo "  -v /data/cache:/data/cache -v $PROJECT_DIR:/workspace \\"
    echo "  -e HF_HOME=/data/cache/hf -e TRANSFORMERS_CACHE=/data/cache/hf/transformers \\"
    echo "  -e PIP_CACHE_DIR=/data/cache/pip nvcr.io/nvidia/pytorch:24.07-py3 /bin/bash"
    echo "----"
    exit 2
  else
    echo "Failed to install CUDA wheels. Check driver and index URL."
    exit 2
  fi
fi

echo "===> Installing critical weight analysis dependencies"
# Core ML libraries
uv pip install \
  transformers>=4.40.0 \
  datasets>=2.14.0 \
  accelerate>=0.20.0 \
  safetensors>=0.3.0 \
  tokenizers>=0.13.0 \
  huggingface-hub>=0.16.0

# Scientific computing
uv pip install \
  numpy>=1.21.0 \
  pandas>=1.3.0 \
  scipy>=1.7.0 \
  matplotlib>=3.5.0 \
  seaborn>=0.11.0

# Utilities
uv pip install \
  tqdm>=4.60.0 \
  pyyaml>=6.0 \
  rich \
  typer

# Development tools
uv pip install \
  jupyter>=1.0 \
  ipython>=7.0 \
  pytest>=6.0 \
  black>=22.0 \
  isort>=5.0

# Install the project in development mode
echo "===> Installing critical_weight_analysis package"
uv pip install -e .

echo "===> GPU and environment validation"
python - <<'PY'
import torch
import platform
import sys
import os

print("=== Environment Validation ===")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA version in PyTorch: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: CUDA not available!")
    sys.exit(1)

print("\n=== Library Imports ===")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import datasets
    print("✅ Datasets")
except ImportError as e:
    print(f"❌ Datasets: {e}")

try:
    import accelerate
    print("✅ Accelerate")
except ImportError as e:
    print(f"❌ Accelerate: {e}")

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ Data science stack")
except ImportError as e:
    print(f"❌ Data science: {e}")

print("\n=== Cache Directories ===")
cache_dirs = [
    ("HF_HOME", os.environ.get("HF_HOME", "Not set")),
    ("TRANSFORMERS_CACHE", os.environ.get("TRANSFORMERS_CACHE", "Not set")),
    ("TORCH_HOME", os.environ.get("TORCH_HOME", "Not set")),
]

for name, path in cache_dirs:
    if path != "Not set" and os.path.exists(path):
        print(f"✅ {name}: {path}")
    else:
        print(f"⚠️  {name}: {path}")

print("\n=== Project Structure ===")
project_files = [
    "src/models/loader.py",
    "src/eval/perplexity.py", 
    "src/sensitivity/metrics.py",
    "src/sensitivity/rank.py",
    "src/data/dev_small.txt",
    "pyproject.toml",
]

for file in project_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file}")

print("\n=== Quick GPU Test ===")
try:
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda") 
    c = a @ b
    print(f"✅ GPU matrix multiplication: {c.shape}")
except Exception as e:
    print(f"❌ GPU test failed: {e}")
PY

# Create basic project scripts
echo "===> Creating utility scripts"

# GPU check script
mkdir -p scripts
cat > scripts/check_gpu.py <<'PY'
#!/usr/bin/env python3
"""GPU diagnostics for critical weight analysis."""

import torch
import sys

def main():
    print("=== GPU Diagnostics ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - check drivers")
        return 1
    
    # GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Memory test
    try:
        torch.cuda.empty_cache()
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = x @ y
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        
        print(f"✅ GPU operations working")
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Cached: {cached:.1f} MB")
        
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY

# Quick test script
cat > scripts/quick_test.py <<'PY'
#!/usr/bin/env python3
"""Quick functionality test for critical weight analysis."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test all core imports."""
    print("=== Testing Imports ===")
    
    try:
        from src.models.loader import load_model, set_seed
        print("✅ Model loader")
    except ImportError as e:
        print(f"❌ Model loader: {e}")
        return False
    
    try:
        from src.eval.perplexity import compute_perplexity
        print("✅ Perplexity evaluation")
    except ImportError as e:
        print(f"❌ Perplexity: {e}")
        return False
    
    try:
        from src.sensitivity.metrics import compute_sensitivity
        print("✅ Sensitivity metrics")
    except ImportError as e:
        print(f"❌ Sensitivity: {e}")
        return False
    
    try:
        from src.sensitivity.rank import rank_topk
        print("✅ Weight ranking")
    except ImportError as e:
        print(f"❌ Ranking: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading with a small model."""
    print("\n=== Testing Model Loading ===")
    
    try:
        from src.models.loader import load_model
        
        print("Loading GPT-2 small...")
        model, tokenizer = load_model("gpt2", device="cuda")
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        print(f"✅ Device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test loading evaluation data."""
    print("\n=== Testing Data Loading ===")
    
    try:
        data_path = "src/data/dev_small.txt"
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return False
        
        with open(data_path, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Loaded {len(texts)} text samples")
        print(f"✅ Sample text: {texts[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Critical Weight Analysis - Quick Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_data_loading,
        test_model_loading,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Results ===")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Ready for research.")
        return 0
    else:
        print("⚠️  Some tests failed. Check setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
PY

chmod +x scripts/*.py

echo "===> Final setup validation"
python scripts/quick_test.py

cat <<'TXT'

🎉 Critical Weight Analysis setup complete!

Project structure:
├── src/
│   ├── models/loader.py      # Model loading utilities
│   ├── eval/perplexity.py    # Perplexity computation
│   ├── sensitivity/          # Sensitivity analysis
│   └── data/dev_small.txt    # Evaluation texts
├── scripts/
│   ├── check_gpu.py          # GPU diagnostics
│   └── quick_test.py         # Functionality test
└── outputs/                  # Results and logs

Next steps:
1) Test your environment:
   python scripts/check_gpu.py
   python scripts/quick_test.py

2) Start research notebook:
   jupyter lab notebooks/

3) Run quick sensitivity analysis:
   python -c "
   from src.models.loader import load_model
   from src.eval.perplexity import compute_perplexity
   model, tokenizer = load_model('gpt2')
   texts = open('src/data/dev_small.txt').read().splitlines()[:5]
   ppl = compute_perplexity(model, tokenizer, texts)
   print(f'Baseline perplexity: {ppl:.2f}')
   "

Happy researching! 🔬
TXT
