#!/usr/bin/env python3
"""
VM Restoration Verification Script
Run this after VM restoration to ensure everything is working correctly.
"""

import sys
import os
import torch
import subprocess

def test_basic_imports():
    """Test all essential package imports."""
    print("🔍 Testing package imports...")
    
    try:
        import transformers
        import datasets
        import accelerate
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import huggingface_hub
        print(f"✅ Core packages: transformers {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytorch_cuda():
    """Test PyTorch and CUDA functionality."""
    print("🔍 Testing PyTorch and CUDA...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test GPU computation
    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = a @ b
        print(f"✅ GPU computation successful: {c.shape}")
        return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_project_structure():
    """Test project files and structure."""
    print("🔍 Testing project structure...")
    
    required_files = [
        'src/models/loader.py',
        'src/eval/perplexity.py',
        'src/sensitivity/metrics.py',
        'src/sensitivity/rank.py',
        'src/data/dev_small.txt',
        'phase1_runner_enhanced.py',
        'scripts/check_gpu.py',
        'scripts/quick_test.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_model_loading():
    """Test model loading functionality."""
    print("🔍 Testing model loading...")
    
    try:
        sys.path.insert(0, '.')
        from src.models.loader import load_model
        
        model, tokenizer = load_model('gpt2', device='cuda')
        device = next(model.parameters()).device
        print(f"✅ Model loaded successfully on {device}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_huggingface_auth():
    """Test HuggingFace authentication."""
    print("🔍 Testing HuggingFace authentication...")
    
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip().split()[-1]
            print(f"✅ Authenticated as: {username}")
            return True
        else:
            print("❌ Not authenticated with HuggingFace")
            print("  Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"⚠️  Could not check HuggingFace auth: {e}")
        return False

def test_analysis_pipeline():
    """Test basic analysis pipeline."""
    print("🔍 Testing analysis pipeline...")
    
    try:
        sys.path.insert(0, '.')
        from src.models.loader import load_model
        from src.eval.perplexity import compute_perplexity
        
        model, tokenizer = load_model('gpt2', device='cuda')
        texts = ['The quick brown fox jumps over the lazy dog.']
        ppl = compute_perplexity(model, tokenizer, texts, device='cuda')
        print(f"✅ Analysis pipeline working, perplexity: {ppl:.2f}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Analysis pipeline failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 VM Restoration Verification")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_basic_imports),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("Project Structure", test_project_structure),
        ("Model Loading", test_model_loading),
        ("HuggingFace Auth", test_huggingface_auth),
        ("Analysis Pipeline", test_analysis_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 VM restoration successful! Ready for research.")
        print("\nNext steps:")
        print("1. Run: python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5")
        print("2. Check: ./scripts/run_research_tests.sh validation")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Run: python scripts/check_gpu.py")
        print("2. Run: python scripts/quick_test.py")
        print("3. Check: docs/VM_RESTORATION_CHECKLIST.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
