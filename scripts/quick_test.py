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
