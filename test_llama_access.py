#!/usr/bin/env python3
"""
LLaMA 7B Access Test & Quick Analysis
Tests if you can access LLaMA and runs a minimal analysis
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

def test_llama_access():
    """Test if LLaMA 2-7B is accessible"""
    print("🦙 Testing LLaMA 2-7B Access")
    print("=" * 40)
    
    # Check HF token
    token = os.getenv('HF_TOKEN')
    if not token:
        print("❌ HF_TOKEN not set!")
        print("Please run: export HF_TOKEN=hf_your_token_here")
        return False
    
    print(f"✅ HF_TOKEN detected: {token[:10]}...")
    
    model_name = "meta-llama/Llama-2-7b-hf"
    
    try:
        print(f"🔄 Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully!")
        
        print(f"🔄 Loading model for {model_name}...")
        print("⚠️ This may take several minutes for 7B parameters...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",          # Automatically distribute across available GPUs
            trust_remote_code=True
        )
        print("✅ Model loaded successfully!")
        
        # Quick test
        print("\n🧪 Running quick functionality test...")
        test_text = "The future of artificial intelligence is"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        print(f"✅ Model working! Output shape: {logits.shape}")
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "model": model_name,
            "access_time": datetime.now().isoformat(),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_gb": total_params * 2 / (1024**3),  # 2 bytes per float16 param
            "device": str(model.device),
            "dtype": str(model.dtype)
        }
        
        print(f"\n📊 Model Information:")
        print(f"• Total Parameters: {total_params:,}")
        print(f"• Model Size: {info['parameter_size_gb']:.1f} GB")
        print(f"• Device: {info['device']}")
        print(f"• Data Type: {info['dtype']}")
        
        # Save info
        with open("llama_access_test.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ LLaMA 2-7B is fully accessible and ready for research!")
        print(f"📁 Test results saved to: llama_access_test.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error accessing LLaMA 2-7B:")
        print(f"Error: {str(e)}")
        
        if "not found" in str(e).lower():
            print("\n🔧 Possible solutions:")
            print("1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
            print("2. Make sure your HuggingFace account is approved")
            print("3. Check that your token has the correct permissions")
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            print("\n🔧 Possible solutions:")
            print("1. Try a smaller model first: pythia-2.8b")
            print("2. Use CPU-only mode (slower): --device cpu")
            print("3. Clear GPU memory: nvidia-smi --gpu-reset")
        else:
            print("\n🔧 Check your internet connection and HF token")
        
        return False

def quick_critical_analysis():
    """Run a very quick critical weight analysis"""
    print("\n🎯 Quick Critical Weight Analysis")
    print("=" * 40)
    
    try:
        # Import our analysis tools
        sys.path.append('/home/ubuntu/nova/critical_weight_analysis/src')
        from models.loader import load_model_and_tokenizer
        from sensitivity.metrics import compute_gradients
        
        model_name = "meta-llama/Llama-2-7b-hf"
        print(f"🔄 Loading {model_name} for analysis...")
        
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        # Quick gradient computation on a tiny sample
        test_texts = ["The quick brown fox", "Artificial intelligence will"]
        
        print("🔄 Computing gradients for sample texts...")
        gradients = compute_gradients(model, tokenizer, test_texts)
        
        print(f"✅ Gradient computation successful!")
        print(f"📊 Gradients computed for {len(gradients)} parameters")
        
        # Find top 10 most sensitive weights
        grad_magnitudes = [(name, grad.abs().max().item()) for name, grad in gradients.items()]
        top_weights = sorted(grad_magnitudes, key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🏆 Top 10 Most Sensitive Weights (Quick Test):")
        for i, (name, magnitude) in enumerate(top_weights, 1):
            print(f"{i:2d}. {name:<60} {magnitude:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🦙 LLaMA 2-7B Research Setup Test")
    print("=" * 50)
    
    # Test 1: Access
    if test_llama_access():
        print("\n" + "="*50)
        
        # Test 2: Quick analysis
        if len(sys.argv) > 1 and sys.argv[1] == "--full-test":
            quick_critical_analysis()
        else:
            print("\n✅ ACCESS TEST COMPLETE!")
            print("\n🎯 Ready for full research! Run:")
            print("./llama_research.sh")
            print("\nOr test analysis capability:")
            print("python test_llama_access.py --full-test")
    else:
        print("\n❌ Setup required before research can begin")
        print("Please follow the setup instructions in setup_llama.sh")
