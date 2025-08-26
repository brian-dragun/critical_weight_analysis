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
