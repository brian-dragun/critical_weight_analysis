# 🤖 Model Selection Guide for Critical Weight Analysis

## 🎯 **Quick Model Recommendations:**

### **👶 Getting Started (Learning & Prototyping):**
```bash
# Perfect for testing your first analysis
python phase1_runner.py --model gpt2 --metric grad_x_weight --topk 50

# Slightly larger for more realistic results
python phase1_runner.py --model EleutherAI/pythia-160m --metric grad_x_weight --topk 100
```

### **🔬 Serious Research (Publication Quality):**
```bash
# Best balance of size and research utility
python phase1_runner.py --model EleutherAI/pythia-2.8b --metric grad_x_weight grad_squared --topk 100 500

# Cross-model validation
python phase1_runner.py --model EleutherAI/pythia-410m --topk 100 --output pythia410m/
python phase1_runner.py --model EleutherAI/pythia-1.4b --topk 100 --output pythia1.4b/
```

### **⚡ Speed Testing (Rapid Iteration):**
```bash
# Ultra-fast for algorithm development
python phase1_runner.py --model distilgpt2 --topk 20 --eval-limit 5 --fast-perturbation

# Small but representative
python phase1_runner.py --model EleutherAI/pythia-70m --topk 50 --eval-limit 10
```

## 📊 **Model Categories & Research Applications:**

### **🏗️ Architecture Comparison Studies:**

#### **GPT-2 Family** (Decoder-Only Transformer)
- **gpt2** (124M): Classic baseline, well-studied
- **gpt2-medium** (355M): Good middle ground
- **gpt2-large** (774M): Higher capacity analysis
- **gpt2-xl** (1.5B): Requires `--no-perturbation` or reduced eval

**Research Use**: Baseline for transformer sensitivity patterns

#### **Pythia Series** (Research-Optimized)
- **pythia-70m**: Ultra-fast prototyping
- **pythia-160m**: Small but realistic
- **pythia-410m**: ⭐ **Sweet spot for most research**
- **pythia-1b**: Medium-scale analysis
- **pythia-1.4b**: 
- **pythia-2.8b**: ⭐ **Production research standard**
- **pythia-6.9b+**: Large-scale (requires memory optimization)

**Research Use**: Cross-scale sensitivity studies, consistent training data

#### **OPT Series** (Meta's GPT Alternative)
- **opt-125m**: Compact alternative to GPT-2
- **opt-350m**: Compare with GPT-2 medium
- **opt-1.3b**: Alternative to Pythia 1.4B
- **opt-6.7b+**: Large-scale alternatives

**Research Use**: Architecture robustness comparison vs GPT-2/Pythia

### **🎓 Research Question → Model Selection:**

#### **"How do sensitivity patterns scale with model size?"**
```bash
# Pythia scaling study
python phase1_runner.py --model EleutherAI/pythia-70m --topk 100 --output scaling_70m/
python phase1_runner.py --model EleutherAI/pythia-410m --topk 100 --output scaling_410m/  
python phase1_runner.py --model EleutherAI/pythia-1.4b --topk 100 --output scaling_1.4b/
python phase1_runner.py --model EleutherAI/pythia-2.8b --topk 100 --output scaling_2.8b/
```

#### **"Do different architectures have different critical weights?"**
```bash
# Architecture comparison
python phase1_runner.py --model gpt2-medium --topk 100 --output gpt2_arch/
python phase1_runner.py --model EleutherAI/pythia-410m --topk 100 --output pythia_arch/
python phase1_runner.py --model facebook/opt-350m --topk 100 --output opt_arch/
```

#### **"How do sensitivity metrics compare?"**
```bash
# Metric validation study
python phase1_runner.py \
  --model EleutherAI/pythia-410m \
  --metric grad_x_weight grad_squared \
  --topk 50 100 500 \
  --eval-limit 100
```

#### **"What's the minimum model size for reliable sensitivity analysis?"**
```bash
# Size threshold study
python phase1_runner.py --model distilgpt2 --topk 50 --output tiny_model/
python phase1_runner.py --model EleutherAI/pythia-70m --topk 50 --output small_model/
python phase1_runner.py --model gpt2 --topk 50 --output medium_model/
```

## 🚀 **Advanced Model Usage:**

### **🦙 Large Models (7B+) - Discovery Mode:**
```bash
# LLaMA 2 (requires HF token)
HF_TOKEN=your_token python phase1_runner.py \
  --model meta-llama/Llama-2-7b-hf \
  --metric grad_x_weight \
  --topk 100 \
  --eval-limit 10 \
  --no-perturbation \
  --output llama7b_discovery/

# Focus on sensitivity discovery only (skip perturbation)
```

### **💻 Code Models - Domain Analysis:**
```bash
# Code vs natural language sensitivity
python phase1_runner.py \
  --model gpt2 \
  --topk 100 \
  --data-file code_samples.txt \
  --output natural_lang_on_code/

# Compare with code-specific models when available
```

### **🔧 Custom Models - Local Analysis:**
```bash
# Your own fine-tuned models
python phase1_runner.py \
  --model /path/to/your/model \
  --metric grad_x_weight \
  --topk 100 \
  --output custom_model_analysis/
```

## ⚠️ **Model-Specific Considerations:**

### **Memory Management:**
- **< 1B parameters**: Full analysis with perturbation
- **1B-7B parameters**: Use `--fast-perturbation` or reduce `--eval-limit`
- **7B+ parameters**: Use `--no-perturbation` for discovery only

### **License & Access:**
- **Open**: GPT-2, Pythia, OPT, DistilGPT (no restrictions)
- **Academic**: Some models require academic email verification
- **Gated**: LLaMA requires approval and HF token
- **Commercial**: Check license terms for commercial research

### **Research Reproducibility:**
- **Pythia models**: ⭐ **Best for reproducible research** (consistent data)
- **GPT-2**: Well-documented, widely compared
- **OPT**: Good alternative with different training
- **Custom models**: Ensure you can share analysis methodology

## 🎯 **Recommended Research Workflows:**

### **Beginner Researcher:**
1. Start: `gpt2` for method learning
2. Validate: `EleutherAI/pythia-160m` for first real results  
3. Scale up: `EleutherAI/pythia-410m` for comprehensive analysis

### **Intermediate Researcher:**
1. Prototype: `EleutherAI/pythia-410m` 
2. Production: `EleutherAI/pythia-2.8b`
3. Compare: Add `gpt2-medium` and `facebook/opt-350m`

### **Advanced Researcher:**
1. Multi-scale: Pythia 70M → 2.8B scaling study
2. Cross-architecture: GPT-2 vs Pythia vs OPT comparison
3. Large-scale: LLaMA 7B+ discovery analysis

## 🔧 **Troubleshooting Model Issues:**

### **Model Won't Load:**
```bash
# Check if model exists and you have access
python -c "from transformers import AutoModel; AutoModel.from_pretrained('MODEL_NAME')"

# For gated models, set HF token
export HF_TOKEN=your_huggingface_token
```

### **Out of Memory:**
```bash
# Reduce evaluation texts
python phase1_runner.py --model LARGE_MODEL --eval-limit 5 --no-perturbation

# Skip perturbation analysis
python phase1_runner.py --model LARGE_MODEL --no-perturbation
```

### **Slow Performance:**
```bash
# Use fast mode for testing
python phase1_runner.py --model MODEL --fast-perturbation --eval-limit 10

# Focus on discovery only
python phase1_runner.py --model MODEL --no-perturbation --topk 50
```

🎉 **Your system works with 50+ models across multiple architectures and sizes!** 🤖✨
