# 🦙 LLaMA 2-7B Research Guide

## Quick Start (3 Steps)

### Step 1: Setup Access (5 minutes)
```bash
# Follow LLaMA access setup
./setup_llama.sh
```

### Step 2: Test Access (2 minutes)
```bash
# Activate environment and test
cd /home/ubuntu/nova/critical_weight_analysis
source .venv/bin/activate
python test_llama_access.py
```

### Step 3: Run Research (20-60 minutes)
```bash
# Complete LLaMA analysis
./llama_research.sh
```

---

## What This Enables

### 🧬 Critical Weight Discovery
- **7 billion parameter analysis**: Find the most important weights in LLaMA 2-7B
- **Layer-wise sensitivity**: Discover which transformer layers contain critical weights
- **Gradient-based ranking**: Use gradient×weight sensitivity to rank all 7B parameters

### 📊 Research Capabilities
- **Scaling studies**: Compare critical patterns from 2.8B (Pythia) → 7B (LLaMA)
- **Architecture analysis**: Compare different model architectures (GPT vs LLaMA)
- **Publication-ready data**: Generate CSV/JSON results for academic papers

### 🔬 Integration with Your Existing Research
- **Cross-validation**: Test if critical layers found in LLaMA also matter in your Pythia experiments
- **Perturbation targeting**: Use LLaMA critical weights to guide your `llm_research_project` experiments
- **Baseline comparison**: Compare perturbation sensitivity across model sizes

---

## Research Workflow Explained

### Phase 1: Discovery (20-30 minutes)
```bash
./llama_research.sh
```
**What it does:**
- Loads LLaMA 2-7B with optimized memory settings
- Computes gradient×weight sensitivity for all 7B parameters
- Ranks and exports top 100 and 500 most critical weights
- **No perturbation** - just pure sensitivity discovery

**Output files:**
- `top_100_weights_grad_x_weight.csv` - Most critical weights
- `top_500_weights_grad_x_weight.csv` - Extended critical set
- `llama_analysis_summary.json` - Layer importance summary

### Phase 2: Analysis (Automatic)
**What it provides:**
- Top 10 most critical layers in LLaMA 2-7B
- Weight distribution across transformer blocks
- Sensitivity magnitude comparisons
- Layer-wise importance patterns

### Phase 3: Integration (Manual)
**Connect to your existing research:**
```bash
# Test critical layers in your perturbation experiments
cd /home/ubuntu/nova/llm_research_project
python your_perturbation_script.py --target-layers "layers.10.mlp,layers.15.self_attn"
```

---

## Memory and Performance Optimizations

### Automatic Optimizations in `llama_research.sh`:
- **Half precision**: `torch.float16` reduces memory by 50%
- **Device mapping**: `device_map="auto"` distributes across available GPUs
- **Discovery mode**: `--no-perturbation` skips expensive evaluations
- **Limited evaluation**: `--eval-limit 20` for manageable runtimes
- **Progress tracking**: Real-time updates on analysis progress

### Your GPU Setup (NVIDIA GH200 480GB):
- **Massive memory**: Can easily handle 7B models
- **Potential for larger models**: Could even handle 13B+ models
- **Multi-GPU capability**: Can distribute model across multiple devices

---

## Expected Results

### Critical Layer Patterns
Based on similar research, expect to find:
- **Attention layers**: High sensitivity in middle layers (10-20)
- **MLP components**: Critical feed-forward projections
- **Embedding layers**: Important but different sensitivity pattern
- **Final layers**: Output projection sensitivities

### Research Insights
- **Scaling laws**: How critical patterns change from 2.8B → 7B
- **Architecture differences**: LLaMA vs Pythia critical layer distribution
- **Universal patterns**: Which critical weights appear across model sizes

---

## Troubleshooting

### Access Issues
```bash
# Test HF token
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"

# If fails, check:
# 1. HuggingFace account approved for LLaMA access
# 2. HF_TOKEN has correct permissions
# 3. Token not expired
```

### Memory Issues
```bash
# Reduce memory usage
python phase1_runner.py \
  --model meta-llama/Llama-2-7b-hf \
  --eval-limit 5 \
  --topk 50 \
  --no-perturbation
```

### Performance Issues
```bash
# Quick test on smaller model first
python phase1_runner.py \
  --model EleutherAI/pythia-2.8b \
  --metric grad_x_weight \
  --topk 100
```

---

## Next Research Steps

### 1. Cross-Model Validation
- Compare LLaMA critical layers with your Pythia findings
- Test if critical patterns transfer across architectures
- Publish scaling law research

### 2. Targeted Perturbation
- Use LLaMA critical weights in your `llm_research_project`
- Test perturbation sensitivity in discovered critical layers
- Compare perturbation vs gradient-based sensitivity

### 3. Advanced Analysis
- Analyze attention head patterns in critical layers
- Study weight magnitude vs sensitivity correlation
- Explore critical weight clustering across layers

### 4. Publication Opportunities
- "Critical Weight Patterns in Large Language Models"
- "Scaling Laws for Neural Network Sensitivity"
- "Cross-Architecture Weight Importance Analysis"

---

## Files Created for LLaMA Research

```
critical_weight_analysis/
├── setup_llama.sh           # HuggingFace access setup
├── test_llama_access.py     # Quick access verification
├── llama_research.sh        # Complete research workflow
└── LLAMA_RESEARCH_GUIDE.md  # This guide
```

Ready to discover what makes LLaMA 2-7B tick! 🚀
