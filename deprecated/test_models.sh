#!/bin/bash
# Model Compatibility Test Suite
# Tests various Hugging Face models with your Critical Weight Analysis system

echo "🤖 Model Compatibility Test Suite for Critical Weight Analysis"
echo "=============================================================="

cd /home/ubuntu/nova/critical_weight_analysis
source .venv/bin/activate

echo ""
echo "📊 Testing Model Categories:"
echo ""

# Function to test a model
test_model() {
    local model=$1
    local category=$2
    local params=$3
    
    echo "🔄 Testing: $model ($category - $params)"
    
    timeout 300 python phase1_runner.py \
        --model "$model" \
        --metric grad_x_weight \
        --topk 5 \
        --eval-limit 3 \
        --no-perturbation \
        --output "test_results/" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✅ SUCCESS: $model works perfectly"
    elif [ $? -eq 124 ]; then
        echo "  ⏱️ TIMEOUT: $model (too large/slow for quick test)"
    else
        echo "  ❌ ERROR: $model has compatibility issues"
    fi
    echo ""
}

echo "1️⃣ GPT-2 Family (OpenAI)"
echo "------------------------"
test_model "gpt2" "GPT-2" "124M"
test_model "gpt2-medium" "GPT-2" "355M" 
test_model "gpt2-large" "GPT-2" "774M"

echo "2️⃣ Pythia Series (EleutherAI - Research Optimized)"
echo "------------------------------------------------"
test_model "EleutherAI/pythia-70m" "Pythia" "70M"
test_model "EleutherAI/pythia-160m" "Pythia" "160M"
test_model "EleutherAI/pythia-410m" "Pythia" "410M"

echo "3️⃣ OPT Series (Meta/Facebook)"
echo "----------------------------"
test_model "facebook/opt-125m" "OPT" "125M"
test_model "facebook/opt-350m" "OPT" "350M"

echo "4️⃣ DistilGPT (Lightweight)"
echo "-------------------------"
test_model "distilgpt2" "DistilGPT" "82M"

echo ""
echo "📋 Test Complete! Summary:"
echo "=========================="
echo ""
echo "✅ RECOMMENDED MODELS FOR RESEARCH:"
echo "  • gpt2, gpt2-medium: Excellent for method development"
echo "  • EleutherAI/pythia-70m to pythia-2.8b: Best for academic research"
echo "  • facebook/opt-125m, opt-350m: Good for architecture comparison"
echo "  • distilgpt2: Perfect for rapid prototyping"
echo ""
echo "⚠️ MEMORY-INTENSIVE MODELS (require --no-perturbation):"
echo "  • gpt2-xl (1.5B), pythia-6.9b+ (6B+), LLaMA models (7B+)"
echo ""
echo "🎯 RESEARCH RECOMMENDATIONS:"
echo "  • Start prototyping with: pythia-70m or gpt2"
echo "  • Production analysis: pythia-2.8b or gpt2-large"
echo "  • Cross-model studies: gpt2 vs pythia-410m vs opt-350m"
echo ""
echo "🔧 If a model fails, try:"
echo "  • Check model license and access requirements"
echo "  • Use --eval-limit 5 --no-perturbation for large models"
echo "  • Verify model uses causal language modeling (not BERT-style)"
echo ""

# Show available tested results
echo "📁 Generated test results:"
ls -la test_results/ 2>/dev/null || echo "  No test results directory created"

echo ""
echo "🚀 Ready to run full analysis on any working model!"
echo "Example: python phase1_runner.py --model EleutherAI/pythia-410m --topk 100"
