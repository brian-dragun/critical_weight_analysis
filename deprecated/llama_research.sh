#!/bin/bash
# LLaMA 7B Research Workflow for Critical Weight Analysis
# Optimized for large model analysis with memory management

echo "🦙 LLaMA 2-7B Critical Weight Analysis"
echo "====================================="

cd /home/ubuntu/nova/critical_weight_analysis
source .venv/bin/activate

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set!"
    echo "Please run: export HF_TOKEN=hf_your_token_here"
    echo "Or see ./setup_llama.sh for detailed setup instructions"
    exit 1
fi

echo "✅ HF_TOKEN detected"
echo "🔧 Starting LLaMA 7B analysis..."
echo ""

# Create dedicated output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="llama7b_research_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "💾 Results will be saved to: $OUTPUT_DIR"
echo ""

echo "Phase 1: Discovery Analysis (Sensitivity Computation Only)"
echo "--------------------------------------------------------"
echo "🔄 Computing gradient×weight sensitivity for 7B parameters..."
echo "⚠️ This may take 10-30 minutes depending on GPU"
echo ""

# Phase 1: Discovery mode - find critical weights without perturbation
python phase1_runner.py \
    --model meta-llama/Llama-2-7b-hf \
    --metric grad_x_weight \
    --topk 100 500 \
    --eval-limit 20 \
    --no-perturbation \
    --output "$OUTPUT_DIR/phase1_discovery" \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Phase 1 Complete: Discovery analysis finished!"
    echo ""
    
    echo "Phase 2: Critical Layer Identification"
    echo "-------------------------------------"
    
    # Analyze the results to find most critical layers
    echo "🔍 Analyzing top critical weights..."
    python -c "
import pandas as pd
import json
from pathlib import Path

# Find the latest results directory
result_dirs = list(Path('$OUTPUT_DIR/phase1_discovery').glob('critical_analysis_*'))
if result_dirs:
    latest_dir = sorted(result_dirs)[-1]
    
    # Load top weights
    top_weights_file = latest_dir / 'top_100_weights_grad_x_weight.csv'
    if top_weights_file.exists():
        df = pd.read_csv(top_weights_file)
        
        # Analyze layer distribution
        layer_counts = df['layer'].value_counts().head(10)
        
        print('🏆 Top 10 Most Critical Layers in LLaMA 2-7B:')
        print('=' * 50)
        for i, (layer, count) in enumerate(layer_counts.items(), 1):
            print(f'{i:2d}. {layer:<40} ({count} critical weights)')
        
        # Save analysis
        analysis = {
            'model': 'meta-llama/Llama-2-7b-hf',
            'total_critical_weights': len(df),
            'top_critical_layers': layer_counts.head(5).to_dict(),
            'most_sensitive_weight': {
                'layer': df.iloc[0]['layer'],
                'sensitivity': float(df.iloc[0]['sensitivity'])
            }
        }
        
        with open(latest_dir / 'llama_analysis_summary.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f'\n📊 Analysis summary saved to: {latest_dir}/llama_analysis_summary.json')
    else:
        print('⚠️ Top weights file not found')
else:
    print('⚠️ No results directory found')
"
    
    echo ""
    echo "Phase 3: Targeted Small-Scale Validation (Optional)"
    echo "-------------------------------------------------"
    echo "🎯 For validation, you could test critical layers on smaller models:"
    echo ""
    echo "# Test critical layers found in LLaMA on comparable smaller models"
    echo "python phase1_runner.py \\"
    echo "  --model EleutherAI/pythia-2.8b \\"
    echo "  --metric grad_x_weight \\"
    echo "  --topk 100 \\"
    echo "  --eval-limit 50 \\"
    echo "  --output \"$OUTPUT_DIR/validation_pythia2.8b\""
    echo ""
    
    echo "✅ LLaMA 7B Research Complete!"
    echo ""
    echo "📋 SUMMARY:"
    echo "==========="
    echo "🤖 Model: LLaMA 2-7B (7 billion parameters)"
    echo "📊 Analysis: Gradient×weight sensitivity across all layers"
    echo "🏆 Output: Top 100 and 500 most critical weights identified"
    echo "💾 Results: $OUTPUT_DIR/"
    echo ""
    echo "📈 RESEARCH INSIGHTS AVAILABLE:"
    echo "• Which layers contain the most critical weights"
    echo "• Sensitivity distribution across 7B parameters" 
    echo "• Layer-wise importance patterns"
    echo "• Comparison baseline for other models"
    echo ""
    echo "🔗 INTEGRATION WITH YOUR EXISTING RESEARCH:"
    echo "• Compare critical layers with your Pythia findings"
    echo "• Test if critical patterns scale from 2.8B → 7B"
    echo "• Validate sensitivity metrics across model sizes"
    echo ""
    echo "📚 NEXT RESEARCH STEPS:"
    echo "1. Compare LLaMA critical layers with Pythia results"
    echo "2. Test critical layers in your llm_research_project"
    echo "3. Analyze scaling patterns from 2.8B → 7B parameters"
    echo "4. Publication: 'Critical Weight Patterns in Large Language Models'"
    
else
    echo ""
    echo "❌ Phase 1 Failed"
    echo "Possible issues:"
    echo "• HF_TOKEN not working (check token permissions)"
    echo "• Model access not approved (check HuggingFace account)"
    echo "• Insufficient GPU memory (try --eval-limit 10)"
    echo "• Network issues (check internet connection)"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Test token: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')\""
    echo "2. Check GPU: nvidia-smi"
    echo "3. Try smaller eval: python phase1_runner.py --model meta-llama/Llama-2-7b-hf --eval-limit 5 --no-perturbation"
fi

echo ""
echo "📁 All results saved in: $OUTPUT_DIR/"
echo "🎉 LLaMA research workflow complete!"
