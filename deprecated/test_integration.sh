#!/bin/bash
# Integration Test: Use Critical Analysis Discoveries in Original Research
# Based on findings: embed_out.weight and layers.2/4.mlp are most critical

echo "🔬 Testing Critical Weight Analysis Discoveries in Original Research Project"
echo "==========================================================================="

cd /home/ubuntu/nova/llm_research_project

# Activate the original project environment  
source .venv/bin/activate

echo ""
echo "📊 Testing Most Critical Layers Found by Critical Weight Analysis:"

# Test the top critical layers identified
echo ""
echo "🎯 Testing Layer 1: gpt_neox.layers.2.mlp.dense_4h_to_h (Rank #5, #10, #13, #14)"
TARGET_LAYER="gpt_neox.layers.2.mlp.dense_4h_to_h" \
accelerate launch --config_file configs/accelerate_config.yaml \
scripts/run_topk_perturb.py | tee logs/critical_layer2_mlp.log

echo ""
echo "🎯 Testing Layer 2: gpt_neox.layers.4.mlp.dense_4h_to_h (Rank #6, #11, #17)" 
TARGET_LAYER="gpt_neox.layers.4.mlp.dense_4h_to_h" \
accelerate launch --config_file configs/accelerate_config.yaml \
scripts/run_topk_perturb.py | tee logs/critical_layer4_mlp.log

echo ""
echo "🎯 Testing Layer 3: embed_out.weight (Rank #1, #2, #3, #4, #7, #8, #9)"
# Note: embed_out might need special handling - test if it exists
TARGET_LAYER="embed_out.weight" \
accelerate launch --config_file configs/accelerate_config.yaml \
scripts/run_topk_perturb.py | tee logs/critical_embed_out.log || echo "⚠️ embed_out.weight may not be accessible via TARGET_LAYER"

echo ""
echo "📊 Comparison with Original Results:"
echo "  Original Layer 5 MLP: [run baseline comparison]"
echo "  Critical Layer 2 MLP: [check critical_layer2_mlp.log]"
echo "  Critical Layer 4 MLP: [check critical_layer4_mlp.log]"

echo ""
echo "✅ Integration test complete! Check logs/ for detailed results."
echo "🔬 Compare PPL changes to validate critical weight analysis predictions."
