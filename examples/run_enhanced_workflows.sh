#!/bin/bash
"""
Example workflows for enhanced critical weight analysis.

Demonstrates different usage patterns and analysis combinations
for PhD research experiments.
"""

# Set up environment
source .venv/bin/activate

echo "🚀 Critical Weight Analysis - Enhanced Example Workflows"
echo "========================================================"

# Create outputs directory
mkdir -p outputs/examples

# Example 1: Basic Enhanced Analysis
echo ""
echo "📊 Example 1: Basic Enhanced Analysis with Downstream Tasks"
echo "-----------------------------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 50 \
    --downstream-tasks \
    --task-samples 50 \
    --weight-analysis \
    --save-plots \
    --out-dir outputs/examples/basic_enhanced \
    --verbose

# Example 2: Architecture Analysis Focus
echo ""
echo "🏗️ Example 2: Architecture-Focused Analysis"
echo "--------------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2-medium \
    --metric hutchinson_diag \
    --topk 100 \
    --mode global \
    --architecture-analysis \
    --weight-analysis \
    --clustering-method kmeans \
    --n-clusters 7 \
    --save-plots \
    --out-dir outputs/examples/architecture_focus \
    --verbose

# Example 3: Advanced Perturbation Study
echo ""
echo "🔬 Example 3: Advanced Perturbation Experiments"
echo "-----------------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --perturb sign_flip \
    --controls random_k,bottom_k \
    --advanced-perturbations adaptive_noise magnitude_scaling quantization \
    --perturb-scale 0.5 \
    --temporal-stability \
    --save-plots \
    --out-dir outputs/examples/advanced_perturbations \
    --verbose

# Example 4: Temporal Stability Research
echo ""
echo "⏱️ Example 4: Multi-Seed Temporal Stability Study"
echo "-------------------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 100 \
    --seeds 0,1,2,3,4 \
    --stability-check \
    --temporal-stability \
    --weight-analysis \
    --downstream-tasks \
    --save-plots \
    --out-dir outputs/examples/temporal_stability \
    --verbose

# Example 5: Comprehensive Research Pipeline
echo ""
echo "🎯 Example 5: Full Research Pipeline (All Features)"
echo "---------------------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2-large \
    --metric hutchinson_diag \
    --topk 200 \
    --mode global \
    --perturb sign_flip \
    --perturb-scale 1.0 \
    --controls random_k,bottom_k \
    --seeds 0,1,2 \
    --stability-check \
    --downstream-tasks \
    --task-samples 100 \
    --weight-analysis \
    --architecture-analysis \
    --temporal-stability \
    --advanced-perturbations adaptive_noise progressive \
    --clustering-method kmeans \
    --n-clusters 5 \
    --data-file src/data/dev_small.txt \
    --save-plots \
    --out-dir outputs/examples/comprehensive_pipeline \
    --verbose

# Example 6: Quick Development Testing
echo ""
echo "⚡ Example 6: Quick Development Testing"
echo "--------------------------------------"
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 20 \
    --max-samples 10 \
    --weight-analysis \
    --architecture-analysis \
    --save-plots \
    --out-dir outputs/examples/quick_test \
    --verbose

# Example 7: Non-Gradient vs Gradient Comparison
echo ""
echo "🔍 Example 7: Non-Gradient vs Gradient Metrics Comparison"
echo "---------------------------------------------------------"

# Run with magnitude (non-gradient)
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric magnitude \
    --topk 100 \
    --weight-analysis \
    --architecture-analysis \
    --save-plots \
    --out-dir outputs/examples/magnitude_analysis \
    --verbose

# Run with grad_x_weight (gradient-based)
python phase1_runner_enhanced.py \
    --model gpt2 \
    --metric grad_x_weight \
    --topk 100 \
    --weight-analysis \
    --architecture-analysis \
    --save-plots \
    --out-dir outputs/examples/gradient_analysis \
    --verbose

echo ""
echo "✅ All example workflows completed!"
echo "📁 Results saved in outputs/examples/"
echo ""
echo "🔍 To compare results:"
echo "  • Check outputs/examples/*/enhanced_analysis.json for detailed analysis"
echo "  • View outputs/examples/*/plots/ for visualizations"
echo "  • Compare weight_analysis sections across different metrics"
echo "  • Examine architecture_analysis for component-specific insights"
echo ""
echo "📊 Key files to examine:"
echo "  • enhanced_analysis.json - Advanced analysis results"
echo "  • sensitivity_stats.json - Basic sensitivity statistics"
echo "  • manifest.json - Full experimental reproducibility info"
echo "  • plots/ - Publication-ready visualizations"
