#!/bin/bash
# Research Testing Script for Critical Weight Analysis
# Author: PhD Research Project - Enhanced System
# Date: August 27, 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/home/ubuntu/nova/critical_weight_analysis"
RESULTS_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/research_tests.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/research_tests.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_DIR/research_tests.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_DIR/research_tests.log"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "phase1_runner_enhanced.py" ]; then
        error "phase1_runner_enhanced.py not found. Please run from the correct directory."
        exit 1
    fi
    
    # Check CUDA availability
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
        error "CUDA not available. Please check GPU setup."
        exit 1
    }
    
    # Check available GPU memory
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    if [ "$gpu_memory" -lt 40000 ]; then
        warning "Available GPU memory ($gpu_memory MB) may be insufficient for large models"
    else
        success "GPU memory check passed: $gpu_memory MB available"
    fi
    
    success "Prerequisites check completed"
}

# Test 1: System Validation
test_system_validation() {
    log "=== TEST 1: System Validation ==="
    
    log "Testing GPT-2 baseline..."
    python phase1_runner_enhanced.py \
        --model gpt2 \
        --metric magnitude \
        --topk 10 \
        --max-samples 5 \
        --out-dir "$RESULTS_DIR/validation/gpt2_baseline" \
        2>&1 | tee "$LOG_DIR/test1_gpt2.log"
    
    if [ $? -eq 0 ]; then
        success "GPT-2 baseline test completed"
    else
        error "GPT-2 baseline test failed"
        return 1
    fi
    
    log "Testing DialoGPT compatibility..."
    python phase1_runner_enhanced.py \
        --model microsoft/DialoGPT-small \
        --metric grad_x_weight \
        --topk 50 \
        --mode per_layer \
        --max-samples 10 \
        --out-dir "$RESULTS_DIR/validation/dialogpt_small" \
        2>&1 | tee "$LOG_DIR/test1_dialogpt.log"
    
    if [ $? -eq 0 ]; then
        success "DialoGPT compatibility test completed"
    else
        error "DialoGPT compatibility test failed"
        return 1
    fi
}

# Test 2: Llama Model Analysis
test_llama_analysis() {
    log "=== TEST 2: Llama Model Analysis ==="
    
    # Check if Llama model is accessible
    log "Testing Llama-3.1-8B access..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')" 2>/dev/null || {
        warning "Llama-3.1-8B not accessible. Trying alternative..."
        # Try Llama-2 or other available model
        LLAMA_MODEL="meta-llama/Llama-2-7b-hf"
    }
    
    LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B"}
    
    log "Running gradient-based sensitivity analysis on $LLAMA_MODEL..."
    python phase1_runner_enhanced.py \
        --model "$LLAMA_MODEL" \
        --metric grad_x_weight \
        --topk 100 \
        --mode per_layer \
        --device cuda \
        --max-samples 30 \
        --save-plots \
        --out-dir "$RESULTS_DIR/llama/grad_x_weight_per_layer" \
        2>&1 | tee "$LOG_DIR/test2_llama_grad.log"
    
    if [ $? -eq 0 ]; then
        success "Llama gradient analysis completed"
    else
        error "Llama gradient analysis failed"
        return 1
    fi
    
    log "Running magnitude-based sensitivity analysis on $LLAMA_MODEL..."
    python phase1_runner_enhanced.py \
        --model "$LLAMA_MODEL" \
        --metric magnitude \
        --topk 500 \
        --mode global \
        --device cuda \
        --max-samples 30 \
        --save-plots \
        --out-dir "$RESULTS_DIR/llama/magnitude_global" \
        2>&1 | tee "$LOG_DIR/test2_llama_magnitude.log"
    
    if [ $? -eq 0 ]; then
        success "Llama magnitude analysis completed"
    else
        warning "Llama magnitude analysis failed (this may be due to memory constraints)"
    fi
}

# Test 3: Mistral Model Analysis
test_mistral_analysis() {
    log "=== TEST 3: Mistral Model Analysis ==="
    
    # Check if Mistral model is accessible
    log "Testing Mistral-7B access..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')" 2>/dev/null || {
        warning "Mistral-7B-v0.1 not accessible. Skipping Mistral tests."
        return 0
    }
    
    log "Running gradient-based sensitivity analysis on Mistral-7B..."
    python phase1_runner_enhanced.py \
        --model mistralai/Mistral-7B-v0.1 \
        --metric grad_x_weight \
        --topk 100 \
        --mode per_layer \
        --device cuda \
        --max-samples 30 \
        --save-plots \
        --out-dir "$RESULTS_DIR/mistral/grad_x_weight_analysis" \
        2>&1 | tee "$LOG_DIR/test3_mistral_grad.log"
    
    if [ $? -eq 0 ]; then
        success "Mistral gradient analysis completed"
    else
        error "Mistral gradient analysis failed"
        return 1
    fi
}

# Test 4: Perturbation Experiments
test_perturbation_experiments() {
    log "=== TEST 4: Perturbation Experiments ==="
    
    log "Running perturbation analysis with controls..."
    python phase1_runner_enhanced.py \
        --model gpt2 \
        --metric grad_x_weight \
        --topk 200 \
        --mode per_layer \
        --perturb sign_flip \
        --controls random_k,bottom_k \
        --seeds 0,1,2 \
        --stability-check \
        --device cuda \
        --max-samples 20 \
        --save-plots \
        --out-dir "$RESULTS_DIR/perturbation/gpt2_full_analysis" \
        2>&1 | tee "$LOG_DIR/test4_perturbation.log"
    
    if [ $? -eq 0 ]; then
        success "Perturbation experiments completed"
    else
        error "Perturbation experiments failed"
        return 1
    fi
}

# Test 5: Multi-metric Comparison
test_multi_metric_comparison() {
    log "=== TEST 5: Multi-metric Comparison ==="
    
    for metric in "magnitude" "grad_x_weight" "grad_squared"; do
        log "Testing metric: $metric"
        python phase1_runner_enhanced.py \
            --model gpt2 \
            --metric "$metric" \
            --topk 100 \
            --mode per_layer \
            --max-samples 15 \
            --save-plots \
            --out-dir "$RESULTS_DIR/multi_metric/${metric}_analysis" \
            2>&1 | tee "$LOG_DIR/test5_${metric}.log"
        
        if [ $? -eq 0 ]; then
            success "Metric $metric completed"
        else
            warning "Metric $metric failed"
        fi
    done
}

# Generate summary report
generate_summary() {
    log "=== Generating Test Summary ==="
    
    SUMMARY_FILE="$RESULTS_DIR/research_test_summary.md"
    
    cat > "$SUMMARY_FILE" << EOF
# Research Testing Summary Report

**Date:** $(date)
**System:** $(hostname)
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
**GPU Memory:** $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1) MB

## Test Results

EOF

    # Check each test result
    for test_dir in validation llama mistral perturbation multi_metric; do
        if [ -d "$RESULTS_DIR/$test_dir" ]; then
            echo "### $test_dir Tests" >> "$SUMMARY_FILE"
            
            # Count successful experiments
            experiment_count=$(find "$RESULTS_DIR/$test_dir" -name "experiment_manifest.json" | wc -l)
            echo "- **Experiments completed:** $experiment_count" >> "$SUMMARY_FILE"
            
            # List generated plots
            plot_count=$(find "$RESULTS_DIR/$test_dir" -name "*.png" | wc -l)
            echo "- **Plots generated:** $plot_count" >> "$SUMMARY_FILE"
            
            echo "" >> "$SUMMARY_FILE"
        fi
    done
    
    cat >> "$SUMMARY_FILE" << EOF

## Next Steps

1. **Review experiment manifests** for reproducibility tracking
2. **Analyze generated plots** for sensitivity patterns
3. **Compare results across models** for consistency validation
4. **Prepare publication-ready figures** from the visualization outputs

## File Locations

- **Results:** \`$RESULTS_DIR\`
- **Logs:** \`$LOG_DIR\`
- **Summary:** \`$SUMMARY_FILE\`

EOF

    success "Summary report generated: $SUMMARY_FILE"
}

# Main execution
main() {
    log "Starting PhD Research Testing Suite"
    log "========================================="
    
    cd "$BASE_DIR"
    
    # Run tests
    check_prerequisites
    
    if test_system_validation; then
        success "System validation passed"
    else
        error "System validation failed. Stopping."
        exit 1
    fi
    
    test_llama_analysis
    test_mistral_analysis
    test_perturbation_experiments
    test_multi_metric_comparison
    
    generate_summary
    
    log "========================================="
    success "Research testing suite completed!"
    log "Check results in: $RESULTS_DIR"
    log "Check logs in: $LOG_DIR"
    
    # Show quick stats
    total_experiments=$(find "$RESULTS_DIR" -name "experiment_manifest.json" | wc -l)
    total_plots=$(find "$RESULTS_DIR" -name "*.png" | wc -l)
    
    echo ""
    echo "📊 Quick Statistics:"
    echo "   • Total experiments completed: $total_experiments"
    echo "   • Total plots generated: $total_plots"
    echo "   • GPU memory used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1) MB"
}

# Handle command line arguments
case "${1:-all}" in
    "validation")
        check_prerequisites && test_system_validation
        ;;
    "llama")
        check_prerequisites && test_llama_analysis
        ;;
    "mistral")
        check_prerequisites && test_mistral_analysis
        ;;
    "perturbation")
        check_prerequisites && test_perturbation_experiments
        ;;
    "multi-metric")
        check_prerequisites && test_multi_metric_comparison
        ;;
    "all")
        main
        ;;
    *)
        echo "Usage: $0 [validation|llama|mistral|perturbation|multi-metric|all]"
        echo "Default: all"
        ;;
esac
