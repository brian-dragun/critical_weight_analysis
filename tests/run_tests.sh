#!/bin/bash

# Enhanced Critical Weight Analysis - Test Runner
# Runs all integration tests and validates the enhanced system

set -e

echo "=================================================="
echo "Enhanced Critical Weight Analysis - Test Suite"
echo "=================================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo "Test directory: $SCRIPT_DIR"

# Ensure we're in the right directory
cd "$PROJECT_ROOT"

echo ""
echo "1. Checking Python environment..."
# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✓ Using uv for Python environment"
    UV_AVAILABLE=true
else
    echo "⚠ uv not available, using standard python"
    UV_AVAILABLE=false
fi

echo ""
echo "2. Installing test dependencies..."

if [ "$UV_AVAILABLE" = true ]; then
    # Install test dependencies with uv
    uv add pytest pytest-cov unittest-xml-reporting matplotlib seaborn scikit-learn
else
    # Install with pip
    pip install pytest pytest-cov unittest-xml-reporting matplotlib seaborn scikit-learn
fi

echo ""
echo "3. Setting up test environment..."

# Create test output directory
mkdir -p "$PROJECT_ROOT/test_results"
mkdir -p "$PROJECT_ROOT/test_results/plots"
mkdir -p "$PROJECT_ROOT/test_results/coverage"

echo ""
echo "4. Running unit tests..."

# Run individual component tests
echo "  → Testing Weight Analyzer..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestWeightAnalyzer -v
else
    python -m pytest tests/test_integration.py::TestWeightAnalyzer -v
fi

echo "  → Testing Temporal Stability Analyzer..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestTemporalStabilityAnalyzer -v
else
    python -m pytest tests/test_integration.py::TestTemporalStabilityAnalyzer -v
fi

echo "  → Testing Architecture Analyzer..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestArchitectureAnalyzer -v
else
    python -m pytest tests/test_integration.py::TestArchitectureAnalyzer -v
fi

echo "  → Testing Advanced Perturbation Methods..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestAdvancedPerturbationMethods -v
else
    python -m pytest tests/test_integration.py::TestAdvancedPerturbationMethods -v
fi

echo "  → Testing Downstream Tasks..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestDownstreamTasks -v
else
    python -m pytest tests/test_integration.py::TestDownstreamTasks -v
fi

echo "  → Testing Enhanced Visualization..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestEnhancedVisualization -v
else
    python -m pytest tests/test_integration.py::TestEnhancedVisualization -v
fi

echo ""
echo "5. Running integration tests..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py::TestIntegration -v
else
    python -m pytest tests/test_integration.py::TestIntegration -v
fi

echo ""
echo "6. Running full test suite with coverage..."

# Run all tests with coverage
if [ "$UV_AVAILABLE" = true ]; then
    uv run python -m pytest tests/test_integration.py \
        --cov=src \
        --cov-report=html:test_results/coverage/html \
        --cov-report=term \
        --junit-xml=test_results/junit.xml \
        -v
else
    python -m pytest tests/test_integration.py \
        --cov=src \
        --cov-report=html:test_results/coverage/html \
        --cov-report=term \
        --junit-xml=test_results/junit.xml \
        -v
fi

echo ""
echo "7. Testing enhanced workflow examples..."

# Test the enhanced workflow examples
echo "  → Testing enhanced workflow script..."
if [ "$UV_AVAILABLE" = true ]; then
    uv run python examples/enhanced_workflows.py --test-mode
else
    python examples/enhanced_workflows.py --test-mode
fi

echo ""
echo "8. Generating test summary..."

# Create test summary
cat > test_results/test_summary.md << EOF
# Enhanced Critical Weight Analysis - Test Results

## Test Suite Overview

This test suite validates all enhanced features of the critical weight analysis system:

### Tested Components

1. **Weight Analyzer** - Clustering and correlation analysis
2. **Temporal Stability Analyzer** - Cross-condition sensitivity tracking
3. **Architecture Analyzer** - Component-wise sensitivity analysis  
4. **Advanced Perturbation Methods** - Sophisticated perturbation strategies
5. **Downstream Tasks** - HellaSwag and LAMBADA evaluation
6. **Enhanced Visualization** - Advanced plotting functions
7. **Integration Tests** - End-to-end workflow validation

### Test Results

- **Unit Tests**: All component tests passed
- **Integration Tests**: End-to-end workflow validated
- **Coverage Report**: Available in \`test_results/coverage/html/index.html\`
- **JUnit XML**: Available in \`test_results/junit.xml\`

### Enhanced Features Validated

✅ **Weight clustering analysis** with silhouette scoring
✅ **Sensitivity correlation analysis** with statistical metrics  
✅ **Temporal stability tracking** across multiple conditions
✅ **Architecture component analysis** with depth trends
✅ **Progressive perturbation methods** with severity scaling
✅ **Clustered perturbation strategies** based on sensitivity
✅ **Downstream task evaluation** on HellaSwag and LAMBADA
✅ **Enhanced visualization suite** with publication-ready plots
✅ **Comprehensive CLI integration** with all new features
✅ **End-to-end workflow examples** with practical usage patterns

### Usage Examples

The enhanced system can now be used with:

\`\`\`bash
# Full enhanced analysis
python phase1_runner_enhanced.py \\
    --model-name gpt2 \\
    --output-dir results \\
    --weight-analysis \\
    --architecture-analysis \\
    --temporal-stability \\
    --downstream-tasks \\
    --advanced-perturbations

# Or use pre-built workflows
python examples/enhanced_workflows.py
\`\`\`

### System Status

🎉 **READY FOR PRODUCTION USE**

The enhanced critical weight analysis system has passed all integration tests
and is ready for advanced PhD research applications.
EOF

echo ""
echo "=================================================="
echo "🎉 ENHANCED TEST SUITE COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo ""
echo "📊 Test Results Summary:"
echo "   • All unit tests passed"
echo "   • Integration tests validated"  
echo "   • Coverage report generated"
echo "   • Enhanced features confirmed working"
echo ""
echo "📁 Results Location:"
echo "   • Test reports: test_results/"
echo "   • Coverage: test_results/coverage/html/"
echo "   • Summary: test_results/test_summary.md"
echo ""
echo "🚀 The enhanced critical weight analysis system is ready!"
echo "=================================================="
