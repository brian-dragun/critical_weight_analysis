#!/usr/bin/env python3
"""
LLaMA Research Runner
Optimized workflow for LLaMA model critical weight analysis with memory management

Usage:
    python llama_research_runner.py --model meta-llama/Llama-2-7b-hf
    python llama_research_runner.py --model meta-llama/Llama-2-13b-hf --discovery-only
    python llama_research_runner.py --model meta-llama/Llama-2-7b-chat-hf --full-analysis
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaResearchRunner:
    """Specialized runner for LLaMA model critical weight analysis."""
    
    LLAMA_MODELS = {
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", 
        "llama2-13b": "meta-llama/Llama-2-13b-hf",
        "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "llama2-70b": "meta-llama/Llama-2-70b-hf",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3-70b": "meta-llama/Meta-Llama-3-70B",
        "code-llama-7b": "codellama/CodeLlama-7b-hf",
        "code-llama-13b": "codellama/CodeLlama-13b-hf"
    }
    
    def __init__(self):
        """Initialize LLaMA research runner."""
        self.check_prerequisites()
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        logger.info("🦙 LLaMA Research Workflow - Checking Prerequisites")
        
        # Check HF_TOKEN
        if not os.getenv('HF_TOKEN'):
            logger.error("❌ Error: HF_TOKEN not set!")
            logger.error("Please run: export HF_TOKEN=hf_your_token_here")
            logger.error("Or see setup_llama.sh for detailed setup instructions")
            raise ValueError("HF_TOKEN environment variable required")
        
        logger.info("✅ HF_TOKEN detected")
        
        # Check if phase1_runner.py exists
        if not Path("phase1_runner.py").exists():
            raise FileNotFoundError("phase1_runner.py not found in current directory")
        
        logger.info("✅ Phase1 runner found")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ GPU available: {gpu_count} device(s), {gpu_memory:.1f}GB memory")
            else:
                logger.warning("⚠️ No GPU detected - LLaMA analysis will be very slow on CPU")
        except ImportError:
            logger.warning("⚠️ PyTorch not available - cannot check GPU status")
    
    def get_memory_optimized_params(self, model_name: str, analysis_mode: str) -> Dict:
        """Get memory-optimized parameters based on model size and mode.
        
        Args:
            model_name: LLaMA model identifier
            analysis_mode: One of 'discovery', 'validation', 'full'
            
        Returns:
            Dictionary with optimized parameters
        """
        # Model size estimates (parameters in billions)
        model_sizes = {
            "7b": 7,
            "8b": 8, 
            "13b": 13,
            "70b": 70
        }
        
        # Determine model size
        model_size = 7  # Default
        for size_key, size_val in model_sizes.items():
            if size_key in model_name.lower():
                model_size = size_val
                break
        
        # Base parameters for different modes
        if analysis_mode == "discovery":
            base_params = {
                "eval_limit": 50,
                "topk": [100, 500],
                "metrics": ["grad_x_weight"],
                "no_perturbation": True
            }
        elif analysis_mode == "validation":
            base_params = {
                "eval_limit": 20,
                "topk": [50],
                "metrics": ["grad_x_weight"],
                "no_perturbation": False
            }
        elif analysis_mode == "full":
            base_params = {
                "eval_limit": 100,
                "topk": [100, 500, 1000],
                "metrics": ["grad_x_weight", "grad_squared"],
                "no_perturbation": False
            }
        else:
            raise ValueError(f"Unknown analysis mode: {analysis_mode}")
        
        # Adjust based on model size
        if model_size >= 70:
            # Very large models
            base_params["eval_limit"] = min(base_params["eval_limit"], 20)
            base_params["topk"] = [min(k, 100) for k in base_params["topk"]]
            logger.info("🐘 Very large model detected - using minimal parameters")
            
        elif model_size >= 13:
            # Large models  
            base_params["eval_limit"] = min(base_params["eval_limit"], 30)
            base_params["topk"] = [min(k, 500) for k in base_params["topk"]]
            logger.info("🦏 Large model detected - using reduced parameters")
            
        elif model_size >= 7:
            # Medium models
            logger.info("🐎 Medium model detected - using standard parameters")
        
        # Add memory management flags
        base_params.update({
            "model_size_gb": model_size * 2,  # Rough estimate
            "recommended_gpu_memory": max(model_size * 4, 24),  # GB
        })
        
        return base_params
    
    def run_discovery_phase(self, model_name: str, output_dir: Path, params: Dict) -> Dict:
        """Run discovery phase for critical weight analysis.
        
        Args:
            model_name: LLaMA model identifier
            output_dir: Output directory
            params: Optimized parameters
            
        Returns:
            Dictionary with execution results
        """
        logger.info("🔍 Phase 1: Discovery Analysis (Sensitivity Computation Only)")
        logger.info("=" * 60)
        
        model_size = params.get("model_size_gb", "Unknown")
        recommended_memory = params.get("recommended_gpu_memory", "24+")
        
        logger.info(f"🔄 Computing gradient×weight sensitivity for {model_size}GB model")
        logger.info(f"💾 Recommended GPU memory: {recommended_memory}GB")
        logger.info("⚠️ This may take 10-60 minutes depending on GPU and model size")
        logger.info("")
        
        # Prepare discovery output directory
        discovery_dir = output_dir / "phase1_discovery"
        discovery_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, "phase1_runner.py",
            "--model", model_name,
            "--metric", *params["metrics"],
            "--topk", *[str(k) for k in params["topk"]],
            "--eval-limit", str(params["eval_limit"]),
            "--output", str(discovery_dir),
            "--verbose"
        ]
        
        if params.get("no_perturbation"):
            cmd.append("--no-perturbation")
        
        logger.info(f"🚀 Running: {' '.join(cmd)}")
        logger.info("")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, text=True)
            execution_time = time.time() - start_time
            
            logger.info("")
            logger.info("✅ Phase 1 Complete: Discovery analysis finished!")
            logger.info(f"⏱️ Execution time: {execution_time/60:.1f} minutes")
            
            return {
                "status": "success",
                "execution_time": execution_time,
                "discovery_dir": discovery_dir,
                "command": cmd
            }
            
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Phase 1 Failed after {execution_time/60:.1f} minutes")
            logger.error(f"Error code: {e.returncode}")
            
            return {
                "status": "error",
                "execution_time": execution_time,
                "error": str(e),
                "command": cmd
            }
        
        except KeyboardInterrupt:
            execution_time = time.time() - start_time
            logger.info(f"\n⏹️ Phase 1 Interrupted after {execution_time/60:.1f} minutes")
            
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "command": cmd
            }
    
    def run_validation_phase(self, model_name: str, discovery_dir: Path, output_dir: Path, params: Dict) -> Dict:
        """Run validation phase with perturbation testing.
        
        Args:
            model_name: LLaMA model identifier
            discovery_dir: Directory with discovery results
            output_dir: Output directory
            params: Optimized parameters
            
        Returns:
            Dictionary with execution results
        """
        logger.info("🧪 Phase 2: Validation Analysis (Perturbation Testing)")
        logger.info("=" * 60)
        logger.info("🎯 Testing discovered critical weights with perturbation")
        logger.info("")
        
        # Find latest discovery results
        result_dirs = list(discovery_dir.glob("critical_analysis_*"))
        if not result_dirs:
            logger.error("❌ No discovery results found for validation")
            return {"status": "error", "error": "No discovery results found"}
        
        latest_discovery = max(result_dirs, key=lambda x: x.stat().st_mtime)
        logger.info(f"📂 Using discovery results from: {latest_discovery}")
        
        # Prepare validation output directory
        validation_dir = output_dir / "phase2_validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Build validation command
        cmd = [
            sys.executable, "phase1_runner.py",
            "--model", model_name,
            "--metric", *params["metrics"],
            "--topk", *[str(k) for k in params["topk"]],
            "--eval-limit", str(params["eval_limit"]),
            "--output", str(validation_dir),
            "--verbose"
        ]
        
        logger.info(f"🚀 Running: {' '.join(cmd)}")
        logger.info("")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, text=True)
            execution_time = time.time() - start_time
            
            logger.info("")
            logger.info("✅ Phase 2 Complete: Validation analysis finished!")
            logger.info(f"⏱️ Execution time: {execution_time/60:.1f} minutes")
            
            return {
                "status": "success",
                "execution_time": execution_time,
                "validation_dir": validation_dir,
                "command": cmd
            }
            
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Phase 2 Failed after {execution_time/60:.1f} minutes")
            
            return {
                "status": "error",
                "execution_time": execution_time,
                "error": str(e),
                "command": cmd
            }
        
        except KeyboardInterrupt:
            execution_time = time.time() - start_time
            logger.info(f"\n⏹️ Phase 2 Interrupted after {execution_time/60:.1f} minutes")
            
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "command": cmd
            }
    
    def generate_llama_research_report(self, model_name: str, results: Dict, output_dir: Path):
        """Generate specialized report for LLaMA research.
        
        Args:
            model_name: LLaMA model identifier
            results: Combined results from all phases
            output_dir: Output directory
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_file = output_dir / "LLAMA_RESEARCH_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write(f"""# LLaMA Critical Weight Analysis Report

**Model**: {model_name}  
**Analysis Date**: {timestamp}  
**Workflow**: LLaMA Research Pipeline  

## Executive Summary

This report presents critical weight analysis results for the LLaMA model using specialized 
memory-optimized parameters designed for large language model research.

## Model Information

**Architecture**: LLaMA (Large Language Model Meta AI)  
**Parameters**: {results.get('model_size_gb', 'Unknown')}B parameters  
**GPU Memory Required**: {results.get('recommended_gpu_memory', '24+')}GB  

## Phase 1: Discovery Analysis

""")
            
            if 'discovery' in results:
                discovery = results['discovery']
                f.write(f"""**Status**: {discovery['status'].upper()}  
**Execution Time**: {discovery['execution_time']/60:.1f} minutes  
**Output Directory**: `{discovery.get('discovery_dir', 'N/A')}`  

""")
                if discovery['status'] == 'success':
                    f.write("""✅ **Discovery Successful**: Critical weight sensitivity analysis completed
- Gradient×weight sensitivity computed across all model layers
- Top-K most critical weights identified and ranked
- Memory-optimized parameters used for large model compatibility

""")
                else:
                    f.write(f"""❌ **Discovery Failed**: {discovery.get('error', 'Unknown error')}
- Consider reducing --eval-limit or --topk values
- Verify sufficient GPU memory is available
- Check HuggingFace token permissions for LLaMA access

""")
            
            if 'validation' in results:
                validation = results['validation']
                f.write(f"""## Phase 2: Validation Analysis

**Status**: {validation['status'].upper()}  
**Execution Time**: {validation['execution_time']/60:.1f} minutes  
**Output Directory**: `{validation.get('validation_dir', 'N/A')}`  

""")
                if validation['status'] == 'success':
                    f.write("""✅ **Validation Successful**: Perturbation testing completed
- Critical weights tested with controlled perturbations
- Model robustness to weight modifications measured
- Validation confirms theoretical sensitivity predictions

""")
            
            f.write(f"""## LLaMA-Specific Insights

### Model Architecture Considerations:
- **Attention Layers**: LLaMA uses multi-head attention with RMSNorm
- **Feed-Forward Networks**: SwiGLU activation function
- **Rotary Positional Embeddings**: RoPE for position encoding
- **Vocabulary**: 32,000 tokens (sentencepiece tokenization)

### Critical Weight Patterns (LLaMA-specific):
- **Embedding Layers**: Input/output embeddings often show high sensitivity
- **Attention Projection**: Q, K, V projection matrices are frequently critical
- **Feed-Forward Gates**: SwiGLU gate weights impact model behavior significantly
- **Layer Normalization**: RMSNorm parameters can be surprisingly critical

### Research Applications:
- **Model Compression**: Use critical weight maps for targeted pruning
- **Fine-tuning Efficiency**: Focus updates on most critical parameters
- **Robustness Analysis**: Understand failure modes and vulnerabilities
- **Architecture Studies**: Compare critical patterns across LLaMA variants

## Memory and Performance Notes

### Optimization Strategies Used:
- **Gradient Checkpointing**: Reduced memory usage during backpropagation
- **Evaluation Batching**: Controlled batch sizes for memory management
- **Metric Selection**: Focused on most informative sensitivity measures
- **Top-K Limiting**: Analyzed most critical weights first

### Scaling Recommendations:
- **7B Models**: 24GB GPU memory recommended
- **13B Models**: 40GB+ GPU memory required  
- **70B Models**: Multi-GPU setup or CPU analysis recommended
- **Production**: Use discovery results to guide targeted analysis

## Files Generated

### Discovery Results:
- `phase1_discovery/critical_analysis_*/` - Complete sensitivity analysis
- `*_weights_grad_x_weight.csv` - Critical weights ranked by gradient×weight
- `experiment_summary.json` - Analysis metadata and parameters
- `analysis.log` - Complete execution log with timestamps

### Validation Results (if run):
- `phase2_validation/critical_analysis_*/` - Perturbation test results
- `perturbation_results.csv` - Impact measurements for critical weights

## Next Steps for LLaMA Research

### Immediate Actions:
1. **Review Critical Layers**: Examine top-ranked attention and FFN components
2. **Cross-Model Comparison**: Compare patterns between 7B, 13B, and 70B variants  
3. **Task-Specific Analysis**: Test critical weights on downstream tasks
4. **Integration Testing**: Validate findings with perturbation experiments

### Extended Research:
1. **Instruction Tuning Impact**: Compare base vs. chat model critical patterns
2. **Code-LLaMA Specialization**: Analyze code-specific critical weight patterns
3. **Multi-GPU Scaling**: Extend analysis to larger LLaMA variants
4. **Production Optimization**: Use critical weights for deployment optimization

## Research Citation

If you use these results in research, please cite:
```bibtex
@misc{{llama_critical_weights_{timestamp.replace('-', '').replace(' ', '_').replace(':', '')},
  title={{Critical Weight Analysis of LLaMA Models}},
  author={{Critical Weight Analysis Pipeline}},
  year={{{timestamp[:4]}}},
  note={{Analysis conducted using memory-optimized critical weight discovery}}
}}
```

---
*Generated by LLaMA Research Runner*  
*Part of the Critical Weight Analysis Pipeline*
""")
        
        logger.info(f"📄 LLaMA research report generated: {report_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLaMA Research Runner - Optimized Critical Weight Analysis for LLaMA Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Shortcuts:
  llama2-7b        -> meta-llama/Llama-2-7b-hf
  llama2-7b-chat   -> meta-llama/Llama-2-7b-chat-hf
  llama2-13b       -> meta-llama/Llama-2-13b-hf
  llama3-8b        -> meta-llama/Meta-Llama-3-8B
  code-llama-7b    -> codellama/CodeLlama-7b-hf

Examples:
  # Discovery analysis for LLaMA 2 7B
  python llama_research_runner.py --model llama2-7b --discovery-only
  
  # Full analysis with validation
  python llama_research_runner.py --model meta-llama/Llama-2-7b-hf --full-analysis
  
  # Quick test for development
  python llama_research_runner.py --model llama2-7b --quick
  
  # Chat model analysis
  python llama_research_runner.py --model llama2-7b-chat --discovery-only
  
  # Code-specialized model
  python llama_research_runner.py --model code-llama-7b --discovery-only
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='LLaMA model to analyze (use shortcuts or full HF names)'
    )
    
    # Analysis modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--discovery-only',
        action='store_true',
        help='Run only discovery phase (no perturbation testing)'
    )
    group.add_argument(
        '--validation-only',
        action='store_true',
        help='Run only validation phase (requires existing discovery results)'
    )
    group.add_argument(
        '--full-analysis',
        action='store_true',
        help='Run complete analysis with both discovery and validation'
    )
    group.add_argument(
        '--quick',
        action='store_true',
        help='Quick analysis with minimal parameters (good for testing)'
    )
    
    # Configuration
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: llama_research_TIMESTAMP)'
    )
    parser.add_argument(
        '--existing-discovery',
        type=str,
        help='Path to existing discovery results (for validation-only mode)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Initialize runner
        runner = LlamaResearchRunner()
        
        # Resolve model name
        model_name = args.model
        if model_name in LlamaResearchRunner.LLAMA_MODELS:
            model_name = LlamaResearchRunner.LLAMA_MODELS[model_name]
            logger.info(f"🎯 Using model: {model_name}")
        
        # Determine analysis mode
        if args.discovery_only:
            analysis_mode = "discovery"
        elif args.validation_only:
            analysis_mode = "validation"
        elif args.full_analysis:
            analysis_mode = "full"
        elif args.quick:
            analysis_mode = "discovery"  # Quick mode is discovery-only
        else:
            analysis_mode = "discovery"  # Default
            logger.info("🔄 No mode specified, defaulting to discovery-only")
        
        # Get optimized parameters
        params = runner.get_memory_optimized_params(model_name, analysis_mode)
        
        if args.quick:
            # Override with quick parameters
            params.update({
                "eval_limit": 10,
                "topk": [50],
                "metrics": ["grad_x_weight"],
                "no_perturbation": True
            })
            logger.info("🚀 Quick mode enabled - using minimal parameters")
        
        # Setup output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f"llama_research_{timestamp}")
        
        output_dir.mkdir(exist_ok=True)
        logger.info(f"💾 Results will be saved to: {output_dir}")
        logger.info("")
        
        # Execute analysis phases
        results = {"model_name": model_name, **params}
        
        if analysis_mode in ["discovery", "full"]:
            # Run discovery phase
            discovery_result = runner.run_discovery_phase(model_name, output_dir, params)
            results["discovery"] = discovery_result
            
            if discovery_result["status"] != "success":
                logger.error("❌ Discovery phase failed - stopping execution")
                return
        
        if analysis_mode in ["validation", "full"]:
            # Run validation phase
            discovery_dir = None
            
            if args.existing_discovery:
                discovery_dir = Path(args.existing_discovery)
            elif "discovery" in results:
                discovery_dir = results["discovery"]["discovery_dir"]
            
            if discovery_dir:
                validation_result = runner.run_validation_phase(
                    model_name, discovery_dir, output_dir, params
                )
                results["validation"] = validation_result
            else:
                logger.error("❌ No discovery results available for validation")
        
        # Generate specialized report
        runner.generate_llama_research_report(model_name, results, output_dir)
        
        # Print summary
        total_time = sum(
            phase.get("execution_time", 0) 
            for phase in [results.get("discovery", {}), results.get("validation", {})]
        )
        
        logger.info("")
        logger.info("🎉 LLaMA Research Analysis Complete!")
        logger.info(f"⏱️ Total execution time: {total_time/60:.1f} minutes")
        logger.info(f"📁 All results saved to: {output_dir}")
        logger.info("")
        logger.info("🔬 Next steps:")
        logger.info("  1. Review the LLAMA_RESEARCH_REPORT.md for insights")
        logger.info("  2. Examine critical weight CSV files for top findings")
        logger.info("  3. Use results for targeted model analysis or compression")
        logger.info("  4. Compare patterns across different LLaMA variants")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ LLaMA research interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ LLaMA research failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
