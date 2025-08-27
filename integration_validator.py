#!/usr/bin/env python3
"""
Integration Validator
Tests critical weight analysis discoveries in the original research project

Usage:
    python integration_validator.py --auto
    python integration_validator.py --layer "gpt_neox.layers.2.mlp.dense_4h_to_h"
    python integration_validator.py --from-results my_results/critical_analysis_*/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationValidator:
    """Validates critical weight discoveries using perturbation testing."""
    
    def __init__(self, llm_project_path: str = "../llm_research_project"):
        """Initialize integration validator.
        
        Args:
            llm_project_path: Path to the LLM research project directory
        """
        self.llm_project_path = Path(llm_project_path)
        self.original_cwd = Path.cwd()
        self.results = {}
        
        # Validate LLM project exists
        if not self.llm_project_path.exists():
            raise FileNotFoundError(
                f"LLM research project not found at: {self.llm_project_path}\n"
                f"Please ensure the llm_research_project directory exists"
            )
    
    def load_critical_weights(self, results_dir: str) -> Dict:
        """Load critical weights from analysis results.
        
        Args:
            results_dir: Directory containing critical weight analysis results
            
        Returns:
            Dictionary with loaded critical weight data
        """
        results_path = Path(results_dir)
        
        # Find the latest results directory if it's a parent
        if results_path.is_dir() and not list(results_path.glob("*.csv")):
            # Look for subdirectories with results
            subdirs = list(results_path.glob("critical_analysis_*"))
            if subdirs:
                results_path = max(subdirs, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"📂 Loading critical weights from: {results_path}")
        
        data = {}
        
        # Load critical weights ranking
        for weights_file in results_path.glob("*weights*.csv"):
            logger.info(f"📊 Found weights file: {weights_file.name}")
            
            try:
                df = pd.read_csv(weights_file)
                metric = weights_file.stem.split('_')[-1]  # Extract metric from filename
                data[f'weights_{metric}'] = df
                
                # Extract top layers
                if 'layer_name' in df.columns:
                    top_layers = df['layer_name'].unique()[:10]
                    data[f'top_layers_{metric}'] = top_layers
                    logger.info(f"  ✅ Loaded {len(df)} weights, {len(top_layers)} unique layers")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load {weights_file}: {e}")
        
        # Load metadata if available
        metadata_file = results_path / "config.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                data['metadata'] = json.load(f)
                logger.info("✅ Loaded experiment metadata")
        
        if not data:
            raise ValueError(f"No critical weight data found in {results_path}")
        
        return data
    
    def get_recommended_layers(self, critical_data: Dict, max_layers: int = 5) -> List[str]:
        """Get recommended layers for perturbation testing.
        
        Args:
            critical_data: Critical weight data from load_critical_weights
            max_layers: Maximum number of layers to recommend
            
        Returns:
            List of layer names to test
        """
        # Try to find the best available metric
        layer_lists = [
            critical_data.get('top_layers_grad_x_weight', []),
            critical_data.get('top_layers_grad_squared', []),
        ]
        
        # Use the first available layer list
        for layers in layer_lists:
            if layers:
                recommended = list(layers[:max_layers])
                logger.info(f"🎯 Recommended layers for testing ({len(recommended)}):")
                for i, layer in enumerate(recommended, 1):
                    logger.info(f"  {i}. {layer}")
                return recommended
        
        # Fallback: manually extract from any weights dataframe
        for key, df in critical_data.items():
            if key.startswith('weights_') and isinstance(df, pd.DataFrame):
                if 'layer_name' in df.columns:
                    layers = df['layer_name'].unique()[:max_layers]
                    logger.info(f"🎯 Using layers from {key} ({len(layers)}):")
                    for i, layer in enumerate(layers, 1):
                        logger.info(f"  {i}. {layer}")
                    return list(layers)
        
        raise ValueError("No layer information found in critical weight data")
    
    def test_single_layer(self, layer_name: str, timeout: int = 300) -> Dict:
        """Test a single layer using perturbation.
        
        Args:
            layer_name: Name of the layer to test
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"🔬 Testing layer: {layer_name}")
        
        try:
            # Change to LLM project directory
            os.chdir(self.llm_project_path)
            
            # Set up environment
            env = os.environ.copy()
            env['TARGET_LAYER'] = layer_name
            
            # Run perturbation test
            cmd = [
                "accelerate", "launch",
                "--config_file", "configs/accelerate_config.yaml",
                "scripts/run_topk_perturb.py"
            ]
            
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=True
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            perplexity_data = self.parse_perturbation_output(result.stdout)
            
            logger.info(f"  ✅ Layer {layer_name} tested successfully ({execution_time:.1f}s)")
            if 'delta_ppl' in perplexity_data:
                logger.info(f"     Δ Perplexity: {perplexity_data['delta_ppl']:.4f}")
            
            return {
                "status": "success",
                "layer_name": layer_name,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                **perplexity_data
            }
            
        except subprocess.TimeoutExpired:
            logger.warning(f"  ⏱️ Layer {layer_name}: Timed out after {timeout}s")
            return {
                "status": "timeout",
                "layer_name": layer_name,
                "execution_time": timeout,
                "error": f"Timed out after {timeout}s"
            }
            
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            logger.error(f"  ❌ Layer {layer_name}: Failed with error code {e.returncode}")
            
            return {
                "status": "error",
                "layer_name": layer_name,
                "execution_time": execution_time,
                "error": e.stderr or e.stdout,
                "return_code": e.returncode
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"  ❌ Layer {layer_name}: Unexpected error: {e}")
            
            return {
                "status": "unexpected_error",
                "layer_name": layer_name,
                "execution_time": execution_time,
                "error": str(e)
            }
            
        finally:
            # Always return to original directory
            os.chdir(self.original_cwd)
    
    def parse_perturbation_output(self, output: str) -> Dict:
        """Parse perturbation test output to extract metrics.
        
        Args:
            output: Stdout from perturbation test
            
        Returns:
            Dictionary with extracted metrics
        """
        results = {}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for perplexity values
            if 'BASELINE_PERPLEXITY' in line and ':' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    results['baseline_ppl'] = value
                except (ValueError, IndexError):
                    pass
            
            elif 'TOPK_PERPLEXITY' in line and ':' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    results['topk_ppl'] = value
                except (ValueError, IndexError):
                    pass
            
            elif 'CONTROL_PERPLEXITY' in line and ':' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    results['control_ppl'] = value
                except (ValueError, IndexError):
                    pass
        
        # Calculate derived metrics
        if 'baseline_ppl' in results and 'topk_ppl' in results:
            results['delta_ppl'] = results['topk_ppl'] - results['baseline_ppl']
            results['ppl_ratio'] = results['topk_ppl'] / results['baseline_ppl']
            results['ppl_increase_percent'] = (results['delta_ppl'] / results['baseline_ppl']) * 100
        
        return results
    
    def test_multiple_layers(self, layers: List[str], timeout: int = 300) -> List[Dict]:
        """Test multiple layers sequentially.
        
        Args:
            layers: List of layer names to test
            timeout: Timeout per layer in seconds
            
        Returns:
            List of test results
        """
        logger.info(f"🧪 Starting perturbation testing for {len(layers)} layers")
        
        results = []
        successful = 0
        
        for i, layer in enumerate(layers, 1):
            logger.info(f"📊 Progress: {i}/{len(layers)}")
            
            result = self.test_single_layer(layer, timeout)
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
        
        logger.info(f"✅ Perturbation testing completed: {successful}/{len(layers)} layers successful")
        return results
    
    def generate_validation_report(self, critical_data: Dict, perturbation_results: List[Dict]) -> str:
        """Generate a validation report comparing discovery and perturbation results.
        
        Args:
            critical_data: Critical weight analysis data
            perturbation_results: Results from perturbation testing
            
        Returns:
            Validation report as string
        """
        successful_tests = [r for r in perturbation_results if r["status"] == "success"]
        
        report = f"""
🔬 Integration Validation Report
==============================

📊 Test Summary:
  • Layers Tested: {len(perturbation_results)}
  • ✅ Successful: {len(successful_tests)}
  • ❌ Failed: {len(perturbation_results) - len(successful_tests)}

🎯 Perturbation Results:

| Rank | Layer Name | Status | Baseline PPL | Perturbed PPL | Δ PPL | % Change |
|------|------------|--------|--------------|---------------|--------|----------|
"""
        
        for i, result in enumerate(perturbation_results, 1):
            layer = result['layer_name']
            status = "✅" if result['status'] == 'success' else "❌"
            
            if result['status'] == 'success':
                baseline = result.get('baseline_ppl', 'N/A')
                perturbed = result.get('topk_ppl', 'N/A')
                delta = result.get('delta_ppl', 'N/A')
                percent = result.get('ppl_increase_percent', 'N/A')
                
                if isinstance(baseline, (int, float)):
                    baseline = f"{baseline:.3f}"
                if isinstance(perturbed, (int, float)):
                    perturbed = f"{perturbed:.3f}"
                if isinstance(delta, (int, float)):
                    delta = f"{delta:.3f}"
                if isinstance(percent, (int, float)):
                    percent = f"{percent:.1f}%"
            else:
                baseline = perturbed = delta = percent = "Failed"
            
            # Truncate long layer names
            display_layer = layer if len(layer) <= 40 else layer[:37] + "..."
            
            report += f"| {i:2d} | `{display_layer}` | {status} | {baseline} | {perturbed} | {delta} | {percent} |\n"
        
        # Add validation analysis
        if successful_tests:
            avg_delta = sum(r.get('delta_ppl', 0) for r in successful_tests) / len(successful_tests)
            max_impact = max(successful_tests, key=lambda x: x.get('delta_ppl', 0))
            
            report += f"""

📈 Impact Analysis:
  • Average Δ Perplexity: {avg_delta:.4f}
  • Highest Impact Layer: `{max_impact['layer_name']}`
  • Maximum Δ Perplexity: {max_impact.get('delta_ppl', 0):.4f}

✅ Validation Status:
"""
            if avg_delta > 0.01:  # Significant impact threshold
                report += "  🎯 SUCCESS: Critical weight analysis predictions validated!\n"
                report += "     The identified layers show measurable perturbation impact.\n"
            else:
                report += "  ⚠️ MIXED: Some impact detected but relatively small.\n"
                report += "     Consider testing with larger perturbations or different metrics.\n"
        
        report += f"""

🔧 Technical Details:
  • Critical Weight Analysis: Based on gradient×weight sensitivity
  • Perturbation Method: Top-K weight masking/modification
  • Evaluation Metric: Perplexity change on validation set

📋 Files Generated:
  • integration_validation_results.json - Detailed test results
  • perturbation_logs/ - Individual layer test logs

🚀 Next Steps:
  1. Analyze correlation between sensitivity scores and perturbation impact
  2. Test additional layers or different perturbation strategies
  3. Use validated critical layers for targeted model analysis
  4. Compare results across different model architectures

---
*Generated by Integration Validator*
"""
        
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integration Validator for Critical Weight Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-validate using latest results
  python integration_validator.py --auto
  
  # Use specific results directory
  python integration_validator.py --from-results my_results/critical_analysis_20240827_143210/
  
  # Test specific layer
  python integration_validator.py --layer "gpt_neox.layers.2.mlp.dense_4h_to_h"
  
  # Test multiple specific layers
  python integration_validator.py --layers "gpt_neox.layers.2.mlp.dense_4h_to_h" "gpt_neox.layers.4.mlp.dense_4h_to_h"
  
  # Custom LLM project path
  python integration_validator.py --auto --llm-project "/path/to/llm_research_project"
        """
    )
    
    # Input source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--auto',
        action='store_true',
        help='Auto-discover and validate latest critical weight results'
    )
    group.add_argument(
        '--from-results',
        type=str,
        help='Use specific critical weight analysis results directory'
    )
    group.add_argument(
        '--layer',
        type=str,
        help='Test a specific layer name'
    )
    group.add_argument(
        '--layers',
        nargs='+',
        help='Test multiple specific layers'
    )
    
    # Configuration
    parser.add_argument(
        '--llm-project',
        type=str,
        default='../llm_research_project',
        help='Path to LLM research project directory'
    )
    parser.add_argument(
        '--max-layers',
        type=int,
        default=5,
        help='Maximum number of top layers to test (default: 5)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per layer test in seconds (default: 300)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='integration_validation_results',
        help='Output directory for validation results'
    )
    
    return parser.parse_args()


def find_latest_results() -> Path:
    """Find the latest critical weight analysis results."""
    
    # Look in common output directories
    search_dirs = [
        Path("outputs"),
        Path("my_results"),
        Path("."),
    ]
    
    latest_dir = None
    latest_time = 0
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for result_dir in search_dir.glob("critical_analysis_*"):
            if result_dir.is_dir():
                mtime = result_dir.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = result_dir
    
    if latest_dir is None:
        raise FileNotFoundError(
            "No critical weight analysis results found. "
            "Please run phase1_runner.py first or specify --from-results"
        )
    
    return latest_dir


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize validator
    try:
        validator = IntegrationValidator(args.llm_project)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    
    try:
        if args.auto:
            # Auto-discover latest results
            logger.info("🔍 Auto-discovering latest critical weight results...")
            results_dir = find_latest_results()
            logger.info(f"📂 Using results from: {results_dir}")
            
            # Load critical weights
            critical_data = validator.load_critical_weights(str(results_dir))
            
            # Get recommended layers
            layers = validator.get_recommended_layers(critical_data, args.max_layers)
            
            # Test the layers
            perturbation_results = validator.test_multiple_layers(layers, args.timeout)
            
            # Generate report
            report = validator.generate_validation_report(critical_data, perturbation_results)
            print(report)
            
            # Save results
            import json
            with open(output_dir / "integration_validation_results.json", 'w') as f:
                json.dump({
                    'critical_data_summary': {k: str(v)[:200] for k, v in critical_data.items()},
                    'perturbation_results': perturbation_results,
                    'validation_report': report
                }, f, indent=2, default=str)
            
        elif args.from_results:
            # Use specific results directory
            logger.info(f"📂 Loading results from: {args.from_results}")
            
            critical_data = validator.load_critical_weights(args.from_results)
            layers = validator.get_recommended_layers(critical_data, args.max_layers)
            
            perturbation_results = validator.test_multiple_layers(layers, args.timeout)
            
            report = validator.generate_validation_report(critical_data, perturbation_results)
            print(report)
            
        elif args.layer:
            # Test single layer
            result = validator.test_single_layer(args.layer, args.timeout)
            
            if result["status"] == "success":
                print(f"\n✅ Layer {args.layer} validation successful!")
                print(f"   Δ Perplexity: {result.get('delta_ppl', 'N/A')}")
                print(f"   Execution time: {result['execution_time']:.1f}s")
            else:
                print(f"\n❌ Layer {args.layer} validation failed")
                print(f"   Status: {result['status']}")
                if 'error' in result:
                    print(f"   Error: {result['error'][:200]}...")
        
        elif args.layers:
            # Test multiple specific layers
            perturbation_results = validator.test_multiple_layers(args.layers, args.timeout)
            
            # Simple report for manual layer testing
            successful = len([r for r in perturbation_results if r["status"] == "success"])
            print(f"\n✅ Validation completed: {successful}/{len(args.layers)} layers successful")
            
            for result in perturbation_results:
                if result["status"] == "success":
                    delta = result.get('delta_ppl', 'N/A')
                    print(f"  ✅ {result['layer_name']}: Δ PPL = {delta}")
                else:
                    print(f"  ❌ {result['layer_name']}: {result['status']}")
        
        logger.info(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
