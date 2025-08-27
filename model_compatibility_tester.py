#!/usr/bin/env python3
"""
Model Compatibility Test Suite
Tests various Hugging Face models with Critical Weight Analysis system

Usage:
    python model_compatibility_tester.py --all
    python model_compatibility_tester.py --category gpt2
    python model_compatibility_tester.py --model gpt2 --quick
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Test suite for model compatibility with critical weight analysis."""
    
    # Model categories with test configurations
    MODEL_CATEGORIES = {
        "gpt2": {
            "description": "GPT-2 Family (OpenAI)",
            "models": [
                ("gpt2", "124M"),
                ("gpt2-medium", "355M"),
                ("gpt2-large", "774M"),
                ("gpt2-xl", "1.5B")
            ]
        },
        "pythia": {
            "description": "Pythia Series (EleutherAI - Research Optimized)",
            "models": [
                ("EleutherAI/pythia-70m", "70M"),
                ("EleutherAI/pythia-160m", "160M"),
                ("EleutherAI/pythia-410m", "410M"),
                ("EleutherAI/pythia-1.4b", "1.4B"),
                ("EleutherAI/pythia-2.8b", "2.8B")
            ]
        },
        "opt": {
            "description": "OPT Series (Meta)",
            "models": [
                ("facebook/opt-125m", "125M"),
                ("facebook/opt-350m", "350M"),
                ("facebook/opt-1.3b", "1.3B"),
                ("facebook/opt-2.7b", "2.7B")
            ]
        },
        "code": {
            "description": "Code Models",
            "models": [
                ("microsoft/CodeGPT-small-py", "124M"),
                ("Salesforce/codegen-350M-mono", "350M"),
                ("microsoft/DialoGPT-small", "117M")
            ]
        },
        "small": {
            "description": "Small/Fast Models (Good for Testing)",
            "models": [
                ("distilgpt2", "82M"),
                ("EleutherAI/pythia-70m", "70M"),
                ("microsoft/DialoGPT-small", "117M")
            ]
        }
    }
    
    def __init__(self, timeout: int = 300, quick_mode: bool = False):
        """Initialize model tester.
        
        Args:
            timeout: Timeout per model test in seconds
            quick_mode: Use minimal parameters for faster testing
        """
        self.timeout = timeout
        self.quick_mode = quick_mode
        self.results = {}
        
        # Test parameters
        self.test_params = {
            "topk": 5 if quick_mode else 10,
            "eval_limit": 3 if quick_mode else 10,
            "metrics": ["grad_x_weight"] if quick_mode else ["grad_x_weight", "grad_squared"]
        }
    
    def test_single_model(self, model_name: str, params: str, category: str) -> Dict:
        """Test a single model for compatibility.
        
        Args:
            model_name: HuggingFace model identifier
            params: Parameter count description
            category: Model category
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"🔄 Testing: {model_name} ({category} - {params})")
        
        # Build command
        cmd = [
            sys.executable, "phase1_runner.py",
            "--model", model_name,
            "--metric", *self.test_params["metrics"],
            "--topk", str(self.test_params["topk"]),
            "--eval-limit", str(self.test_params["eval_limit"]),
            "--output", "test_results/",
            "--no-perturbation"  # Skip perturbation for compatibility testing
        ]
        
        start_time = time.time()
        
        try:
            # Run with timeout
            result = subprocess.run(
                cmd, 
                timeout=self.timeout,
                capture_output=True,
                text=True,
                check=True
            )
            
            execution_time = time.time() - start_time
            logger.info(f"  ✅ SUCCESS: {model_name} works perfectly ({execution_time:.1f}s)")
            
            return {
                "status": "success",
                "execution_time": execution_time,
                "model": model_name,
                "category": category,
                "params": params,
                "stdout": result.stdout[-500:],  # Last 500 chars
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            logger.warning(f"  ⏱️ TIMEOUT: {model_name} (exceeded {self.timeout}s)")
            return {
                "status": "timeout",
                "execution_time": self.timeout,
                "model": model_name,
                "category": category,
                "params": params,
                "error": f"Timed out after {self.timeout}s"
            }
            
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            logger.error(f"  ❌ ERROR: {model_name} has compatibility issues")
            logger.error(f"    Error code: {e.returncode}")
            
            return {
                "status": "error",
                "execution_time": execution_time,
                "model": model_name,
                "category": category,
                "params": params,
                "error": e.stderr or e.stdout,
                "return_code": e.returncode
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"  ❌ UNEXPECTED ERROR: {model_name} - {e}")
            
            return {
                "status": "unexpected_error",
                "execution_time": execution_time,
                "model": model_name,
                "category": category,
                "params": params,
                "error": str(e)
            }
    
    def test_category(self, category: str) -> List[Dict]:
        """Test all models in a category.
        
        Args:
            category: Category name from MODEL_CATEGORIES
            
        Returns:
            List of test results
        """
        if category not in self.MODEL_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        
        cat_info = self.MODEL_CATEGORIES[category]
        logger.info(f"🏷️ {cat_info['description']}")
        logger.info("=" * (len(cat_info['description']) + 4))
        
        results = []
        for model_name, params in cat_info["models"]:
            result = self.test_single_model(model_name, params, category)
            results.append(result)
            self.results[model_name] = result
        
        logger.info("")  # Empty line for readability
        return results
    
    def test_all_categories(self) -> Dict[str, List[Dict]]:
        """Test all model categories.
        
        Returns:
            Dictionary mapping category names to lists of results
        """
        logger.info("🤖 Model Compatibility Test Suite for Critical Weight Analysis")
        logger.info("=" * 64)
        logger.info("")
        
        if self.quick_mode:
            logger.info("🚀 Quick mode enabled - using minimal test parameters")
        
        logger.info("📊 Testing Model Categories:")
        logger.info("")
        
        all_results = {}
        
        for category in self.MODEL_CATEGORIES:
            category_results = self.test_category(category)
            all_results[category] = category_results
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, List[Dict]]) -> str:
        """Generate a summary report of test results.
        
        Args:
            results: Test results from test_all_categories
            
        Returns:
            Summary report as string
        """
        total_tests = sum(len(cat_results) for cat_results in results.values())
        successful = sum(
            len([r for r in cat_results if r["status"] == "success"])
            for cat_results in results.values()
        )
        timeouts = sum(
            len([r for r in cat_results if r["status"] == "timeout"])
            for cat_results in results.values()
        )
        errors = total_tests - successful - timeouts
        
        report = f"""
🏆 Model Compatibility Test Summary
=======================================

📊 Overall Results:
  • Total Models Tested: {total_tests}
  • ✅ Successful: {successful} ({successful/total_tests*100:.1f}%)
  • ⏱️ Timeouts: {timeouts} ({timeouts/total_tests*100:.1f}%)
  • ❌ Errors: {errors} ({errors/total_tests*100:.1f}%)

📋 Category Breakdown:
"""
        
        for category, cat_results in results.items():
            cat_info = self.MODEL_CATEGORIES[category]
            cat_successful = len([r for r in cat_results if r["status"] == "success"])
            cat_total = len(cat_results)
            
            report += f"\n  {cat_info['description']}:\n"
            report += f"    ✅ {cat_successful}/{cat_total} models compatible\n"
            
            # List successful models
            successful_models = [r["model"] for r in cat_results if r["status"] == "success"]
            if successful_models:
                report += f"    Working: {', '.join(successful_models[:3])}"
                if len(successful_models) > 3:
                    report += f" (+{len(successful_models)-3} more)"
                report += "\n"
        
        report += f"""
🔧 Recommended for Production:
  • Quick Testing: distilgpt2, EleutherAI/pythia-70m
  • Research: EleutherAI/pythia-410m, gpt2
  • Large Scale: EleutherAI/pythia-2.8b (if compatible)

⚡ Performance Notes:
  • Small models (<200M): ~30-60 seconds
  • Medium models (200M-1B): ~2-5 minutes  
  • Large models (>1B): May timeout in quick tests

🚀 Next Steps:
  • Use compatible models for critical weight analysis
  • Consider timeout increases for large models
  • Report compatibility issues to development team
"""
        
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Model Compatibility Test Suite for Critical Weight Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models (comprehensive)
  python model_compatibility_tester.py --all
  
  # Test specific category
  python model_compatibility_tester.py --category gpt2
  
  # Quick test for CI/development
  python model_compatibility_tester.py --quick --category small
  
  # Test single model
  python model_compatibility_tester.py --model gpt2
  
  # Extended timeout for large models
  python model_compatibility_tester.py --category pythia --timeout 600
        """
    )
    
    # Test scope
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--all',
        action='store_true',
        help='Test all model categories'
    )
    group.add_argument(
        '--category',
        type=str,
        choices=list(ModelTester.MODEL_CATEGORIES.keys()),
        help='Test specific model category'
    )
    group.add_argument(
        '--model',
        type=str,
        help='Test single model (HuggingFace identifier)'
    )
    
    # Test configuration
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use minimal parameters for faster testing'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per model in seconds (default: 300)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_compatibility_results',
        help='Output directory for test results'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = ModelTester(timeout=args.timeout, quick_mode=args.quick)
    
    try:
        if args.all:
            # Test all categories
            results = tester.test_all_categories()
            
            # Generate and save summary
            summary = tester.generate_summary_report(results)
            print(summary)
            
            # Save detailed results
            import json
            with open(output_dir / "compatibility_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            with open(output_dir / "summary_report.txt", 'w') as f:
                f.write(summary)
            
        elif args.category:
            # Test specific category
            results = tester.test_category(args.category)
            
            # Print summary for this category
            successful = len([r for r in results if r["status"] == "success"])
            total = len(results)
            print(f"\n✅ Category '{args.category}': {successful}/{total} models compatible")
            
        elif args.model:
            # Test single model
            # Find category for this model (for display purposes)
            category = "custom"
            for cat, info in ModelTester.MODEL_CATEGORIES.items():
                if any(model[0] == args.model for model in info["models"]):
                    category = cat
                    break
            
            result = tester.test_single_model(args.model, "unknown", category)
            
            if result["status"] == "success":
                print(f"\n✅ Model {args.model} is compatible!")
                print(f"   Execution time: {result['execution_time']:.1f}s")
            else:
                print(f"\n❌ Model {args.model} failed compatibility test")
                print(f"   Status: {result['status']}")
                if 'error' in result:
                    print(f"   Error: {result['error'][:200]}...")
        
        logger.info(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Testing failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
