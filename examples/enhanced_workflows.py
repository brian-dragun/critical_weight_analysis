#!/usr/bin/env python3
"""
Enhanced Critical Weight Analysis Workflow Examples

Demonstrates different analysis patterns and provides template code
for common research scenarios.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(cmd: list, description: str):
    """Run a command with error handling."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr[-500:])
        return False

def workflow_1_basic_enhanced():
    """Basic enhanced analysis with downstream tasks."""
    cmd = [
        "python", "phase1_runner_enhanced.py",
        "--model", "gpt2",
        "--metric", "grad_x_weight", 
        "--topk", "50",
        "--downstream-tasks",
        "--task-samples", "20",  # Reduced for speed
        "--weight-analysis",
        "--max-samples", "20",   # Reduced for speed
        "--save-plots",
        "--out-dir", "outputs/examples/basic_enhanced",
        "--verbose"
    ]
    
    return run_command(cmd, "Basic Enhanced Analysis")

def workflow_2_architecture_focus():
    """Architecture-focused analysis.""" 
    cmd = [
        "python", "phase1_runner_enhanced.py",
        "--model", "gpt2",
        "--metric", "magnitude",  # Non-gradient for speed
        "--topk", "50",
        "--architecture-analysis",
        "--weight-analysis",
        "--clustering-method", "kmeans",
        "--n-clusters", "5",
        "--max-samples", "15",
        "--save-plots",
        "--out-dir", "outputs/examples/architecture_focus",
        "--verbose"
    ]
    
    return run_command(cmd, "Architecture-Focused Analysis")

def workflow_3_advanced_perturbations():
    """Advanced perturbation experiments."""
    cmd = [
        "python", "phase1_runner_enhanced.py", 
        "--model", "gpt2",
        "--metric", "grad_x_weight",
        "--topk", "30",
        "--perturb", "sign_flip",
        "--controls", "random_k,bottom_k",
        "--advanced-perturbations", "adaptive_noise", "magnitude_scaling",
        "--perturb-scale", "0.3",
        "--max-samples", "15",
        "--save-plots",
        "--out-dir", "outputs/examples/advanced_perturbations",
        "--verbose"
    ]
    
    return run_command(cmd, "Advanced Perturbation Experiments")

def workflow_4_temporal_stability():
    """Multi-seed temporal stability study."""
    cmd = [
        "python", "phase1_runner_enhanced.py",
        "--model", "gpt2", 
        "--metric", "magnitude",
        "--topk", "50",
        "--seeds", "0,1,2",
        "--stability-check",
        "--temporal-stability", 
        "--weight-analysis",
        "--max-samples", "15",
        "--save-plots",
        "--out-dir", "outputs/examples/temporal_stability",
        "--verbose"
    ]
    
    return run_command(cmd, "Multi-Seed Temporal Stability Study")

def workflow_5_quick_development():
    """Quick development testing."""
    cmd = [
        "python", "phase1_runner_enhanced.py",
        "--model", "gpt2",
        "--metric", "magnitude", 
        "--topk", "20",
        "--max-samples", "10",
        "--weight-analysis",
        "--architecture-analysis", 
        "--save-plots",
        "--out-dir", "outputs/examples/quick_test",
        "--verbose"
    ]
    
    return run_command(cmd, "Quick Development Testing")

def analyze_results():
    """Analyze and compare results from workflows."""
    print(f"\n{'='*60}")
    print("📊 RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    examples_dir = Path("outputs/examples")
    if not examples_dir.exists():
        print("❌ No results found. Run workflows first.")
        return
    
    for workflow_dir in examples_dir.iterdir():
        if workflow_dir.is_dir():
            print(f"\n📁 {workflow_dir.name}:")
            
            # Check for enhanced analysis results
            enhanced_file = workflow_dir / "enhanced_analysis.json"
            if enhanced_file.exists():
                try:
                    with open(enhanced_file) as f:
                        data = json.load(f)
                    
                    print(f"  ✅ Enhanced analysis: {len(data)} components")
                    
                    # Summarize key findings
                    if "weight_analysis" in data:
                        wa = data["weight_analysis"]
                        if "summary_statistics" in wa:
                            print(f"  📊 Weight analysis: {len(wa['summary_statistics'])} layers analyzed")
                    
                    if "architecture_analysis" in data:
                        aa = data["architecture_analysis"]
                        if "summary_insights" in aa:
                            most_sensitive = aa["summary_insights"].get("most_sensitive_component", "unknown")
                            print(f"  🏗️ Most sensitive component: {most_sensitive}")
                    
                    if "downstream_tasks" in data:
                        dt = data["downstream_tasks"]
                        if "hellaswag_accuracy" in dt:
                            accuracy = dt["hellaswag_accuracy"]
                            print(f"  🎯 HellaSwag accuracy: {accuracy:.3f}")
                        if "lambada_accuracy" in dt:
                            accuracy = dt["lambada_accuracy"]
                            print(f"  🎯 LAMBADA accuracy: {accuracy:.3f}")
                
                except Exception as e:
                    print(f"  ❌ Error reading enhanced analysis: {e}")
            
            # Check for plots
            plots_dir = workflow_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                print(f"  📈 Generated {len(plot_files)} plots")

def main():
    """Run example workflows."""
    print("🚀 Critical Weight Analysis - Enhanced Example Workflows")
    print("=" * 60)
    
    # Ensure we're in the right directory
    if not Path("phase1_runner_enhanced.py").exists():
        print("❌ Please run this script from the critical_weight_analysis directory")
        sys.exit(1)
    
    # Create examples output directory
    Path("outputs/examples").mkdir(parents=True, exist_ok=True)
    
    workflows = [
        ("Basic Enhanced Analysis", workflow_1_basic_enhanced),
        ("Architecture Focus", workflow_2_architecture_focus), 
        ("Advanced Perturbations", workflow_3_advanced_perturbations),
        ("Temporal Stability", workflow_4_temporal_stability),
        ("Quick Development", workflow_5_quick_development)
    ]
    
    # Run workflows
    results = {}
    for name, workflow_func in workflows:
        print(f"\n🔄 Running: {name}")
        success = workflow_func()
        results[name] = success
        
        if not success:
            print(f"⚠️ {name} failed, continuing with next workflow...")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 WORKFLOW SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:30} {status}")
    
    # Analyze results
    analyze_results()
    
    print(f"\n{'='*60}")
    print("🎉 COMPLETION")
    print(f"{'='*60}")
    print("📁 All results saved in outputs/examples/")
    print("🔍 Check enhanced_analysis.json files for detailed analysis")
    print("📊 View plots/ directories for visualizations")
    print("")
    print("💡 Next steps:")
    print("   1. Compare weight_analysis across different metrics")
    print("   2. Examine architecture_analysis for component insights")
    print("   3. Review temporal_stability for ranking consistency")
    print("   4. Use advanced_perturbations for robustness analysis")

if __name__ == "__main__":
    main()
