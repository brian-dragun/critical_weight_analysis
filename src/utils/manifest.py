"""
Manifest logging utilities for reproducible research.

Tracks experimental configurations, environment details, and results
for full reproducibility of critical weight analysis experiments.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentManifest:
    """
    Experiment manifest logger for tracking all aspects of a research run.
    
    Captures configuration, environment, git state, library versions,
    and results for full reproducibility.
    """
    
    def __init__(self, experiment_name: str, output_dir: Union[str, Path]):
        """
        Initialize experiment manifest.
        
        Args:
            experiment_name: Name/identifier for this experiment
            output_dir: Directory to save manifest and results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest = {
            "experiment_name": experiment_name,
            "timestamp_start": datetime.now().isoformat(),
            "manifest_version": "1.0",
        }
        
        # Initialize sections
        self._log_environment()
        self._log_git_state()
        self._log_library_versions()
        
        logger.info(f"Initialized experiment manifest: {experiment_name}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experimental configuration."""
        self.manifest["config"] = config
        logger.debug(f"Logged configuration: {len(config)} parameters")
    
    def log_model_info(self, model_name: str, model_details: Optional[Dict] = None):
        """Log model information."""
        self.manifest["model"] = {
            "name": model_name,
            "details": model_details or {}
        }
        
        # Try to get model parameter count if model object is available
        if hasattr(model_details, 'get') and 'model_object' in model_details:
            try:
                model = model_details['model_object']
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.manifest["model"]["total_parameters"] = total_params
                self.manifest["model"]["trainable_parameters"] = trainable_params
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log dataset/calibration data information."""
        self.manifest["data"] = data_info
        logger.debug(f"Logged data info: {data_info}")
    
    def log_seeds(self, seeds: List[int]):
        """Log random seeds used."""
        self.manifest["seeds"] = {
            "random_seeds": seeds,
            "torch_seed": torch.initial_seed(),
            "numpy_seed": np.random.get_state()[1][0] if hasattr(np.random, 'get_state') else None
        }
    
    def log_hardware_info(self):
        """Log hardware configuration."""
        hardware_info = {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
        
        # GPU information
        if torch.cuda.is_available():
            hardware_info["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_cached": torch.cuda.memory_reserved(),
            }
        else:
            hardware_info["gpu"] = {"available": False}
        
        self.manifest["hardware"] = hardware_info
        logger.debug("Logged hardware information")
    
    def log_results(self, results: Dict[str, Any]):
        """Log experimental results."""
        if "results" not in self.manifest:
            self.manifest["results"] = {}
        
        self.manifest["results"].update(results)
        logger.debug(f"Logged results: {list(results.keys())}")
    
    def log_timing(self, phase: str, duration: float):
        """Log timing information for different phases."""
        if "timing" not in self.manifest:
            self.manifest["timing"] = {}
        
        self.manifest["timing"][phase] = {
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration)
        }
    
    def log_files(self, files: Dict[str, str]):
        """Log output files generated."""
        self.manifest["output_files"] = files
        logger.debug(f"Logged {len(files)} output files")
    
    def finalize(self):
        """Finalize manifest and save to disk."""
        self.manifest["timestamp_end"] = datetime.now().isoformat()
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.manifest["timestamp_start"])
        end_time = datetime.fromisoformat(self.manifest["timestamp_end"])
        total_duration = (end_time - start_time).total_seconds()
        
        self.manifest["total_duration_seconds"] = total_duration
        self.manifest["total_duration_formatted"] = self._format_duration(total_duration)
        
        # Save manifest
        manifest_path = self.output_dir / "experiment_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        logger.info(f"Experiment manifest saved to: {manifest_path}")
        return manifest_path
    
    def _log_environment(self):
        """Log environment variables and system info."""
        self.manifest["environment"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "user": os.environ.get("USER", "unknown"),
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if any(key in k.upper() for key in ["CUDA", "TORCH", "HF", "TRANSFORMERS", "PATH"])
            }
        }
        
        # Add SLURM information if available (for cluster environments)
        slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
        if slurm_vars:
            self.manifest["environment"]["slurm"] = slurm_vars
    
    def _log_git_state(self):
        """Log git repository state for reproducibility."""
        try:
            # Get git commit hash
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get git branch
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check for uncommitted changes
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            self.manifest["git"] = {
                "commit_hash": git_hash,
                "branch": git_branch,
                "has_uncommitted_changes": bool(git_status),
                "status": git_status if git_status else "clean"
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.manifest["git"] = {"error": "Git information not available"}
    
    def _log_library_versions(self):
        """Log versions of key libraries."""
        import importlib
        
        libraries = {
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        
        # Try to get versions of optional libraries
        optional_libs = ["transformers", "pandas", "sklearn", "matplotlib", "seaborn"]
        
        for lib_name in optional_libs:
            try:
                lib = importlib.import_module(lib_name)
                if hasattr(lib, "__version__"):
                    libraries[lib_name] = lib.__version__
            except ImportError:
                libraries[lib_name] = "not_installed"
        
        self.manifest["library_versions"] = libraries
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def create_manifest(
    experiment_name: str,
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    seeds: Optional[List[int]] = None
) -> ExperimentManifest:
    """
    Create and initialize an experiment manifest.
    
    Args:
        experiment_name: Name/identifier for this experiment
        output_dir: Directory to save manifest and results
        config: Experimental configuration dictionary
        model_name: Name of the model being used
        seeds: List of random seeds used
        
    Returns:
        ExperimentManifest instance
        
    Examples:
        >>> manifest = create_manifest(
        ...     "llama_7b_grad_weight_k100",
        ...     "outputs/experiment_1",
        ...     config={"metric": "grad_x_weight", "k": 100},
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     seeds=[42, 123, 456]
        ... )
        >>> # ... run experiment ...
        >>> manifest.log_results({"perplexity": 12.5, "accuracy": 0.85})
        >>> manifest.finalize()
    """
    manifest = ExperimentManifest(experiment_name, output_dir)
    
    if config:
        manifest.log_config(config)
    
    if model_name:
        manifest.log_model_info(model_name)
    
    if seeds:
        manifest.log_seeds(seeds)
    
    # Always log hardware info
    manifest.log_hardware_info()
    
    return manifest


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an existing experiment manifest.
    
    Args:
        manifest_path: Path to the manifest JSON file
        
    Returns:
        Manifest dictionary
        
    Examples:
        >>> manifest = load_manifest("outputs/experiment_1/experiment_manifest.json")
        >>> print(f"Experiment: {manifest['experiment_name']}")
        >>> print(f"Model: {manifest['model']['name']}")
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    logger.info(f"Loaded manifest for experiment: {manifest.get('experiment_name', 'unknown')}")
    return manifest


def compare_manifests(
    manifest1_path: Union[str, Path],
    manifest2_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Compare two experiment manifests to identify differences.
    
    Args:
        manifest1_path: Path to first manifest
        manifest2_path: Path to second manifest
        
    Returns:
        Dictionary with comparison results
        
    Examples:
        >>> comparison = compare_manifests("exp1/manifest.json", "exp2/manifest.json")
        >>> print(f"Config differences: {comparison['config_diff']}")
    """
    manifest1 = load_manifest(manifest1_path)
    manifest2 = load_manifest(manifest2_path)
    
    comparison = {
        "experiment1": manifest1.get("experiment_name", "unknown"),
        "experiment2": manifest2.get("experiment_name", "unknown"),
        "same_git_hash": manifest1.get("git", {}).get("commit_hash") == manifest2.get("git", {}).get("commit_hash"),
        "same_model": manifest1.get("model", {}).get("name") == manifest2.get("model", {}).get("name"),
        "config_diff": {},
        "library_diff": {},
    }
    
    # Compare configurations
    config1 = manifest1.get("config", {})
    config2 = manifest2.get("config", {})
    
    all_config_keys = set(config1.keys()) | set(config2.keys())
    for key in all_config_keys:
        val1 = config1.get(key, "<missing>")
        val2 = config2.get(key, "<missing>")
        if val1 != val2:
            comparison["config_diff"][key] = {"exp1": val1, "exp2": val2}
    
    # Compare library versions
    lib1 = manifest1.get("library_versions", {})
    lib2 = manifest2.get("library_versions", {})
    
    all_lib_keys = set(lib1.keys()) | set(lib2.keys())
    for key in all_lib_keys:
        val1 = lib1.get(key, "<missing>")
        val2 = lib2.get(key, "<missing>")
        if val1 != val2:
            comparison["library_diff"][key] = {"exp1": val1, "exp2": val2}
    
    return comparison
