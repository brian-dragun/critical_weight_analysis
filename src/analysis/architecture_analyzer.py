"""
Model architecture analysis for critical weight research.

Analyzes how weight sensitivity patterns relate to model architecture,
including attention heads, feed-forward layers, embeddings, and layer depth.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import re
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class ArchitectureAnalyzer:
    """Analyzes weight sensitivity patterns across model architecture components."""
    
    def __init__(self):
        self.component_patterns = {
            "attention_query": [r"\.q_proj", r"\.query", r"attention.*query"],
            "attention_key": [r"\.k_proj", r"\.key", r"attention.*key"],
            "attention_value": [r"\.v_proj", r"\.value", r"attention.*value"],
            "attention_output": [r"\.o_proj", r"\.out_proj", r"attention.*output"],
            "feedforward_up": [r"\.up_proj", r"\.fc1", r"\.w1", r"mlp.*up"],
            "feedforward_down": [r"\.down_proj", r"\.fc2", r"\.w2", r"mlp.*down"],
            "feedforward_gate": [r"\.gate_proj", r"\.w3", r"mlp.*gate"],
            "layer_norm": [r"\.ln", r"\.norm", r"layer_norm"],
            "embedding": [r"embed", r"wte", r"word_embedding"],
            "position_embedding": [r"pos", r"position"],
            "output_projection": [r"lm_head", r"output", r"classifier"]
        }
    
    def classify_parameters(self, model: torch.nn.Module) -> Dict[str, Dict[str, List[str]]]:
        """Classify model parameters by architectural component."""
        
        classification = defaultdict(lambda: defaultdict(list))
        
        for name, param in model.named_parameters():
            # Extract layer number if present
            layer_match = re.search(r'\.(\d+)\.', name)
            layer_num = int(layer_match.group(1)) if layer_match else -1
            
            # Classify by component type
            component_type = "other"
            for comp_type, patterns in self.component_patterns.items():
                if any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns):
                    component_type = comp_type
                    break
            
            classification[component_type][f"layer_{layer_num}"].append(name)
        
        return dict(classification)
    
    def analyze_component_sensitivity(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Analyze sensitivity patterns by architectural component."""
        
        classification = self.classify_parameters(model)
        component_analysis = {}
        
        for component_type, layers in classification.items():
            component_stats = {
                "layers": {},
                "overall_stats": {},
                "layer_depth_analysis": {}
            }
            
            all_scores = []
            layer_means = []
            layer_numbers = []
            
            for layer_key, param_names in layers.items():
                layer_scores = []
                
                for param_name in param_names:
                    if param_name in sensitivity_dict:
                        scores = sensitivity_dict[param_name].flatten().cpu().numpy()
                        layer_scores.extend(scores)
                        all_scores.extend(scores)
                
                if layer_scores:
                    layer_mean = np.mean(layer_scores)
                    layer_means.append(layer_mean)
                    
                    # Extract layer number
                    layer_num = int(layer_key.split('_')[1]) if '_' in layer_key else -1
                    if layer_num >= 0:
                        layer_numbers.append(layer_num)
                    
                    component_stats["layers"][layer_key] = {
                        "mean_sensitivity": float(layer_mean),
                        "std_sensitivity": float(np.std(layer_scores)),
                        "max_sensitivity": float(np.max(layer_scores)),
                        "min_sensitivity": float(np.min(layer_scores)),
                        "num_weights": len(layer_scores),
                        "param_names": param_names
                    }
            
            # Overall component statistics
            if all_scores:
                component_stats["overall_stats"] = {
                    "mean_sensitivity": float(np.mean(all_scores)),
                    "std_sensitivity": float(np.std(all_scores)),
                    "max_sensitivity": float(np.max(all_scores)),
                    "min_sensitivity": float(np.min(all_scores)),
                    "total_weights": len(all_scores),
                    "num_layers": len(layers)
                }
            
            # Layer depth analysis
            if len(layer_means) > 1 and len(layer_numbers) > 1:
                # Correlation between layer depth and sensitivity
                if len(layer_numbers) == len(layer_means):
                    correlation = np.corrcoef(layer_numbers, layer_means)[0, 1]
                    component_stats["layer_depth_analysis"] = {
                        "depth_sensitivity_correlation": float(correlation),
                        "early_layer_mean": float(np.mean([m for i, m in zip(layer_numbers, layer_means) if i < np.median(layer_numbers)])),
                        "late_layer_mean": float(np.mean([m for i, m in zip(layer_numbers, layer_means) if i >= np.median(layer_numbers)])),
                        "layer_range": (int(min(layer_numbers)), int(max(layer_numbers)))
                    }
            
            component_analysis[component_type] = component_stats
        
        return component_analysis
    
    def compare_attention_components(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Detailed analysis of attention mechanism components."""
        
        attention_components = ["attention_query", "attention_key", "attention_value", "attention_output"]
        component_analysis = self.analyze_component_sensitivity(sensitivity_dict, model)
        
        attention_comparison = {
            "component_rankings": {},
            "layer_wise_comparison": {},
            "sensitivity_ratios": {}
        }
        
        # Overall component ranking
        component_means = {}
        for comp in attention_components:
            if comp in component_analysis and "overall_stats" in component_analysis[comp]:
                component_means[comp] = component_analysis[comp]["overall_stats"].get("mean_sensitivity", 0.0)
        
        sorted_components = sorted(component_means.items(), key=lambda x: x[1], reverse=True)
        attention_comparison["component_rankings"] = {
            "ranking": [comp for comp, _ in sorted_components],
            "mean_sensitivities": component_means
        }
        
        # Layer-wise comparison
        if len(component_means) > 1:
            # Find common layers across components
            common_layers = set()
            for comp in attention_components:
                if comp in component_analysis:
                    comp_layers = set(component_analysis[comp]["layers"].keys())
                    if not common_layers:
                        common_layers = comp_layers
                    else:
                        common_layers = common_layers.intersection(comp_layers)
            
            for layer in common_layers:
                layer_comparison = {}
                for comp in attention_components:
                    if comp in component_analysis and layer in component_analysis[comp]["layers"]:
                        layer_comparison[comp] = component_analysis[comp]["layers"][layer]["mean_sensitivity"]
                
                if len(layer_comparison) > 1:
                    attention_comparison["layer_wise_comparison"][layer] = layer_comparison
        
        # Sensitivity ratios
        if len(component_means) >= 2:
            ratios = {}
            for i, (comp1, sens1) in enumerate(sorted_components):
                for comp2, sens2 in sorted_components[i+1:]:
                    if sens2 > 0:
                        ratios[f"{comp1}_vs_{comp2}"] = sens1 / sens2
            attention_comparison["sensitivity_ratios"] = ratios
        
        return attention_comparison
    
    def analyze_depth_patterns(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Analyze how sensitivity changes with layer depth."""
        
        # Extract layer information
        layer_sensitivities = defaultdict(list)
        
        for param_name, tensor in sensitivity_dict.items():
            layer_match = re.search(r'\.(\d+)\.', param_name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                mean_sensitivity = tensor.mean().item()
                layer_sensitivities[layer_num].append(mean_sensitivity)
        
        # Compute layer-wise statistics
        layer_stats = {}
        layer_numbers = []
        layer_means = []
        
        for layer_num, sensitivities in layer_sensitivities.items():
            layer_mean = np.mean(sensitivities)
            layer_stats[layer_num] = {
                "mean_sensitivity": float(layer_mean),
                "std_sensitivity": float(np.std(sensitivities)),
                "num_parameters": len(sensitivities)
            }
            layer_numbers.append(layer_num)
            layer_means.append(layer_mean)
        
        # Analyze depth trends
        depth_analysis = {
            "layer_stats": layer_stats,
            "depth_trends": {}
        }
        
        if len(layer_numbers) > 2:
            # Correlation with depth
            correlation = np.corrcoef(layer_numbers, layer_means)[0, 1]
            
            # Divide into early, middle, late layers
            sorted_layers = sorted(layer_numbers)
            third = len(sorted_layers) // 3
            
            early_layers = sorted_layers[:third] if third > 0 else [sorted_layers[0]]
            middle_layers = sorted_layers[third:2*third] if third > 0 else []
            late_layers = sorted_layers[2*third:] if third > 0 else [sorted_layers[-1]]
            
            early_mean = np.mean([layer_stats[l]["mean_sensitivity"] for l in early_layers])
            middle_mean = np.mean([layer_stats[l]["mean_sensitivity"] for l in middle_layers]) if middle_layers else 0
            late_mean = np.mean([layer_stats[l]["mean_sensitivity"] for l in late_layers])
            
            depth_analysis["depth_trends"] = {
                "depth_correlation": float(correlation),
                "early_layer_sensitivity": float(early_mean),
                "middle_layer_sensitivity": float(middle_mean) if middle_layers else None,
                "late_layer_sensitivity": float(late_mean),
                "sensitivity_pattern": self._classify_depth_pattern(early_mean, middle_mean if middle_layers else early_mean, late_mean),
                "layer_ranges": {
                    "early": early_layers,
                    "middle": middle_layers,
                    "late": late_layers
                }
            }
        
        return depth_analysis
    
    def _classify_depth_pattern(self, early: float, middle: float, late: float) -> str:
        """Classify the depth sensitivity pattern."""
        
        if early > middle and middle > late:
            return "decreasing"
        elif early < middle and middle < late:
            return "increasing"
        elif middle > early and middle > late:
            return "peak_middle"
        elif early > middle and late > middle:
            return "valley_middle"
        elif abs(early - late) < 0.1 * max(early, late):
            return "stable"
        else:
            return "irregular"
    
    def generate_architecture_report(
        self, 
        sensitivity_dict: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Generate comprehensive architectural analysis report."""
        
        report = {
            "component_analysis": self.analyze_component_sensitivity(sensitivity_dict, model),
            "attention_analysis": self.compare_attention_components(sensitivity_dict, model),
            "depth_analysis": self.analyze_depth_patterns(sensitivity_dict, model),
            "parameter_classification": self.classify_parameters(model)
        }
        
        # Add summary insights
        component_analysis = report["component_analysis"]
        most_sensitive_component = max(
            component_analysis.keys(),
            key=lambda k: component_analysis[k].get("overall_stats", {}).get("mean_sensitivity", 0)
        )
        
        report["summary_insights"] = {
            "most_sensitive_component": most_sensitive_component,
            "total_components_analyzed": len(component_analysis),
            "depth_pattern": report["depth_analysis"].get("depth_trends", {}).get("sensitivity_pattern", "unknown")
        }
        
        return report
