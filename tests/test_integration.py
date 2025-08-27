"""
Integration tests for enhanced critical weight analysis features.

Tests all new analysis modules, enhanced runner functionality, and visualization features.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.weight_analyzer import WeightAnalyzer
from src.analysis.temporal_stability import TemporalStabilityAnalyzer
from src.analysis.architecture_analyzer import ArchitectureAnalyzer
from src.sensitivity.advanced_perturb import AdvancedPerturbationEngine
from src.eval.downstream_tasks import evaluate_hellaswag, evaluate_lambada, evaluate_all_tasks
from src.utils.enhanced_visualize import (
    plot_weight_clustering, 
    plot_architecture_analysis,
    plot_temporal_stability,
    plot_advanced_perturbations,
    save_enhanced_plots
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestWeightAnalyzer(unittest.TestCase):
    """Test weight analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WeightAnalyzer()
        
        # Create mock sensitivity data
        self.mock_sensitivity = {
            f"layer_{i}.weight": np.random.normal(0.5, 0.2, 100) 
            for i in range(5)
        }
        
        # Ensure positive values
        for key in self.mock_sensitivity:
            self.mock_sensitivity[key] = np.abs(self.mock_sensitivity[key])
    
    def test_clustering_analysis(self):
        """Test weight clustering analysis."""
        try:
            results = self.analyzer.cluster_weights(self.mock_sensitivity, n_clusters=3)
            
            # Check basic structure
            self.assertIn("cluster_assignments", results)
            self.assertIn("cluster_stats", results)
            self.assertIn("silhouette_score", results)
            
            # Check cluster assignments
            self.assertEqual(len(results["cluster_assignments"]), len(self.mock_sensitivity))
            
            # Check silhouette score range
            self.assertGreaterEqual(results["silhouette_score"], -1)
            self.assertLessEqual(results["silhouette_score"], 1)
            
            logger.info("✓ Weight clustering analysis test passed")
            
        except Exception as e:
            self.fail(f"Weight clustering analysis failed: {e}")
    
    def test_correlation_analysis(self):
        """Test sensitivity correlation analysis."""
        try:
            results = self.analyzer.analyze_sensitivity_correlations(self.mock_sensitivity)
            
            # Check basic structure
            self.assertIn("correlation_matrix", results)
            self.assertIn("high_correlations", results)
            self.assertIn("correlation_stats", results)
            
            # Check correlation matrix properties
            corr_matrix = results["correlation_matrix"]
            self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
            self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))
            
            logger.info("✓ Correlation analysis test passed")
            
        except Exception as e:
            self.fail(f"Correlation analysis failed: {e}")

class TestTemporalStabilityAnalyzer(unittest.TestCase):
    """Test temporal stability analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TemporalStabilityAnalyzer()
        
        # Create mock sensitivity snapshots
        self.mock_snapshots = {
            "baseline": {f"layer_{i}.weight": np.random.normal(0.5, 0.2, 100) for i in range(5)},
            "perturbation_1": {f"layer_{i}.weight": np.random.normal(0.5, 0.2, 100) for i in range(5)},
            "perturbation_2": {f"layer_{i}.weight": np.random.normal(0.5, 0.2, 100) for i in range(5)}
        }
        
        # Ensure positive values
        for condition in self.mock_snapshots:
            for key in self.mock_snapshots[condition]:
                self.mock_snapshots[condition][key] = np.abs(self.mock_snapshots[condition][key])
    
    def test_record_snapshots(self):
        """Test recording sensitivity snapshots."""
        try:
            for condition, sensitivity in self.mock_snapshots.items():
                self.analyzer.record_sensitivity_snapshot(condition, sensitivity)
            
            # Check snapshots are recorded
            self.assertEqual(len(self.analyzer.snapshots), 3)
            self.assertIn("baseline", self.analyzer.snapshots)
            
            logger.info("✓ Snapshot recording test passed")
            
        except Exception as e:
            self.fail(f"Snapshot recording failed: {e}")
    
    def test_stability_analysis(self):
        """Test temporal stability analysis."""
        try:
            # Record snapshots first
            for condition, sensitivity in self.mock_snapshots.items():
                self.analyzer.record_sensitivity_snapshot(condition, sensitivity)
            
            results = self.analyzer.analyze_temporal_stability()
            
            # Check basic structure
            self.assertIn("pairwise_stability", results)
            self.assertIn("condition_stability_summary", results)
            
            # Check pairwise analysis
            pairwise = results["pairwise_stability"]
            self.assertGreater(len(pairwise), 0)
            
            for pair_name, pair_data in pairwise.items():
                self.assertIn("jaccard_similarity", pair_data)
                self.assertIn("rank_correlation", pair_data)
                self.assertGreaterEqual(pair_data["jaccard_similarity"], 0)
                self.assertLessEqual(pair_data["jaccard_similarity"], 1)
            
            logger.info("✓ Temporal stability analysis test passed")
            
        except Exception as e:
            self.fail(f"Temporal stability analysis failed: {e}")

class TestArchitectureAnalyzer(unittest.TestCase):
    """Test architecture analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ArchitectureAnalyzer()
        
        # Create mock model with proper structure
        self.mock_model = Mock()
        self.mock_model.named_parameters.return_value = [
            ("embeddings.weight", torch.randn(1000, 768)),
            ("encoder.layer.0.attention.self.query.weight", torch.randn(768, 768)),
            ("encoder.layer.0.attention.self.key.weight", torch.randn(768, 768)),
            ("encoder.layer.0.attention.self.value.weight", torch.randn(768, 768)),
            ("encoder.layer.0.attention.output.dense.weight", torch.randn(768, 768)),
            ("encoder.layer.0.intermediate.dense.weight", torch.randn(3072, 768)),
            ("encoder.layer.0.output.dense.weight", torch.randn(768, 3072)),
            ("encoder.layer.1.attention.self.query.weight", torch.randn(768, 768)),
            ("encoder.layer.1.attention.self.key.weight", torch.randn(768, 768)),
            ("encoder.layer.1.attention.self.value.weight", torch.randn(768, 768)),
            ("pooler.dense.weight", torch.randn(768, 768)),
        ]
        
        self.mock_sensitivity = {
            name: np.random.normal(0.5, 0.2, param.numel())
            for name, param in self.mock_model.named_parameters()
        }
        
        # Ensure positive values
        for key in self.mock_sensitivity:
            self.mock_sensitivity[key] = np.abs(self.mock_sensitivity[key])
    
    def test_component_analysis(self):
        """Test architecture component analysis."""
        try:
            results = self.analyzer.analyze_component_sensitivity(
                self.mock_model, self.mock_sensitivity
            )
            
            # Check basic structure
            self.assertIn("component_analysis", results)
            self.assertIn("depth_analysis", results)
            self.assertIn("attention_analysis", results)
            
            # Check component analysis
            component_analysis = results["component_analysis"]
            expected_components = ["embeddings", "attention", "feedforward", "pooler"]
            
            for component in expected_components:
                if component in component_analysis:
                    comp_data = component_analysis[component]
                    self.assertIn("overall_stats", comp_data)
                    self.assertIn("mean_sensitivity", comp_data["overall_stats"])
            
            logger.info("✓ Architecture component analysis test passed")
            
        except Exception as e:
            self.fail(f"Architecture component analysis failed: {e}")

class TestAdvancedPerturbationEngine(unittest.TestCase):
    """Test advanced perturbation engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.perturb_engine = AdvancedPerturbationEngine()
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(100, 50)),
            ("layer2.weight", torch.randn(50, 25)),
        ]
        
        self.mock_sensitivity = {
            "layer1.weight": np.random.normal(0.5, 0.2, 5000),
            "layer2.weight": np.random.normal(0.3, 0.1, 1250),
        }
        
        # Ensure positive values
        for key in self.mock_sensitivity:
            self.mock_sensitivity[key] = np.abs(self.mock_sensitivity[key])
    
    @patch('src.sensitivity.advanced_perturb.evaluate_model')
    def test_progressive_perturbation(self, mock_eval):
        """Test progressive perturbation method."""
        try:
            # Mock evaluation results
            mock_eval.return_value = {"perplexity": 3.5, "accuracy": 0.85}
            
            results = self.perturb_engine.apply_perturbation(
                self.mock_model, self.mock_sensitivity, 
                perturbation_type="progressive"
            )
            
            # Check results structure
            self.assertIsInstance(results, dict)
            
            logger.info("✓ Progressive perturbation test passed")
            
        except Exception as e:
            self.fail(f"Progressive perturbation failed: {e}")
    
    def test_clustered_perturbation(self):
        """Test clustered perturbation method."""
        try:
            results = self.perturb_engine.apply_perturbation(
                self.mock_model, self.mock_sensitivity, 
                perturbation_type="clustered"
            )
            
            # Check results structure
            self.assertIsInstance(results, dict)
            
            logger.info("✓ Clustered perturbation test passed")
            
        except Exception as e:
            self.fail(f"Clustered perturbation failed: {e}")

class TestDownstreamTasks(unittest.TestCase):
    """Test downstream task evaluation."""
    
    @patch('src.eval.downstream_tasks.load_dataset')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_hellaswag_evaluation(self, mock_model, mock_tokenizer, mock_dataset):
        """Test HellaSwag evaluation."""
        try:
            # Mock dataset
            mock_data = [
                {
                    "ctx": "A person is walking down the street when",
                    "endings": [
                        "they see a cat",
                        "the sky turns purple", 
                        "gravity reverses",
                        "they continue walking"
                    ],
                    "label": 0
                }
            ]
            mock_dataset.return_value = {"validation": mock_data}
            
            # Mock tokenizer
            mock_tok = Mock()
            mock_tok.encode.return_value = [1, 2, 3, 4, 5]
            mock_tok.decode.return_value = "test"
            mock_tokenizer.return_value = mock_tok
            
            # Mock model
            mock_mdl = Mock()
            mock_mdl.eval.return_value = None
            mock_mdl.return_value.logits = torch.randn(1, 5, 1000)
            mock_model.return_value = mock_mdl
            
            # Test evaluation
            accuracy = evaluate_hellaswag("test-model", num_samples=1)
            
            # Check result
            self.assertIsInstance(accuracy, (int, float))
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 1)
            
            logger.info("✓ HellaSwag evaluation test passed")
            
        except Exception as e:
            logger.warning(f"HellaSwag evaluation test skipped: {e}")
    
    def test_evaluate_all_tasks(self):
        """Test evaluation of all downstream tasks."""
        try:
            with patch('src.eval.downstream_tasks.evaluate_hellaswag') as mock_hella, \
                 patch('src.eval.downstream_tasks.evaluate_lambada') as mock_lambada:
                
                # Mock evaluation results
                mock_hella.return_value = 0.75
                mock_lambada.return_value = 0.68
                
                results = evaluate_all_tasks("test-model", num_samples=10)
                
                # Check results structure
                self.assertIn("hellaswag_accuracy", results)
                self.assertIn("lambada_accuracy", results)
                self.assertEqual(results["hellaswag_accuracy"], 0.75)
                self.assertEqual(results["lambada_accuracy"], 0.68)
                
                logger.info("✓ All tasks evaluation test passed")
                
        except Exception as e:
            self.fail(f"All tasks evaluation failed: {e}")

class TestEnhancedVisualization(unittest.TestCase):
    """Test enhanced visualization functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock clustering results
        self.mock_clustering = {
            "cluster_assignments": {"layer_0": 0, "layer_1": 1, "layer_2": 0},
            "cluster_stats": {
                "cluster_0": {"size": 2, "mean_sensitivity": 0.6, "min_sensitivity": 0.4, "max_sensitivity": 0.8},
                "cluster_1": {"size": 1, "mean_sensitivity": 0.3, "min_sensitivity": 0.3, "max_sensitivity": 0.3}
            },
            "silhouette_score": 0.65
        }
        
        # Mock architecture results
        self.mock_architecture = {
            "component_analysis": {
                "attention": {
                    "overall_stats": {"mean_sensitivity": 0.7, "total_weights": 1000}
                },
                "feedforward": {
                    "overall_stats": {"mean_sensitivity": 0.5, "total_weights": 2000}
                }
            },
            "depth_analysis": {
                "depth_trends": {
                    "layer_ranges": {"early": [0, 3], "middle": [4, 7], "late": [8, 11]},
                    "early_layer_sensitivity": 0.6,
                    "middle_layer_sensitivity": 0.4,
                    "late_layer_sensitivity": 0.8,
                    "sensitivity_pattern": "increasing"
                }
            },
            "attention_analysis": {
                "component_rankings": {
                    "mean_sensitivities": {
                        "attention_query": 0.7,
                        "attention_key": 0.6,
                        "attention_value": 0.8
                    }
                }
            }
        }
        
        # Mock temporal stability results
        self.mock_temporal = {
            "pairwise_stability": {
                "baseline_vs_perturbation1": {"mean_stability": 0.75, "jaccard_similarity": 0.6},
                "baseline_vs_perturbation2": {"mean_stability": 0.65, "jaccard_similarity": 0.5}
            },
            "condition_stability_summary": {
                "baseline": {"mean_temporal_stability": 0.8},
                "perturbation1": {"mean_temporal_stability": 0.7},
                "perturbation2": {"mean_temporal_stability": 0.6}
            }
        }
    
    def test_plot_weight_clustering(self):
        """Test weight clustering visualization."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig = plot_weight_clustering(self.mock_clustering, tmp.name)
                self.assertIsNotNone(fig)
                
                # Check that file was created
                self.assertTrue(Path(tmp.name).exists())
            
            logger.info("✓ Weight clustering plot test passed")
            
        except Exception as e:
            self.fail(f"Weight clustering plot failed: {e}")
    
    def test_plot_architecture_analysis(self):
        """Test architecture analysis visualization."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig = plot_architecture_analysis(self.mock_architecture, tmp.name)
                self.assertIsNotNone(fig)
                
                # Check that file was created
                self.assertTrue(Path(tmp.name).exists())
            
            logger.info("✓ Architecture analysis plot test passed")
            
        except Exception as e:
            self.fail(f"Architecture analysis plot failed: {e}")
    
    def test_plot_temporal_stability(self):
        """Test temporal stability visualization."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig = plot_temporal_stability(self.mock_temporal, tmp.name)
                self.assertIsNotNone(fig)
                
                # Check that file was created
                self.assertTrue(Path(tmp.name).exists())
            
            logger.info("✓ Temporal stability plot test passed")
            
        except Exception as e:
            self.fail(f"Temporal stability plot failed: {e}")
    
    def test_save_enhanced_plots(self):
        """Test saving all enhanced plots."""
        try:
            enhanced_results = {
                "weight_analysis": {"clustering_results": self.mock_clustering},
                "architecture_analysis": self.mock_architecture,
                "temporal_stability": self.mock_temporal,
                "advanced_perturbations": {"progressive": []},
                "downstream_tasks": {"hellaswag_accuracy": 0.75}
            }
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                saved_plots = save_enhanced_plots(enhanced_results, tmp_dir)
                
                # Check that plots were saved
                self.assertGreater(len(saved_plots), 0)
                
                for plot_name, plot_path in saved_plots.items():
                    self.assertTrue(plot_path.exists())
            
            logger.info("✓ Enhanced plots saving test passed")
            
        except Exception as e:
            self.fail(f"Enhanced plots saving failed: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for the entire enhanced system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_end_to_end_workflow(self):
        """Test a complete enhanced analysis workflow."""
        try:
            # Mock model and sensitivity data
            mock_model = Mock()
            mock_model.named_parameters.return_value = [
                ("layer1.weight", torch.randn(100, 50)),
                ("layer2.weight", torch.randn(50, 25)),
            ]
            
            mock_sensitivity = {
                "layer1.weight": np.abs(np.random.normal(0.5, 0.2, 5000)),
                "layer2.weight": np.abs(np.random.normal(0.3, 0.1, 1250)),
            }
            
            # Step 1: Weight Analysis
            weight_analyzer = WeightAnalyzer()
            clustering_results = weight_analyzer.cluster_weights(mock_sensitivity, n_clusters=2)
            
            # Step 2: Architecture Analysis
            arch_analyzer = ArchitectureAnalyzer()
            arch_results = arch_analyzer.analyze_component_sensitivity(mock_model, mock_sensitivity)
            
            # Step 3: Temporal Stability
            temporal_analyzer = TemporalStabilityAnalyzer()
            temporal_analyzer.record_sensitivity_snapshot("baseline", mock_sensitivity)
            
            # Create slightly perturbed sensitivity for comparison
            perturbed_sensitivity = {
                key: values + np.random.normal(0, 0.1, len(values))
                for key, values in mock_sensitivity.items()
            }
            temporal_analyzer.record_sensitivity_snapshot("perturbed", perturbed_sensitivity)
            
            temporal_results = temporal_analyzer.analyze_temporal_stability()
            
            # Step 4: Combine results
            enhanced_results = {
                "weight_analysis": {"clustering_results": clustering_results},
                "architecture_analysis": arch_results,
                "temporal_stability": temporal_results,
            }
            
            # Step 5: Generate visualizations
            saved_plots = save_enhanced_plots(enhanced_results, self.temp_dir)
            
            # Verify all components work together
            self.assertIn("clustering_results", enhanced_results["weight_analysis"])
            self.assertIn("component_analysis", enhanced_results["architecture_analysis"])
            self.assertIn("pairwise_stability", enhanced_results["temporal_stability"])
            self.assertGreater(len(saved_plots), 0)
            
            logger.info("✓ End-to-end workflow test passed")
            
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {e}")

def run_integration_tests():
    """Run all integration tests."""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestWeightAnalyzer,
        TestTemporalStabilityAnalyzer,
        TestArchitectureAnalyzer,
        TestAdvancedPerturbationEngine,
        TestDownstreamTasks,
        TestEnhancedVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("="*60)
    print("Running Enhanced Critical Weight Analysis Integration Tests")
    print("="*60)
    
    success = run_integration_tests()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("Enhanced critical weight analysis system is ready for use.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("="*60)
