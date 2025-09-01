(.venv) ubuntu@192-222-58-149:~/nova/critical_weight_analysis$ python esa_runner.py   --model meta-llama/Llama-3.1-8B   --metric grad_x_weight   --mode per_layer   --topk 100   --max-samples 200   --device cuda   --save-plots   --out-dir outputs/p2/llama31_8b/gradxw_perlayer_k100
2025-09-01 16:27:14 | INFO | __main__ | 🚀 Starting Critical Weight Analysis - Enhanced Phase 1
2025-09-01 16:27:14 | INFO | __main__ | 📁 Output directory: outputs/p2/llama31_8b/gradxw_perlayer_k100
2025-09-01 16:27:14 | INFO | src.utils.manifest | Initialized experiment manifest: meta-llama/Llama-3.1-8B_grad_x_weight_k100
2025-09-01 16:27:14 | INFO | __main__ | 🤖 Loading model: meta-llama/Llama-3.1-8B
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  7.09it/s]
2025-09-01 16:27:19 | INFO | __main__ | 📚 Loading evaluation data
2025-09-01 16:27:19 | INFO | __main__ | Using default evaluation texts
2025-09-01 16:27:19 | INFO | __main__ | Using 100 evaluation texts
2025-09-01 16:27:19 | INFO | __main__ | 🔬 Running analysis across 1 seeds...
2025-09-01 16:27:19 | INFO | __main__ | 📊 Running analysis for seed 42 (1/1)
2025-09-01 16:27:19 | INFO | __main__ | 🔍 Computing grad_x_weight sensitivity...
2025-09-01 16:27:19 | INFO | src.sensitivity.metrics | Computing grad_x_weight sensitivity on 100 texts
Processing 100 texts: 100%|████████████████████████████████████████████| 4/4 [00:07<00:00,  1.90s/it, loss=4.2584, avg_loss=3.8937]
2025-09-01 16:27:26 | INFO | src.sensitivity.metrics | Average loss for sensitivity computation: 3.8937
2025-09-01 16:27:26 | INFO | src.sensitivity.metrics | 📊 Computing sensitivity for 226 parameters...
Computing sensitivity: 100%|████████████████████████████████████████████████████████████████████| 226/226 [00:01<00:00, 142.07it/s]
2025-09-01 16:27:28 | INFO | src.sensitivity.metrics | Computed sensitivity for 226 parameters
2025-09-01 16:27:28 | INFO | __main__ | ✅ Sensitivity analysis completed in 9.20s
2025-09-01 16:27:28 | INFO | __main__ | 📊 Processed 8,029,995,008 weights across 226 layers
2025-09-01 16:27:28 | INFO | __main__ | 🏆 Computing Top-100 ranking (per_layer mode)...
2025-09-01 16:27:28 | INFO | src.sensitivity.rank | Ranking top-100 weights (per_layer=True, global=False)
2025-09-01 16:27:28 | INFO | src.sensitivity.rank | 🏆 Ranking top-100 weights per layer...
Ranking layers: 100%|████████████████████████████████████████████████████████████████████████████| 226/226 [00:34<00:00,  6.62it/s]
2025-09-01 16:28:02 | INFO | __main__ | ✅ Ranking completed in 34.14s
2025-09-01 16:28:02 | INFO | __main__ | 🎯 Selected 22600 weights across 226 parameters
2025-09-01 16:28:02 | INFO | __main__ | 💾 Saving results to outputs/p2/llama31_8b/gradxw_perlayer_k100
2025-09-01 16:30:26 | INFO | __main__ | ✅ Saved 3 result files
2025-09-01 16:30:26 | INFO | __main__ | 📈 Generating visualization plots
2025-09-01 16:39:58 | INFO | src.utils.visualize | Saved sensitivity distribution plot to outputs/p2/llama31_8b/gradxw_perlayer_k100/plots/grad_x_weight_k100_sensitivity_distribution.png
2025-09-01 16:42:24 | INFO | src.utils.visualize | Saved layer comparison plot to outputs/p2/llama31_8b/gradxw_perlayer_k100/plots/grad_x_weight_k100_layer_comparison.png
2025-09-01 16:43:03 | INFO | src.utils.visualize | Saved sensitivity heatmap to outputs/p2/llama31_8b/gradxw_perlayer_k100/plots/grad_x_weight_k100_sensitivity_heatmap.png
2025-09-01 16:43:03 | INFO | src.utils.visualize | Saved 3 plots to outputs/p2/llama31_8b/gradxw_perlayer_k100/plots
2025-09-01 16:43:03 | INFO | __main__ | 🎉 Analysis completed successfully!
2025-09-01 16:43:03 | INFO | __main__ | 📊 Results summary:
2025-09-01 16:43:03 | INFO | __main__ |   • Total weights analyzed: 8,029,995,008
2025-09-01 16:43:03 | INFO | __main__ |   • Top-100 weights selected using grad_x_weight
2025-09-01 16:43:03 | INFO | __main__ |   • Ranking mode: per_layer
2025-09-01 16:43:03 | INFO | __main__ | 📁 All results saved to: outputs/p2/llama31_8b/gradxw_perlayer_k100
2025-09-01 16:43:03 | INFO | src.utils.manifest | Experiment manifest saved to: outputs/p2/llama31_8b/gradxw_perlayer_k100/experiment_manifest.json
2025-09-01 16:43:03 | INFO | __main__ | 📋 Experiment manifest saved to: outputs/p2/llama31_8b/gradxw_perlayer_k100/experiment_manifest.json




(.venv) ubuntu@192-222-58-149:~/nova/critical_weight_analysis$ python esa_runner.py \
  --model meta-llama/Llama-3.1-8B \
  --metric grad_x_weight \
  --mode per_layer \
  --topk 100 \
  --max-samples 200 \
  --device cuda \
  --save-plots \
  --out-dir outputs/p1_baseline/llama31_8b
2025-09-01 16:44:01 | INFO | __main__ | 🚀 Starting Critical Weight Analysis - Enhanced Phase 1
2025-09-01 16:44:01 | INFO | __main__ | 📁 Output directory: outputs/p1_baseline/llama31_8b
2025-09-01 16:44:02 | INFO | src.utils.manifest | Initialized experiment manifest: meta-llama/Llama-3.1-8B_grad_x_weight_k100
2025-09-01 16:44:02 | INFO | __main__ | 🤖 Loading model: meta-llama/Llama-3.1-8B
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  7.23it/s]
2025-09-01 16:44:06 | INFO | __main__ | 📚 Loading evaluation data
2025-09-01 16:44:06 | INFO | __main__ | Using default evaluation texts
2025-09-01 16:44:06 | INFO | __main__ | Using 100 evaluation texts
2025-09-01 16:44:06 | INFO | __main__ | 🔬 Running analysis across 1 seeds...
2025-09-01 16:44:06 | INFO | __main__ | 📊 Running analysis for seed 42 (1/1)
2025-09-01 16:44:06 | INFO | __main__ | 🔍 Computing grad_x_weight sensitivity...
2025-09-01 16:44:06 | INFO | src.sensitivity.metrics | Computing grad_x_weight sensitivity on 100 texts
Processing 100 texts: 100%|████████████████████████████████████████████| 4/4 [00:05<00:00,  1.32s/it, loss=4.2584, avg_loss=3.8937]
2025-09-01 16:44:11 | INFO | src.sensitivity.metrics | Average loss for sensitivity computation: 3.8937
2025-09-01 16:44:11 | INFO | src.sensitivity.metrics | 📊 Computing sensitivity for 226 parameters...
Computing sensitivity: 100%|████████████████████████████████████████████████████████████████████| 226/226 [00:01<00:00, 147.03it/s]
2025-09-01 16:44:13 | INFO | src.sensitivity.metrics | Computed sensitivity for 226 parameters
2025-09-01 16:44:13 | INFO | __main__ | ✅ Sensitivity analysis completed in 6.84s
2025-09-01 16:44:13 | INFO | __main__ | 📊 Processed 8,029,995,008 weights across 226 layers
2025-09-01 16:44:13 | INFO | __main__ | 🏆 Computing Top-100 ranking (per_layer mode)...
2025-09-01 16:44:13 | INFO | src.sensitivity.rank | Ranking top-100 weights (per_layer=True, global=False)
2025-09-01 16:44:13 | INFO | src.sensitivity.rank | 🏆 Ranking top-100 weights per layer...
Ranking layers: 100%|████████████████████████████████████████████████████████████████████████████| 226/226 [00:34<00:00,  6.63it/s]
2025-09-01 16:44:47 | INFO | __main__ | ✅ Ranking completed in 34.10s
2025-09-01 16:44:47 | INFO | __main__ | 🎯 Selected 22600 weights across 226 parameters
2025-09-01 16:44:47 | INFO | __main__ | 💾 Saving results to outputs/p1_baseline/llama31_8b
2025-09-01 16:47:10 | INFO | __main__ | ✅ Saved 3 result files
2025-09-01 16:47:10 | INFO | __main__ | 📈 Generating visualization plots
