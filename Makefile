# ESA Research Baseline Testing Makefile
# ======================================
# Quick commands for running baseline tests across model phases
# Usage: make smoke-phase1, make standard-llama, make extended-mixtral, etc.

.PHONY: help smoke standard extended clean-baselines status
.DEFAULT_GOAL := help

# Configuration
PYTHON := python
BASELINE_RUNNER := scripts/baseline_runner.py
OUTPUT_DIR := outputs/baselines
CONFIG_FILE := baselines_config.yaml

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)ESA Research Baseline Testing Commands$(RESET)"
	@echo "======================================"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make smoke-phase1      # Test core models (fast)"
	@echo "  make generate-docs     # Create documentation from results"
	@echo "  make standard-core     # Full baseline on core models"
	@echo "  make extended-all      # Complete evaluation suite"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# =============================================================================
# SMOKE TESTS (Quick validation - 1-5 minutes per model)
# =============================================================================

smoke-phase1: ## Run smoke tests on all Phase 1 core models
	@echo "$(YELLOW)Running smoke tests on Phase 1 core models...$(RESET)"
	$(PYTHON) $(BASELINE_RUNNER) --model meta-llama/Llama-3.1-8B --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model mistralai/Mistral-7B-v0.3 --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model microsoft/Phi-3-mini-4k-instruct --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model EleutherAI/pythia-1.4b --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16

smoke-llama: ## Smoke test Llama 3.1 8B only
	$(PYTHON) $(BASELINE_RUNNER) --model meta-llama/Llama-3.1-8B --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16

smoke-mistral: ## Smoke test Mistral 7B only  
	$(PYTHON) $(BASELINE_RUNNER) --model mistralai/Mistral-7B-v0.3 --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16

smoke-all: ## Run smoke tests on ALL models (Phase 1-3)
	@echo "$(YELLOW)Running smoke tests on all models...$(RESET)"
	make smoke-phase1
	make smoke-phase2
	make smoke-phase3

# =============================================================================
# STANDARD BASELINES (Full evaluation - 15-30 minutes per model)
# =============================================================================

standard-core: ## Standard baseline on Phase 1 core models
	@echo "$(YELLOW)Running standard baselines on core models...$(RESET)"
	$(PYTHON) $(BASELINE_RUNNER) --model meta-llama/Llama-3.1-8B --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model mistralai/Mistral-7B-v0.3 --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model microsoft/Phi-3-mini-4k-instruct --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16

standard-llama: ## Standard baseline for Llama 3.1 8B
	$(PYTHON) $(BASELINE_RUNNER) --model meta-llama/Llama-3.1-8B --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 8192 --seeds 1337 123 999 --dtype bf16

standard-pythia: ## Standard baseline for Pythia scaling suite
	$(PYTHON) $(BASELINE_RUNNER) --model EleutherAI/pythia-410m --baseline standard --datasets wikitext-103 c4 --context-lens 1024 2048 --seeds 1337 123 999 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model EleutherAI/pythia-1.4b --baseline standard --datasets wikitext-103 c4 --context-lens 1024 2048 --seeds 1337 123 999 --dtype bf16
	$(PYTHON) $(BASELINE_RUNNER) --model EleutherAI/pythia-6.9b --baseline standard --datasets wikitext-103 c4 --context-lens 1024 2048 --seeds 1337 123 999 --dtype bf16

standard-all: ## Standard baseline on all models
	make standard-core
	make standard-pythia
	$(PYTHON) $(BASELINE_RUNNER) --model google/gemma-2-9b --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16

# =============================================================================
# EXTENDED BASELINES (Full research suite - 1-3 hours per model)
# =============================================================================

extended-llama: ## Extended baseline for Llama (with long-context and zero-shot)
	$(PYTHON) $(BASELINE_RUNNER) --model meta-llama/Llama-3.1-8B --baseline extended --datasets wikitext-103 openwebtext --context-lens 1024 4096 --long-context 8192 16384 32768 --seeds 1337 123 999 --eval-suites hellaswag piqa boolq arc_e --dtype bf16

extended-mixtral: ## Extended baseline for Mixtral (MoE analysis)
	$(PYTHON) $(BASELINE_RUNNER) --model mistralai/Mixtral-8x7B-v0.1 --baseline extended --datasets wikitext-103 openwebtext --context-lens 1024 4096 --long-context 8192 16384 --seeds 1337 123 999 --eval-suites hellaswag piqa boolq --dtype bf16

extended-core: ## Extended baseline on core models only
	make extended-llama
	$(PYTHON) $(BASELINE_RUNNER) --model mistralai/Mistral-7B-v0.3 --baseline extended --datasets wikitext-103 openwebtext --context-lens 1024 4096 --seeds 1337 123 999 --eval-suites hellaswag piqa boolq --dtype bf16

# =============================================================================
# PHASE-BASED WORKFLOWS
# =============================================================================

phase1: ## Complete Phase 1: smoke + standard on core models
	@echo "$(GREEN)Starting Phase 1: Core Models$(RESET)"
	make smoke-phase1
	make standard-core
	@echo "$(GREEN)Phase 1 complete!$(RESET)"

phase2: ## Phase 2: scaling validation
	@echo "$(GREEN)Starting Phase 2: Scale Validation$(RESET)"
	make standard-pythia
	$(PYTHON) $(BASELINE_RUNNER) --model google/gemma-2-9b --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16
	@echo "$(GREEN)Phase 2 complete!$(RESET)"

phase3: ## Phase 3: architecture diversity
	@echo "$(GREEN)Starting Phase 3: Architecture Diversity$(RESET)"
	make extended-mixtral
	$(PYTHON) $(BASELINE_RUNNER) --model Qwen/Qwen2.5-14B --baseline standard --datasets wikitext-103 c4 --context-lens 1024 4096 --seeds 1337 123 999 --dtype bf16
	@echo "$(GREEN)Phase 3 complete!$(RESET)"

full-research: ## Complete research pipeline (all phases + extended)
	@echo "$(GREEN)Starting full research baseline pipeline...$(RESET)"
	make phase1
	make phase2  
	make phase3
	make extended-core
	@echo "$(GREEN)Full research pipeline complete!$(RESET)"

# =============================================================================
# DEVELOPMENT & TESTING
# =============================================================================

test-setup: ## Quick test to verify environment setup
	@echo "$(YELLOW)Testing environment setup...$(RESET)"
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	$(PYTHON) scripts/quick_test.py

fast-test: ## Ultra-fast test with TinyLlama
	$(PYTHON) $(BASELINE_RUNNER) --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --baseline smoke --datasets wikitext-2-raw-v1 --context-lens 1024 --seeds 1337 --dtype bf16

# =============================================================================
# DOCUMENTATION
# =============================================================================

# =============================================================================
# DOCUMENTATION
# =============================================================================

generate-docs: ## Generate baseline documentation (report + execution log)
	@echo "$(YELLOW)Generating dynamic baseline documentation...$(RESET)"
	$(PYTHON) scripts/generate_baseline_docs.py --baseline-dir $(OUTPUT_DIR) --output-dir outputs --timestamp
	@echo "$(GREEN)Documentation generated successfully!$(RESET)"
	@echo "$(BLUE)Files created:$(RESET)"
	@echo "  - outputs/baseline_report.md (current)"
	@echo "  - outputs/baseline_execution_log.md (current)"
	@echo "  - outputs/baseline_report_YYYYMMDD_HHMMSS.md (timestamped)"
	@echo "  - outputs/baseline_execution_log_YYYYMMDD_HHMMSS.md (timestamped)"

generate-docs-current: ## Generate current documentation only (no timestamp)
	@echo "$(YELLOW)Generating current baseline documentation...$(RESET)"
	$(PYTHON) scripts/generate_baseline_docs.py --baseline-dir $(OUTPUT_DIR) --output-dir outputs
	@echo "$(GREEN)Current documentation updated!$(RESET)"
	@echo "$(BLUE)Files updated:$(RESET)"
	@echo "  - outputs/baseline_report.md"
	@echo "  - outputs/baseline_execution_log.md"

# =============================================================================
# UTILITIES
# =============================================================================

status: ## Show baseline results status
	@echo "$(BLUE)Baseline Results Status$(RESET)"
	@echo "======================="
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		find $(OUTPUT_DIR) -name "manifest.json" | wc -l | xargs echo "Completed baselines:"; \
		find $(OUTPUT_DIR) -name "results_summary.json" -exec ls -la {} \; | tail -5; \
	else \
		echo "No baseline results found. Run 'make smoke-phase1' to start."; \
	fi

clean-baselines: ## Remove all baseline outputs (CAUTION!)
	@echo "$(RED)Removing all baseline outputs...$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(OUTPUT_DIR); \
		echo "$(GREEN)Baseline outputs cleared.$(RESET)"; \
	else \
		echo "$(YELLOW)Cancelled.$(RESET)"; \
	fi

plot-results: ## Generate summary plots from baseline results
	@echo "$(YELLOW)Generating baseline summary plots...$(RESET)"
	$(PYTHON) scripts/generate_research_report.py --baseline-dir $(OUTPUT_DIR) --output-dir outputs/plots

check-models: ## List all configured models
	@echo "$(BLUE)Configured Models$(RESET)"
	@echo "================="
	@echo "$(GREEN)Phase 1 (Core):$(RESET)"
	@echo "  - meta-llama/Llama-3.1-8B"
	@echo "  - mistralai/Mistral-7B-v0.3"  
	@echo "  - microsoft/Phi-3-mini-4k-instruct"
	@echo "  - EleutherAI/pythia-1.4b"
	@echo ""
	@echo "$(YELLOW)Phase 2 (Scale):$(RESET)"
	@echo "  - EleutherAI/pythia-410m"
	@echo "  - EleutherAI/pythia-6.9b"
	@echo "  - google/gemma-2-9b"
	@echo ""
	@echo "$(RED)Phase 3 (Diverse):$(RESET)"
	@echo "  - mistralai/Mixtral-8x7B-v0.1"
	@echo "  - Qwen/Qwen2.5-14B"
	@echo "  - TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =============================================================================
# EXAMPLES
# =============================================================================

example-commands: ## Show example command usage
	@echo "$(BLUE)Example Commands$(RESET)"
	@echo "==============="
	@echo ""
	@echo "$(GREEN)Quick start (5 minutes):$(RESET)"
	@echo "  make smoke-llama"
	@echo ""
	@echo "$(GREEN)Research baseline (30 minutes):$(RESET)"
	@echo "  make standard-llama"
	@echo ""
	@echo "$(GREEN)Full evaluation (2-3 hours):$(RESET)"
	@echo "  make extended-llama"
	@echo ""
	@echo "$(GREEN)Complete Phase 1 research:$(RESET)"
	@echo "  make phase1"
	@echo ""
	@echo "$(GREEN)Manual command example:$(RESET)"
	@echo "  $(PYTHON) $(BASELINE_RUNNER) \\"
	@echo "    --model meta-llama/Llama-3.1-8B \\"
	@echo "    --baseline standard \\"
	@echo "    --datasets wikitext-103 c4 \\"
	@echo "    --context-lens 1024 4096 \\"
	@echo "    --seeds 1337 123 999 \\"
	@echo "    --dtype bf16"
