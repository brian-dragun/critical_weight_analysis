#!/usr/bin/env bash
# Quick VM restoration script for Lambda Labs
# Run this when you spin up a new VM to quickly restore your environment

set -euo pipefail

echo "🚀 Lambda Labs VM Quick Setup for Critical Weight Analysis"
echo "=========================================================="

# Update system
echo "===> Updating system packages"
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential software-properties-common

# Install UV package manager
echo "===> Installing UV package manager"
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc || export PATH="$HOME/.local/bin:$PATH"

# Clone project
echo "===> Cloning critical weight analysis project"
mkdir -p ~/nova
cd ~/nova
if [ ! -d "critical_weight_analysis" ]; then
    git clone https://github.com/brian-dragun/critical_weight_analysis.git
fi
cd critical_weight_analysis

# Run main setup
echo "===> Running main setup script"
chmod +x setup/setup.sh
bash setup/setup.sh

echo ""
echo "🎉 Quick setup complete! Your Lambda Labs VM is ready for research."
echo ""
echo "Next steps:"
echo "1. Configure HuggingFace: huggingface-cli login"
echo "2. Test the system: python scripts/quick_test.py"
echo "3. Start research: python phase1_runner_enhanced.py --model gpt2 --metric magnitude --topk 10 --max-samples 5"
