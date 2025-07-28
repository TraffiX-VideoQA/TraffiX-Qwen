#!/bin/bash

# TUMTraffic-QA Installation Script
# This script sets up the environment for TUMTraffic-QA evaluation

echo "🚗 Installing TUMTraffic-QA..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📦 Creating conda environment..."
    conda create -n tumtraffic python=3.10 -y
    conda activate tumtraffic
else
    echo "⚠️  Conda not found. Please install conda or use a virtual environment."
    echo "You can create a virtual environment with: python -m venv tumtraffic"
    echo "Then activate it with: source tumtraffic/bin/activate (Linux/Mac) or tumtraffic\\Scripts\\activate (Windows)"
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing TUMTraffic-QA package..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎉 TUMTraffic-QA is ready to use!"
echo ""
echo "Quick start:"
echo "1. Activate the environment: conda activate tumtraffic"
echo "2. Run evaluation: python eval_tumtraf.py --help"
echo "3. Run inference: python scripts/TUMTrafficgpt_inference.py --help"
echo ""
echo "For more information, see the README.md file." 