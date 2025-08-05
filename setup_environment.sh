#!/bin/bash

# Finding Donors - Environment Setup Script
# This script sets up the conda environment for the Finding Donors project

echo "🎯 Setting up Finding Donors project environment..."
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'finding_donors' with Python 3.8..."
conda create -n finding_donors python=3.8 -y

if [ $? -eq 0 ]; then
    echo "✅ Conda environment created successfully!"
else
    echo "❌ Failed to create conda environment"
    exit 1
fi

# Activate environment and install packages
echo "📚 Installing required packages..."
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate finding_donors

# Install packages using pip (faster than conda for these packages)
/opt/homebrew/anaconda3/envs/finding_donors/bin/pip install numpy pandas scikit-learn matplotlib seaborn jupyter ipython

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Failed to install packages"
    exit 1
fi

# Test installation
echo "🧪 Testing installation..."
/opt/homebrew/anaconda3/envs/finding_donors/bin/python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('✅ All packages working correctly!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To get started:"
    echo "1. conda activate finding_donors"
    echo "2. jupyter notebook finding_donors_solution.ipynb"
    echo ""
else
    echo "❌ Package testing failed"
    exit 1
fi
