#!/bin/bash

# Setup script for Neural Network Scorecard Backend
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Neural Network Scorecard Backend..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "📦 Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "📦 Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Success message
echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""

