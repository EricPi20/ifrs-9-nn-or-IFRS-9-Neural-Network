#!/bin/bash

# Quick start script for Neural Network Scorecard Backend
# This script activates the virtual environment and starts the backend server

cd "$(dirname "$0")/nn-scorecard/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the installation steps first:"
    echo "  1. cd nn-scorecard/backend"
    echo "  2. python3 -m venv venv"
    echo "  3. source venv/bin/activate"
    echo "  4. pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo "Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "🚀 Starting Neural Network Scorecard Backend..."
echo "📍 Backend will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

