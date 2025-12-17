#!/bin/bash

# Quick start script for Neural Network Scorecard Frontend
# This script starts the frontend development server

cd "$(dirname "$0")/nn-scorecard/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ Node modules not found!"
    echo "Please install dependencies first:"
    echo "  npm install"
    exit 1
fi

echo "🚀 Starting Neural Network Scorecard Frontend..."
echo "📍 Frontend will be available at: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev

