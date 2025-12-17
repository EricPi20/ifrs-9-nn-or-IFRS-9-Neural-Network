#!/bin/bash

# Simple startup script for Neural Network Scorecard
# Run this script to start both backend and frontend

echo "🚀 Starting Neural Network Scorecard Application..."
echo ""

# Check if we're in the right directory
if [ ! -d "nn-scorecard" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Start backend in background
echo "📦 Starting Backend Server..."
cd nn-scorecard/backend
source venv/bin/activate
nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../../backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Logs: tail -f backend.log"
cd ../..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting Frontend Server..."
cd nn-scorecard/frontend
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
cd ../..

echo ""
echo "✅ Servers are starting!"
echo ""
echo "📍 Frontend: http://localhost:5173"
echo "📍 Backend:  http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 To view backend logs: tail -f backend.log"
echo "🛑 To stop servers: pkill -f 'uvicorn|vite'"
echo ""
echo "Press Ctrl+C to stop this script (servers will continue running)"
echo ""

# Wait for user interrupt
wait

