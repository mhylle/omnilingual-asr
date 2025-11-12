#!/bin/bash
# Start the Omnilingual ASR API server

echo "Starting Omnilingual ASR API..."
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r api_requirements.txt

# Create uploads directory
mkdir -p uploads

# Start server
echo ""
echo "🚀 Starting API server..."
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""

python3 api.py
