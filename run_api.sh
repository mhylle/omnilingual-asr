#!/bin/bash
# Quick script to run API without full setup
# Use this if you already have dependencies installed

echo "🚀 Starting Omnilingual ASR API..."
echo ""
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Try to activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
python3 api.py
