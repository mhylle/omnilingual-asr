#!/bin/bash
# Start the Angular Frontend on Port 4202

echo "Starting Omnilingual ASR Frontend..."
echo "====================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

echo "🚀 Starting development server..."
echo "   Port: 4202"
echo "   URL: http://localhost:4202"
echo "   Press Ctrl+C to stop"
echo ""

# Start the dev server
npm start
