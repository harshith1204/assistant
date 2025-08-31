#!/bin/bash

# AI Research & Chat Assistant Startup Script

echo "🚀 Starting AI Research & Chat Assistant..."
echo "================================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your GROQ_API_KEY"
    echo ""
fi

# Create necessary directories
mkdir -p chroma_db
mkdir -p static

echo "📦 Installing/Updating dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🌐 Starting server..."
echo "================================================"
echo ""
echo "📍 Chat Interface: http://localhost:8000/chat"
echo "📍 API Docs: http://localhost:8000/docs"
echo "📍 WebSocket: ws://localhost:8000/ws/chat/{connection_id}"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Start the application
python -m app.main