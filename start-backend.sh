#!/bin/bash

# Start Backend Server Script

echo "🚀 Starting Backend Server..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please update it with your API keys."
    echo ""
    echo "Required:"
    echo "  - GROQ_API_KEY: Get from https://console.groq.com"
    echo ""
    echo "After updating .env, run this script again."
    exit 1
fi

# Check if GROQ_API_KEY is set
if ! grep -q "GROQ_API_KEY=gsk_" .env; then
    echo "⚠️  GROQ_API_KEY not set in .env file!"
    echo "Please update your .env file with a valid Groq API key."
    echo "Get your API key from: https://console.groq.com"
    exit 1
fi

# Install Python dependencies if needed
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start the backend server
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""
cd app && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000