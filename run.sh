#!/bin/bash

# Research & Brainstorming Engine Startup Script

echo "=========================================="
echo "Research & Brainstorming Engine"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your API keys before running."
    exit 1
fi

# Check for Groq API key
if ! grep -q "GROQ_API_KEY=gsk_" .env; then
    echo "WARNING: Groq API key not configured in .env"
    echo "Please add your Groq API key to continue."
    echo "Get your key from: https://console.groq.com/keys"
    exit 1
fi

# Start the application
echo "Starting Research Engine API..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000