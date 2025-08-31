#!/bin/bash

# Start Frontend Development Server Script

echo "🎨 Starting Frontend Development Server..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the frontend dev server
echo "✅ Starting Vite dev server on http://localhost:8080"
echo ""
npm run dev