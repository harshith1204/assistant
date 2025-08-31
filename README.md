# AI Business Assistant - Full Stack Application

A powerful AI-powered research and brainstorming engine with real-time chat interface, featuring WebSocket communication, memory management, and integration with CRM/PMS systems.

## 🚀 Features

- **Real-time Chat Interface**: WebSocket-based communication for instant responses
- **AI-Powered Research**: Comprehensive web research with source citations
- **Memory Management**: Short-term and long-term memory using Mem0
- **Streaming Responses**: Real-time streaming of AI responses
- **Multi-phase Processing**: Research → Analysis → Thinking phases
- **CRM/PMS Integration**: Save research briefs to external systems
- **Beautiful UI**: Modern React interface with Tailwind CSS

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Groq API Key (required)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd assistant
```

### 2. Set up environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your API key from: https://console.groq.com
```

Required environment variable:
- `GROQ_API_KEY`: Your Groq API key for LLM access

### 3. Install dependencies

#### Backend (Python)
```bash
pip install -r requirements.txt
```

#### Frontend (Node.js)
```bash
cd frontend
npm install
cd ..
```

## 🚀 Quick Start

### Option 1: Start everything with one command

```bash
./start-all.sh
```

This will:
- Install all dependencies
- Start the backend server on http://localhost:8000
- Start the frontend dev server on http://localhost:8080
- Open API documentation at http://localhost:8000/docs

### Option 2: Start servers separately

#### Terminal 1 - Backend:
```bash
./start-backend.sh
```

#### Terminal 2 - Frontend:
```bash
./start-frontend.sh
```

## 📁 Project Structure

```
/workspace/
├── app/                      # Backend FastAPI application
│   ├── main.py              # Main FastAPI app with endpoints
│   ├── websocket_handler.py # WebSocket connection manager
│   ├── config.py            # Configuration management
│   ├── chat_models.py       # Chat data models
│   ├── models.py            # Research data models
│   └── core/                # Core business logic
│       ├── chat_engine.py   # Chat processing engine
│       ├── memory_manager.py # Memory management with Mem0
│       ├── research_engine.py # Research orchestration
│       ├── web_research.py  # Web search functionality
│       └── synthesis.py     # Content synthesis
├── frontend/                 # Frontend React application
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ChatInterfaceReal.tsx # Main chat interface
│   │   │   └── chat/        # Chat-related components
│   │   ├── services/        # API services
│   │   │   └── api.ts       # HTTP and WebSocket clients
│   │   └── pages/           # Page components
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Vite configuration
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── start-*.sh              # Startup scripts
```

## 🔌 API Endpoints

### REST API

- `POST /chat/message` - Send a chat message
- `GET /chat/conversations` - List all conversations
- `GET /chat/conversation/{id}` - Get conversation details
- `DELETE /chat/conversation/{id}` - Delete a conversation
- `POST /research/run` - Run research on a query
- `GET /research/brief/{id}` - Get research brief
- `GET /health` - Health check endpoint

### WebSocket

- `ws://localhost:8000/ws/chat/{connection_id}` - WebSocket endpoint

WebSocket message types:
- `chat` - Send chat message
- `research` - Request research
- `list_conversations` - Get conversation list
- `get_history` - Get conversation history
- `ping` - Keep connection alive

## 🔧 Configuration

### Backend Configuration (.env)

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:8080"]
LLM_MODEL=moonshotai/kimi-k2-instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
```

### Frontend Configuration

Development (`.env.development`):
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Production (`.env.production`):
```env
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com
```

## 🧪 Testing the Integration

1. Start both servers using `./start-all.sh`
2. Open http://localhost:8080 in your browser
3. The connection status should show "connected" (green dot)
4. Try sending a message like "What are the latest trends in AI?"
5. Enable research mode for comprehensive analysis
6. Check the browser console for WebSocket messages

## 🐛 Troubleshooting

### Backend Issues

1. **GROQ_API_KEY not found**
   - Make sure you've created a `.env` file
   - Add your Groq API key: `GROQ_API_KEY=gsk_...`

2. **Port 8000 already in use**
   - Kill the existing process: `lsof -ti:8000 | xargs kill -9`
   - Or change the port in `start-backend.sh`

### Frontend Issues

1. **WebSocket connection failed**
   - Check that the backend is running
   - Verify CORS settings in backend `.env`
   - Check browser console for errors

2. **Port 8080 already in use**
   - Kill the existing process: `lsof -ti:8080 | xargs kill -9`
   - Or change the port in `vite.config.ts`

### Connection Issues

1. **Check backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check WebSocket connection**:
   - Open browser DevTools → Network → WS tab
   - Look for connection to `/ws/chat/`

## 🚢 Deployment

### Backend Deployment

1. Update production environment variables
2. Use a production ASGI server:
   ```bash
   pip install gunicorn
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Frontend Deployment

1. Build for production:
   ```bash
   cd frontend
   npm run build
   ```

2. Serve the `dist` folder with any static file server

### Docker Deployment (Optional)

Create a `Dockerfile` for containerized deployment:

```dockerfile
# Backend
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Groq Console](https://console.groq.com/)

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.