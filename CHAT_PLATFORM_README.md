# AI Research & Chat Platform with Memory Management

## Overview

This is a sophisticated AI-powered chat platform that combines conversational AI with deep research capabilities and intelligent memory management using Mem0. The platform maintains both short-term and long-term context to provide personalized, context-aware responses.

## Key Features

### 🧠 Intelligent Memory Management
- **Short-term Memory**: Maintains conversation context within the current session
- **Long-term Memory**: Persists important information across sessions using Mem0
- **Entity & Topic Extraction**: Automatically identifies and remembers key entities and topics
- **Context-Aware Responses**: Uses historical context to provide more relevant answers

### 💬 Advanced Chat Capabilities
- **Real-time WebSocket Communication**: Instant, bidirectional communication
- **Streaming Responses**: Watch AI responses generate in real-time
- **Conversation Management**: Create, save, and resume conversations
- **Multi-turn Dialogue**: Maintains context across multiple exchanges

### 🔍 Integrated Research Engine
- **Deep Web Research**: Performs comprehensive web searches when needed
- **Source Citations**: Provides credible sources for all research findings
- **RICE-Scored Ideas**: Generates actionable ideas with prioritization scores
- **Research Brief Generation**: Creates detailed research documents

### 🎨 Modern User Interface
- **Beautiful Chat UI**: Clean, responsive design with gradient themes
- **Conversation Sidebar**: Easy navigation between multiple conversations
- **Real-time Status Updates**: Connection status, typing indicators, and progress updates
- **Suggestion Pills**: Quick-access follow-up questions
- **Research Mode Toggle**: Switch between chat and research modes

## Architecture

### Backend Components

1. **Chat Engine** (`app/core/chat_engine.py`)
   - Manages conversation flow
   - Integrates with research engine
   - Handles message processing

2. **Memory Manager** (`app/core/memory_manager.py`)
   - Mem0 integration for persistent memory
   - ChromaDB vector store for embeddings
   - Context retrieval and management

3. **WebSocket Handler** (`app/websocket_handler.py`)
   - Real-time bidirectional communication
   - Connection management
   - Message routing

4. **Research Engine** (`app/core/research_engine.py`)
   - Web scraping and search
   - Content synthesis
   - Idea generation

### Frontend Components

- **Single-Page Application** (`static/chat.html`)
- WebSocket client for real-time updates
- Responsive design with mobile support
- Rich text formatting for research results

## Memory Management System

### How It Works

1. **Message Processing**
   - Every message is processed and stored in memory
   - Key information is extracted and indexed
   - Context is built from relevant memories

2. **Context Building**
   - Recent messages (short-term)
   - Relevant memories (long-term)
   - Identified entities and topics
   - Previous research results

3. **Memory Types**
   - **Conversational Memory**: Chat history and context
   - **Research Memory**: Previous research briefs and findings
   - **Entity Memory**: People, places, organizations mentioned
   - **Preference Memory**: User preferences and patterns

## API Endpoints

### Chat Endpoints
- `POST /chat/message` - Send a chat message
- `GET /chat/conversations` - List all conversations
- `GET /chat/conversation/{id}` - Get specific conversation
- `DELETE /chat/conversation/{id}` - Delete conversation
- `POST /chat/memory/update` - Update memory
- `GET /chat/memory/stats` - Get memory statistics

### WebSocket Endpoint
- `WS /ws/chat/{connection_id}` - Real-time chat connection

### Research Endpoints
- `POST /research/run` - Run research
- `GET /research/brief/{id}` - Get research brief
- `POST /research/save` - Save to CRM/PMS

## Installation & Setup

### Prerequisites
- Python 3.8+
- GROQ API Key

### Quick Start

1. **Clone and setup environment**
```bash
cd /workspace
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your-secret-key-change-in-production
```

4. **Run the application**
```bash
./run_chat.sh
# Or directly:
python -m app.main
```

5. **Access the platform**
- Chat Interface: http://localhost:8000/chat
- API Documentation: http://localhost:8000/docs

## Usage Guide

### Starting a Conversation

1. Open the chat interface at http://localhost:8000/chat
2. Type your message in the input field
3. Press Enter or click Send

### Using Research Mode

1. Click the "Research Mode" button in the header
2. Ask research questions like:
   - "Research the latest trends in AI"
   - "Analyze competitor pricing strategies"
   - "Find market opportunities in sustainable technology"

### Managing Conversations

- **New Chat**: Click "+ New Chat" in the sidebar
- **Switch Conversations**: Click on any conversation in the sidebar
- **Clear Chat**: Use the Clear button in the header
- **Delete Conversation**: Via API endpoint

### Customizing Behavior

Toggle these options in the input area:
- **Stream**: Enable/disable streaming responses
- **Memory**: Enable/disable long-term memory
- **Web Search**: Enable/disable web search capabilities

## WebSocket Message Types

### Client to Server
```javascript
// Chat message
{
  type: "chat",
  data: {
    message: "Hello",
    conversation_id: "abc123",
    stream: true,
    use_long_term_memory: true,
    use_web_search: true
  }
}

// Research request
{
  type: "research",
  data: {
    query: "Market analysis for...",
    conversation_id: "abc123"
  }
}
```

### Server to Client
```javascript
// Regular message
{
  type: "message",
  data: {
    conversation_id: "abc123",
    message: { /* ChatMessage object */ },
    suggestions: ["Follow-up 1", "Follow-up 2"]
  }
}

// Stream chunk
{
  type: "stream_chunk",
  data: {
    conversation_id: "abc123",
    content: "Partial response...",
    is_final: false
  }
}
```

## Configuration

### Environment Variables

```env
# Required
GROQ_API_KEY=your_api_key

# Optional
SECRET_KEY=your-secret-key
LLM_MODEL=mixtral-8x7b-32768
LLM_TEMPERATURE=0.7
MEMORY_TTL_HOURS=24
MAX_CONVERSATION_HISTORY=100
WEBSOCKET_PING_INTERVAL=30
```

### Memory Settings

Adjust in `app/config.py`:
- `memory_ttl_hours`: Short-term memory retention
- `max_memory_items`: Maximum memories per user
- `default_context_window`: Messages to include in context

## Advanced Features

### Custom Memory Operations

```python
# Update memory via API
POST /chat/memory/update
{
  "conversation_id": "abc123",
  "memory_type": "long_term",
  "key": "user_preference",
  "value": "Prefers technical explanations",
  "operation": "add"
}
```

### Research Integration

The chat automatically triggers research when:
- Questions contain research indicators
- Current information is needed
- Complex analysis is required

### Context Persistence

- Conversations are maintained across sessions
- User preferences are remembered
- Research history influences future responses

## Performance Optimization

### Caching
- Short-term memory cached in-memory
- ChromaDB for vector similarity search
- Research results cached for reuse

### Scalability
- Async/await for non-blocking operations
- WebSocket for efficient real-time communication
- Connection pooling for database operations

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if the server is running
   - Verify firewall settings
   - Check browser console for errors

2. **Memory Not Persisting**
   - Ensure ChromaDB directory has write permissions
   - Check Mem0 configuration
   - Verify GROQ API key is valid

3. **Research Not Working**
   - Verify web search is enabled
   - Check internet connectivity
   - Review research engine logs

### Debug Mode

Enable debug logging:
```python
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Secret Key**: Change the default secret key in production
3. **CORS**: Configure appropriate CORS origins for production
4. **WebSocket**: Implement authentication for production use
5. **Memory**: Consider data privacy when storing user information

## Future Enhancements

- [ ] User authentication and multi-tenancy
- [ ] Voice input/output support
- [ ] File upload and analysis
- [ ] Integration with more LLM providers
- [ ] Advanced analytics dashboard
- [ ] Export conversation history
- [ ] Collaborative chat rooms
- [ ] Plugin system for extensions

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is provided as-is for educational and development purposes.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check application logs for detailed error messages

---

**Built with ❤️ using FastAPI, Groq, Mem0, and modern web technologies**