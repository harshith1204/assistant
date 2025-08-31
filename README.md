# Research & Brainstorming Engine

A production-ready Python backend for AI-powered research and brainstorming with CRM/PMS integration. This system turns fuzzy prompts into structured research briefs with citations, actionable ideas, and execution plans.

## 🚀 Features

### Core Capabilities
- **Query Understanding**: LLM-powered parsing and expansion of research queries
- **Web Research Pipeline**: Multi-source search, content extraction, and credibility scoring
- **Synthesis Engine**: Findings generation with citations and evidence
- **Idea Generation**: RICE-scored actionable recommendations
- **Plan Creation**: Convert ideas into structured execution plans
- **CRM/PMS Integration**: Save research to your existing systems

### Research Pipeline
1. **Parse & Understand**: Extract entities, geography, industry, timeframe
2. **Expand Queries**: Generate focused sub-queries for comprehensive coverage
3. **Search & Fetch**: Gather sources from web with credibility scoring
4. **Extract Evidence**: Pull facts, statistics, and quotes with citations
5. **Synthesize Findings**: Create findings with confidence scores
6. **Generate Ideas**: Produce RICE-scored actionable recommendations
7. **Create Plans**: Transform ideas into execution roadmaps

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key (for LLM operations)
- Optional: CRM/PMS API credentials
- Optional: Redis for caching (production)
- Optional: PostgreSQL for persistence (production)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd research-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Run the application**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# CRM Integration (optional)
CRM_BASE_URL=https://your-crm-api.com
CRM_AUTH_TOKEN=your_crm_token

# PMS Integration (optional)
PMS_BASE_URL=https://your-pms-api.com
PMS_AUTH_TOKEN=your_pms_token

# Research Settings
MAX_SOURCES_PER_QUERY=20
CREDIBILITY_THRESHOLD=0.6
RECENCY_MONTHS=12
LLM_MODEL=gpt-4-turbo-preview
```

## 📡 API Endpoints

### Research Operations

#### Run Research
```http
POST /research/run
Content-Type: application/json

{
  "query": "How to grow B2B edtech in India?",
  "scope": ["market", "competitors", "pricing"],
  "geo": "India",
  "industry": "EdTech",
  "max_sources": 20
}
```

#### Get Research Brief
```http
GET /research/brief/{brief_id}
```

#### Save to CRM/PMS
```http
POST /research/save
Content-Type: application/json

{
  "brief_id": "uuid",
  "crm_ref": {
    "lead_id": "uuid",
    "business_id": "uuid"
  },
  "pms_ref": {
    "project_id": "uuid"
  },
  "create_tasks": true
}
```

#### Convert Ideas to Plan
```http
POST /research/ideas-to-plan
Content-Type: application/json

{
  "brief_id": "uuid",
  "selected_ideas": ["idea1", "idea2"],
  "timeline_weeks": 12
}
```

## 📊 Data Models

### Research Brief Structure
```json
{
  "brief_id": "uuid",
  "query": "Original research query",
  "date": "2025-01-01T00:00:00",
  "entities": ["B2B", "edtech", "India"],
  "key_questions": [
    "What is the market size?",
    "Who are the competitors?"
  ],
  "findings": [
    {
      "title": "Market Growth Opportunity",
      "summary": "The B2B edtech market in India...",
      "evidence": [
        {
          "quote": "Market expected to grow 25% YoY",
          "url": "https://source.com/article",
          "credibility_score": 0.85
        }
      ],
      "confidence": 0.75,
      "key_insights": ["insight1", "insight2"]
    }
  ],
  "ideas": [
    {
      "idea": "Partner with state education boards",
      "rationale": "Based on market analysis...",
      "rice": {
        "reach": 5000,
        "impact": 3,
        "confidence": 0.7,
        "effort": 30,
        "score": 350
      },
      "prerequisites": ["Legal compliance", "Local team"],
      "risks": ["Regulatory changes"]
    }
  ],
  "executive_summary": "Executive summary text..."
}
```

## 🧪 Testing

Run the example usage script:
```bash
python example_usage.py
```

Run tests (if available):
```bash
pytest tests/
```

## 🏗️ Architecture

```
app/
├── core/
│   ├── query_understanding.py  # LLM query parsing
│   ├── web_research.py        # Web scraping pipeline
│   ├── synthesis.py           # Finding synthesis
│   └── research_engine.py     # Main orchestrator
├── integrations/
│   ├── crm_client.py          # CRM API client
│   └── pms_client.py          # PMS API client
├── models.py                   # Pydantic models
├── config.py                   # Configuration
└── main.py                     # FastAPI application
```

## 🔒 Security & Compliance

- **PII Protection**: Automatic sanitization of personal information
- **Citation Requirements**: All claims backed by sources
- **Quote Limits**: Short excerpts only, with source links
- **Domain Controls**: Credibility scoring and source filtering
- **Rate Limiting**: Configurable request limits
- **robots.txt Compliance**: Respects website crawling rules

## 🚀 Production Deployment

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: research_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

## 📈 Monitoring

The application includes:
- Structured logging with `structlog`
- Health check endpoint at `/health`
- Research status tracking
- Error handling and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Your License Here]

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the example usage script
- Review the API documentation at `/docs` (FastAPI automatic docs)

## 🔮 Future Enhancements

- [ ] Real-time research updates via WebSocket
- [ ] Advanced hallucination detection
- [ ] Multi-language support
- [ ] Custom domain knowledge bases
- [ ] Research collaboration features
- [ ] Advanced caching strategies
- [ ] ML-based source ranking
- [ ] Research templates library