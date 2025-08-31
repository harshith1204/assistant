# Groq Cloud Model Options

## Available Models for Research Engine

Groq provides several high-performance models optimized for speed. Here are the recommended options for your Research & Brainstorming Engine:

### Recommended Models

1. **mixtral-8x7b-32768** (Default)
   - Context: 32,768 tokens
   - Best for: Complex reasoning, research synthesis
   - Speed: Very fast
   - Quality: Excellent for research tasks

2. **llama2-70b-4096**
   - Context: 4,096 tokens  
   - Best for: General purpose, quick responses
   - Speed: Ultra fast
   - Quality: Good for structured data extraction

3. **gemma-7b-it**
   - Context: 8,192 tokens
   - Best for: Instruction following, summaries
   - Speed: Extremely fast
   - Quality: Good for simple tasks

### Configuration

Set your preferred model in `.env`:
```env
LLM_MODEL=mixtral-8x7b-32768
```

### Performance Tips

1. **For Research Tasks**: Use `mixtral-8x7b-32768` for best quality
2. **For Speed**: Use `gemma-7b-it` for simpler operations
3. **Token Limits**: Groq models have varying context windows, plan accordingly

### API Key Setup

1. Sign up at https://console.groq.com
2. Generate an API key from the dashboard
3. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`

### Rate Limits

Groq free tier includes:
- 30 requests per minute
- 14,400 requests per day

For production use, consider upgrading to a paid plan.

### Why Groq?

- **Speed**: 10-100x faster than traditional LLM APIs
- **Quality**: State-of-the-art open models
- **Cost**: Competitive pricing
- **Reliability**: Hardware-accelerated inference