"""Configuration management for the Research Engine"""

import secrets
import string
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    
    # CRM Configuration
    crm_base_url: str = Field("https://stage-api.simpo.ai/crm", env="CRM_BASE_URL")
    crm_api_key: Optional[str] = Field(None, env="CRM_API_KEY")
    crm_auth_token: Optional[str] = Field(None, env="CRM_AUTH_TOKEN")
    
    # PMS Configuration
    pms_base_url: str = Field("https://stage-api.simpo.ai/pms", env="PMS_BASE_URL")
    pms_api_key: Optional[str] = Field(None, env="PMS_API_KEY")
    pms_auth_token: Optional[str] = Field(None, env="PMS_AUTH_TOKEN")
    
    # Database
    database_url: str = Field("sqlite:///./research.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # Search APIs
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    google_cse_id: Optional[str] = Field(None, env="GOOGLE_CSE_ID")
    bing_api_key: Optional[str] = Field(None, env="BING_API_KEY")
    
    # Web Scraping
    proxy_url: Optional[str] = Field(None, env="PROXY_URL")
    user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="USER_AGENT"
    )
    
    # Application Settings
    app_name: str = Field("Research & Brainstorming Engine", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(3600, env="RATE_LIMIT_PERIOD")
    
    # LLM Settings
    llm_model: str = Field("llama3-8b-8192", env="LLM_MODEL")
    llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4000, env="LLM_MAX_TOKENS")
    
    # LLM Settings for different use cases
    chat_temperature: float = Field(0.2, env="CHAT_TEMPERATURE")
    chat_max_tokens: int = Field(2000, env="CHAT_MAX_TOKENS")
    synthesis_temperature: float = Field(0.2, env="SYNTHESIS_TEMPERATURE")
    synthesis_max_tokens: int = Field(500, env="SYNTHESIS_MAX_TOKENS")
    query_understanding_temperature: float = Field(0.2, env="QUERY_UNDERSTANDING_TEMPERATURE")
    summary_max_tokens: int = Field(200, env="SUMMARY_MAX_TOKENS")
    
    # Research Settings
    max_sources_per_query: int = Field(20, env="MAX_SOURCES_PER_QUERY")
    credibility_threshold: float = Field(0.6, env="CREDIBILITY_THRESHOLD")
    recency_months: int = Field(12, env="RECENCY_MONTHS")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    # WebSocket settings
    websocket_ping_interval: int = Field(30, env="WEBSOCKET_PING_INTERVAL")
    websocket_ping_timeout: int = Field(10, env="WEBSOCKET_PING_TIMEOUT")
    
    # Memory settings
    memory_ttl_hours: int = Field(24, env="MEMORY_TTL_HOURS")
    max_memory_items: int = Field(1000, env="MAX_MEMORY_ITEMS")
    memory_temperature: float = Field(0.1, env="MEMORY_TEMPERATURE")
    memory_max_tokens: int = Field(1000, env="MEMORY_MAX_TOKENS")
    memory_embedder_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="MEMORY_EMBEDDER_MODEL")
    memory_collection_name: str = Field("chat_memories", env="MEMORY_COLLECTION_NAME")
    memory_db_path: str = Field("./chroma_db", env="MEMORY_DB_PATH")
    memory_embedder_provider: str = Field("huggingface", env="MEMORY_EMBEDDER_PROVIDER")
    
    # Chat settings
    max_conversation_history: int = Field(100, env="MAX_CONVERSATION_HISTORY")
    default_context_window: int = Field(10, env="DEFAULT_CONTEXT_WINDOW")
    
    # Session settings
    secret_key: str = Field(default_factory=lambda: ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)), env="SECRET_KEY")
    session_expire_minutes: int = Field(1440, env="SESSION_EXPIRE_MINUTES")  # 24 hours
    
    # Groq model alias for compatibility
    @property
    def groq_model(self) -> str:
        return self.llm_model
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()