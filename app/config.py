"""Configuration management for the Research Engine"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
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
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4000, env="LLM_MAX_TOKENS")
    
    # Research Settings
    max_sources_per_query: int = Field(20, env="MAX_SOURCES_PER_QUERY")
    credibility_threshold: float = Field(0.6, env="CREDIBILITY_THRESHOLD")
    recency_months: int = Field(12, env="RECENCY_MONTHS")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()