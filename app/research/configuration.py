"""Unified configuration management for Groq + DDGS Research system."""

import os
from enum import Enum
from typing import Any, List, Optional, Dict

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import core models and settings
from app.config import settings


class SearchAPI(Enum):
    """Enumeration of available search API providers - Groq optimized."""

    DDGS = "ddgs"  # DuckDuckGo Search (primary)
    NONE = "none"  # No search (for testing)


###################
# Business Research Types and Enums
###################
class BusinessResearchType(str, Enum):
    """Types of business research supported"""
    MARKET_ANALYSIS = "market_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    INDUSTRY_REPORT = "industry_report"
    FEASIBILITY_STUDY = "feasibility_study"
    STRATEGIC_PLANNING = "strategic_planning"
    PRODUCT_DEVELOPMENT = "product_development"
    MARKET_ENTRY = "market_entry"
    INVESTMENT_ANALYSIS = "investment_analysis"





class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )

    max_concurrent_research_units: int = Field(
        default=5,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.DDGS,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": "ddgs",
                "description": "Search API to use for research - optimized for Groq + DDGS stack.",
                "options": [
                    {"label": "DuckDuckGo Search", "value": SearchAPI.DDGS.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="groq:llama3-8b-8192",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "groq:llama3-8b-8192",
                "description": "Model for summarizing research results from DDGS search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="groq:llama3-8b-8192",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "groq:llama3-8b-8192",
                "description": "Model for conducting research with DDGS search integration."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="groq:llama3-8b-8192",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "groq:llama3-8b-8192",
                "description": "Model for compressing research findings from sub-agents."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="groq:llama3-8b-8192",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "groq:llama3-8b-8192",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )



    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


###################
# Business Configuration Factory
###################

class BusinessDeepResearchConfig:
    """Configuration factory for business-focused deep research"""

    @staticmethod
    def create_business_config(
        industry_focus: str = None,
        geography_focus: str = None,
        research_depth: str = "standard"  # "light", "standard", "deep"
    ) -> Configuration:
        """Create configuration optimized for business research"""

        # Base configuration using existing Groq setup
        config = Configuration(
            # Use Groq models (matching existing setup)
            research_model=f"groq:{settings.llm_model}",
            summarization_model=f"groq:{settings.llm_model}",
            compression_model=f"groq:{settings.llm_model}",
            final_report_model=f"groq:{settings.llm_model}",

            # Use custom DDGS integration (not built-in search APIs)
            search_api=SearchAPI.NONE,

            # Business-optimized concurrency and depth settings
            max_concurrent_research_units=BusinessDeepResearchConfig._get_concurrent_units(research_depth),
            max_researcher_iterations=BusinessDeepResearchConfig._get_iterations(research_depth),
            max_react_tool_calls=BusinessDeepResearchConfig._get_tool_calls(research_depth),

            # Content processing optimized for business content
            max_content_length=35000,  # Reasonable for business reports/articles
            max_structured_output_retries=3,

            # Model-specific token limits (conservative for Groq)
            research_model_max_tokens=4000,
            summarization_model_max_tokens=3000,
            compression_model_max_tokens=4000,
            final_report_model_max_tokens=5000,
        )

        return config

    @staticmethod
    def _get_concurrent_units(depth: str) -> int:
        """Get optimal concurrent research units - optimized for Groq performance"""
        depth_map = {
            "light": 1,      # Minimize for speed
            "standard": 2,   # Balanced for Groq rate limits
            "deep": 3        # Maximum for comprehensive research
        }
        return depth_map.get(depth, 2)

    @staticmethod
    def _get_iterations(depth: str) -> int:
        """Get optimal research iterations - balanced for quality and speed"""
        depth_map = {
            "light": 2,      # Quick results
            "standard": 4,   # Balanced depth
            "deep": 6        # Comprehensive analysis
        }
        return depth_map.get(depth, 4)

    @staticmethod
    def _get_tool_calls(depth: str) -> int:
        """Get optimal tool calls per researcher - optimized for DDGS efficiency"""
        depth_map = {
            "light": 4,      # Minimal for speed
            "standard": 8,   # Efficient for quality
            "deep": 12       # Comprehensive coverage
        }
        return depth_map.get(depth, 8)

    @staticmethod
    def get_business_research_config(
        query_type: str = "general",
        industry: str = None,
        geography: str = None
    ) -> Dict[str, Any]:
        """Get complete configuration dictionary for business research"""

        # Determine research depth based on query type
        depth_configs = {
            "market_analysis": "deep",
            "competitor_analysis": "standard",
            "industry_report": "deep",
            "feasibility_study": "deep",
            "quick_research": "light",
            "general": "standard"
        }

        research_depth = depth_configs.get(query_type, "standard")

        # Create base configuration
        config = BusinessDeepResearchConfig.create_business_config(
            industry_focus=industry,
            geography_focus=geography,
            research_depth=research_depth
        )

        # Convert to dictionary for LangGraph
        config_dict = {
            "configurable": {
                "research_model": config.research_model,
                "summarization_model": config.summarization_model,
                "compression_model": config.compression_model,
                "final_report_model": config.final_report_model,
                "search_api": "none",  # Use custom tools
                "max_concurrent_research_units": config.max_concurrent_research_units,
                "max_researcher_iterations": config.max_researcher_iterations,
                "max_react_tool_calls": config.max_react_tool_calls,
                "max_content_length": config.max_content_length,
                "max_structured_output_retries": config.max_structured_output_retries,
                "research_model_max_tokens": config.research_model_max_tokens,
                "summarization_model_max_tokens": config.summarization_model_max_tokens,
                "compression_model_max_tokens": config.compression_model_max_tokens,
                "final_report_model_max_tokens": config.final_report_model_max_tokens,
            }
        }

        # Add business-specific metadata
        config_dict["configurable"].update({
            "query_type": query_type,
            "industry": industry,
            "geography": geography,
            "research_depth": research_depth,
            "business_focus": True
        })

        return config_dict