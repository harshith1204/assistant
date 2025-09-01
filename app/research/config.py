"""Business-specific configuration for Open Deep Research integration"""

from typing import Dict, Any
from .configuration import Configuration, SearchAPI
from app.config import settings


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

            # Enable clarification for complex business queries
            allow_clarification=True,

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
                "allow_clarification": config.allow_clarification,
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


