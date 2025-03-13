"""Factory for creating LLM clients."""

from enum import Enum
from typing import Optional, Type, Any

from .adapter import StructuredLLMAdapter
from .base import StructuredLLMClient, LLMConfig
from .providers import (
    OpenAIStructuredClient,
    AnthropicStructuredClient,
    GeminiStructuredClient,
    VertexAIStructuredClient,
    GroqStructuredClient,
    LiteLLMStructuredClient,
    CohereStructuredClient
)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    VERTEX = "vertex"
    GROQ = "groq"
    LITELLM = "litellm"
    COHERE = "cohere"


class LLMFactory:
    """Factory for creating LLM clients."""
    
    _provider_to_class = {
        LLMProvider.OPENAI: OpenAIStructuredClient,
        LLMProvider.ANTHROPIC: AnthropicStructuredClient,
        LLMProvider.GEMINI: GeminiStructuredClient,
        LLMProvider.VERTEX: VertexAIStructuredClient,
        LLMProvider.GROQ: GroqStructuredClient,
        LLMProvider.LITELLM: LiteLLMStructuredClient,
        LLMProvider.COHERE: CohereStructuredClient,
    }

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str,
        config: Optional[LLMConfig] = None,
        **kwargs: Any
    ) -> StructuredLLMClient:
        """Create an LLM client instance.
        
        Args:
            provider: Provider identifier
            api_key: API key for the provider
            config: Optional client configuration
            **kwargs: Additional configuration options
            
        Returns:
            Configured LLM client instance
            
        Raises:
            ValueError: If provider is not supported or api_key is invalid
        """
        if not provider or not isinstance(provider, str):
            raise ValueError("Provider must be a non-empty string")
            
        if not api_key or not isinstance(api_key, str):
            raise ValueError("API key must be a non-empty string")
            
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            supported = ", ".join(sorted(p.value for p in LLMProvider))
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {supported}"
            )
            
        client_class = cls._provider_to_class[provider_enum]
        structured_client = client_class(
            api_key=api_key,
            model=config.model if config else None,
            temperature=config.temperature if config else 0.0,
            max_retries=config.max_retries if config else 3,
            **kwargs
        )
        
        return StructuredLLMAdapter(structured_client) 