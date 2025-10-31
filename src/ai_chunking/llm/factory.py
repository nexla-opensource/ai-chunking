"""Factory for creating LLM clients."""

from enum import Enum
from typing import Optional, Any, Union

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
        provider: Union[str, LLMProvider],
        api_key: Optional[str] = None,
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
        if not provider:
            raise ValueError("Provider must be specified")

        # Normalize provider to enum
        try:
            provider_enum = provider if isinstance(provider, LLMProvider) else LLMProvider(str(provider).lower())
        except ValueError:
            supported = ", ".join(sorted(p.value for p in LLMProvider))
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {supported}"
            )
            
        client_class = cls._provider_to_class[provider_enum]

        # Build init kwargs while avoiding passing None for model
        init_kwargs: dict[str, Any] = {}
        if config is not None:
            init_kwargs["model"] = config.model
            init_kwargs["temperature"] = config.temperature
            init_kwargs["max_retries"] = config.max_retries

        # Provider-specific handling
        if provider_enum == LLMProvider.VERTEX:
            project_id = kwargs.pop("project_id", None)
            if not project_id:
                raise ValueError("Vertex provider requires 'project_id' to be provided via kwargs")
            structured_client = client_class(
                project_id=project_id,
                **init_kwargs,
                **kwargs,
            )
        else:
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string for this provider")
            structured_client = client_class(
                api_key=api_key,
                **init_kwargs,
                **kwargs,
            )
        
        return StructuredLLMAdapter(structured_client)
