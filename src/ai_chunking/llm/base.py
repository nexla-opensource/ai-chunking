"""Base class for structured LLM clients using Instructor."""

from abc import ABC, abstractmethod
from typing import TypeVar, Type, Any
from pydantic import BaseModel

from .cache import llm_cache

class LLMConfig(BaseModel):
    """Configuration for LLM clients."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 50_000
    max_retries: int = 3

class LLMError(Exception):
    """Base exception class for LLM-related errors."""
    pass

T = TypeVar('T', bound=BaseModel)

class StructuredLLMClient(ABC):
    """Base class for structured LLM clients that use Instructor for structured outputs."""

    @llm_cache.cache_llm_response()
    @abstractmethod
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate a structured response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            response_model: The Pydantic model class to structure the response
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            An instance of the response_model containing the structured output
        """
        pass
    
    @llm_cache.cache_llm_response()
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate a raw text response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The raw completion text from the LLM
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass 