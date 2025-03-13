"""LiteLLM structured LLM client implementation."""

from typing import TypeVar, Type, Any
import logging
import json

import litellm
from pydantic import BaseModel
import tiktoken
import instructor

from ..base import StructuredLLMClient
from ..models import LiteLLMModels

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class LiteLLMStructuredClient(StructuredLLMClient):
    """LiteLLM structured LLM client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str | LiteLLMModels = LiteLLMModels.GPT_4O,
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """Initialize LiteLLM structured client.
        
        Args:
            api_key: API key for the model provider
            model: Model to use (can be string or LiteLLMModels enum)
            temperature: Sampling temperature
            max_retries: Maximum number of retries
        """
        litellm.api_key = api_key
        # Create a completion function that matches OpenAI's interface
        async def completion_fn(*args, **kwargs):
            return await litellm.acompletion(*args, **kwargs)
        # Patch the completion function with instructor
        self.completion_fn = instructor.patch(completion_fn, mode=instructor.Mode.MD_JSON)
        self.model = model.value if isinstance(model, LiteLLMModels) else model
        self.temperature = temperature
        self.max_retries = max_retries
        # Use OpenAI tokenizer for OpenAI-compatible models
        self._tokenizer = tiktoken.encoding_for_model("gpt-4")
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate structured response using LiteLLM.
        
        Args:
            prompt: Input prompt
            response_model: Pydantic model class for response
            **kwargs: Additional arguments passed to completion
            
        Returns:
            Structured response as Pydantic model instance
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = await self.completion_fn(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                response_model=response_model
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            raise
            
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate raw text response using LiteLLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments passed to completion
            
        Returns:
            Raw completion text
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature)
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self._tokenizer.encode(text)) 