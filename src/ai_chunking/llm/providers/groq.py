"""Groq structured LLM client implementation."""

from typing import TypeVar, Type, Any
import logging
import json

from groq import AsyncGroq
from pydantic import BaseModel
import tiktoken
import instructor

from ..base import StructuredLLMClient

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class GroqStructuredClient(StructuredLLMClient):
    """Groq structured LLM client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """Initialize Groq structured client.
        
        Args:
            api_key: Groq API key
            model: Model to use
            temperature: Sampling temperature
            max_retries: Maximum number of retries
        """
        base_client = AsyncGroq(api_key=api_key)
        # Patch the client with instructor in MD_JSON mode since Groq doesn't support function calling
        self.client = instructor.from_openai(base_client, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        # Using OpenAI tokenizer as approximation since Groq uses similar models
        self._tokenizer = tiktoken.encoding_for_model("gpt-4")
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate structured response using Groq.
        
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
            response = await self.client.chat.completions.create(
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
        """Generate raw text response using Groq.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments passed to completion
            
        Returns:
            Raw completion text
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = await self.client.chat.completions.create(
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