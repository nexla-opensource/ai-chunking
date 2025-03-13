"""OpenAI structured LLM client implementation."""

from typing import TypeVar, Type, Any, Optional
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel
import tiktoken
import instructor

from ..base import StructuredLLMClient
from ..models import OpenAIModels

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class OpenAIStructuredClient(StructuredLLMClient):
    """OpenAI structured LLM client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str | OpenAIModels = OpenAIModels.GPT_4O,
        temperature: float = 0.0,
        max_retries: int = 3,
        client: Optional[AsyncOpenAI] = None,
        **kwargs: Any
    ):
        """Initialize OpenAI structured client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (can be string or OpenAIModels enum)
            temperature: Sampling temperature
            max_retries: Maximum number of retries
            client: Optional pre-configured OpenAI client
            **kwargs: Additional client options
        """
        base_client = client or AsyncOpenAI(api_key=api_key, **kwargs)
        # Patch the client with instructor
        self.client = instructor.patch(base_client)
        self.model = model.value if isinstance(model, OpenAIModels) else model
        self.temperature = temperature
        self.max_retries = max_retries
        self._tokenizer = tiktoken.encoding_for_model(self.model)
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate structured response using OpenAI.
        
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
                max_retries=kwargs.get("max_retries", self.max_retries),
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
        """Generate raw text response using OpenAI.
        
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
                temperature=kwargs.get("temperature", self.temperature),
                max_retries=kwargs.get("max_retries", self.max_retries)
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