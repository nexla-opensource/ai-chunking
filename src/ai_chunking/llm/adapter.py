"""Adapter to make structured LLM clients compatible with old interface."""

from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from .base import StructuredLLMClient

T = TypeVar('T', bound=BaseModel)

class StructuredLLMAdapter(StructuredLLMClient):
    """Adapts structured LLM clients to the old LLMClient interface."""
    
    def __init__(self, structured_client: StructuredLLMClient):
        """Initialize adapter.
        
        Args:
            structured_client: The structured LLM client to adapt
        """
        self.client = structured_client
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate a structured response using the wrapped client.
        
        Args:
            prompt: The prompt to send to the LLM
            response_model: The Pydantic model class to structure the response
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            An instance of the response_model containing the structured output
        """
        return await self.client.structured_generate(prompt, response_model, **kwargs)
    
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate a raw text response using the wrapped client.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The raw completion text from the LLM
        """
        return await self.client.generate(prompt, **kwargs)
    
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the wrapped client.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        return await self.client.count_tokens(text)
        
    async def complete(
        self,
        prompt: str,
        response_format: Optional[Union[Type[T], Dict[str, Any]]] = None
    ) -> Union[str, T, Dict[str, Any]]:
        """Complete a prompt using the structured LLM client.
        
        This method adapts the structured_generate method to match the old interface.
        If response_format is provided, it must be a Pydantic model class.
        If no response_format is provided, returns the raw completion text.
        
        Args:
            prompt: The prompt to complete
            response_format: Optional Pydantic model class for structured output
            
        Returns:
            Union[str, T, Dict[str, Any]]: The completion text or structured output
            
        Raises:
            ValueError: If response_format is a dict (JSON schema not supported)
        """
        if response_format is None:
            # For unstructured completion, use raw text response
            return await self.generate(prompt=prompt)
            
        if isinstance(response_format, dict):
            raise ValueError(
                "JSON schema response format not supported. "
                "Please use a Pydantic model class instead."
            )
            
        if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
            raise ValueError("response_format must be a Pydantic model class")
            
        return await self.structured_generate(
            prompt=prompt,
            response_model=response_format
        ) 