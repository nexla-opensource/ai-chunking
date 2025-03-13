"""Cohere structured LLM client implementation."""

from typing import TypeVar, Type, Any, Optional
import logging
import json

from cohere.client import AsyncClient as CohereClient
from pydantic import BaseModel

from ..base import StructuredLLMClient
from ..models import CohereModels

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class CohereStructuredClient(StructuredLLMClient):
    """Cohere structured LLM client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str | CohereModels = CohereModels.COMMAND,
        temperature: float = 0.0,
        max_retries: int = 3,
        client: Optional[CohereClient] = None,
        **kwargs: Any
    ):
        """Initialize Cohere structured client.
        
        Args:
            api_key: Cohere API key
            model: Model to use (can be string or CohereModels enum)
            temperature: Sampling temperature
            max_retries: Maximum number of retries
            client: Optional pre-configured Cohere client
            **kwargs: Additional client options
        """
        self.client = client or CohereClient(api_key=api_key, **kwargs)
        self.model = model.value if isinstance(model, CohereModels) else model
        self.temperature = temperature
        self.max_retries = max_retries
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate structured response using Cohere.
        
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
            # Create a detailed JSON schema prompt
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            
            structured_prompt = f"""You are a structured data extraction system. Your task is to generate a valid JSON object that matches the following schema:

{schema_str}

The response MUST:
1. Be valid JSON
2. Match the schema exactly
3. Include all required fields
4. Use correct data types for each field

User request: {prompt}

Respond ONLY with the JSON object, no other text. Format in JSON with proper indentation."""
            
            response = await self.client.generate(
                prompt=structured_prompt,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature)
            )
            
            # Extract JSON from response
            try:
                json_str = response.generations[0].text
                # Clean up any markdown formatting if present
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].strip()
                    
                return response_model.model_validate_json(json_str)
                
            except Exception as e:
                logger.error(f"Error parsing Cohere response as JSON: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            raise
            
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate raw text response using Cohere.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments passed to completion
            
        Returns:
            Raw completion text
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = await self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature)
            )
            return response.generations[0].text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens (approximate)
        """
        # Round to nearest integer since ProcessedChunk expects an integer
        return round(len(text.split()) * 1.3) 