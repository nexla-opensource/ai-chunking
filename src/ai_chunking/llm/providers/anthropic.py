"""Anthropic structured LLM client implementation."""

from typing import TypeVar, Type, Any, Optional
import logging
import json

from anthropic import AsyncAnthropic
from pydantic import BaseModel
import tiktoken
import instructor

from ..base import StructuredLLMClient
from ..models import AnthropicModels
from ai_chunking.utils.json_utils import safe_loads

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class AnthropicStructuredClient(StructuredLLMClient):
    """Anthropic structured LLM client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str | AnthropicModels = AnthropicModels.CLAUDE_35_SONNET,
        temperature: float = 0.0,
        max_retries: int = 3,
        client: Optional[AsyncAnthropic] = None,
        **kwargs: Any
    ):
        """Initialize Anthropic structured client.
        
        Args:
            api_key: Anthropic API key
            model: Model to use (can be string or AnthropicModels enum)
            temperature: Sampling temperature
            max_retries: Maximum number of retries
            client: Optional pre-configured Anthropic client
            **kwargs: Additional client options
        """
        base_client = client or AsyncAnthropic(api_key=api_key, **kwargs)
        # Use raw Anthropic client since we'll handle structured output ourselves
        self.client = base_client
        self.model = model.value if isinstance(model, AnthropicModels) else model
        self.temperature = temperature
        self.max_retries = max_retries
        
    async def structured_generate(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """Generate structured response using Anthropic.
        
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
            # Create a JSON schema prompt
            schema = response_model.model_json_schema()
            structured_prompt = f"""
            Please provide a response in JSON format according to this schema:
            {json.dumps(schema, indent=2)}
            
            The response should be valid JSON that matches this schema.
            
            User request: {prompt}
            """
            
            # Get raw response from Claude
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 4096)  # Default max tokens for Claude
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            # Clean up any markdown formatting if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            else:
                # Try to find JSON object between curly braces
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    response_text = response_text[start:end + 1]
            
            response_text = response_text.strip()
            
            try:
                # First validate it's proper JSON
                parsed_json = safe_loads(response_text)
                if not parsed_json:
                    raise ValueError("Empty or invalid JSON response")
                # Then validate against the model
                return response_model.model_validate(parsed_json)
            except Exception as e:
                logger.error(f"Invalid JSON response: {response_text}")
                raise
            
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            raise
            
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate raw text response using Anthropic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments passed to completion
            
        Returns:
            Raw completion text
            
        Raises:
            Exception: If generation fails
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 4096)  # Default max tokens for Claude
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens (approximate for Claude)
        """
        # Anthropic doesn't provide a direct token counter, using approximate
        # Round to nearest integer since ProcessedChunk expects an integer
        return round(len(text.split()) * 1.3) 