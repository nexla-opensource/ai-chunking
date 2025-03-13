"""Model definitions for structured LLM clients."""

from enum import Enum

class OpenAIModels(str, Enum):
    """OpenAI model options."""
    GPT_4O = "gpt-4o"  # Latest GPT-4 Omni model
    GPT_4O_MINI = "gpt-4o-mini"  # Fast, inexpensive GPT-4o variant
    GPT_4_TURBO = "gpt-4-turbo"  # Latest GPT-4 Turbo with Vision
    GPT_4 = "gpt-4"  # Base GPT-4
    GPT_35_TURBO = "gpt-35-turbo"  # Latest GPT-3.5 Turbo
    GPT_35_TURBO_16K = "gpt-35-turbo-16k"  # Extended context GPT-3.5

class AnthropicModels(str, Enum):
    """Anthropic model options."""
    CLAUDE_35_SONNET = "claude-3-5-sonnet-latest"  # Latest most intelligent model
    CLAUDE_35_HAIKU = "claude-3-5-haiku-latest"  # Latest fastest model
    CLAUDE_3_OPUS = "claude-3-opus-latest"  # Most capable Claude 3
    CLAUDE_3_SONNET = "claude-3-sonnet"  # Balanced Claude 3
    CLAUDE_3_HAIKU = "claude-3-haiku"  # Fast Claude 3
    CLAUDE_2 = "claude-2"  # Legacy Claude 2

class GeminiModels(str, Enum):
    """Google Gemini model options."""
    GEMINI_ULTRA = "gemini-ultra"  # Most capable
    GEMINI_PRO = "gemini-pro"  # Balanced
    GEMINI_PRO_VISION = "gemini-pro-vision"  # Vision model

class VertexAIModels(str, Enum):
    """Vertex AI model options."""
    GEMINI_ULTRA = "gemini-ultra"
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    PALM2 = "text-bison"  # Legacy PaLM 2
    PALM2_CHAT = "chat-bison"  # Legacy PaLM 2 Chat

class GroqModels(str, Enum):
    """Groq model options."""
    MIXTRAL = "mixtral-8x7b-32768"  # Mixtral model
    LLAMA2_70B = "llama2-70b-4096"  # LLaMA 2 70B

class CohereModels(str, Enum):
    """Cohere model options."""
    COMMAND = "command"  # Latest command model
    COMMAND_LIGHT = "command-light"  # Lighter version
    COMMAND_NIGHTLY = "command-nightly"  # Experimental version
    COMMAND_R = "command-r"  # Research version

class LiteLLMModels(str, Enum):
    """LiteLLM supported model options."""
    # OpenAI Models
    GPT_4O = "gpt-4o"  # Latest GPT-4 Omni model
    GPT_4O_MINI = "gpt-4o-mini"  # Fast, inexpensive GPT-4o variant
    GPT_4_TURBO = "gpt-4-turbo"  # Latest GPT-4 Turbo
    
    # Anthropic Models
    CLAUDE_35_SONNET = "claude-3-5-sonnet-latest"  # Latest most intelligent model
    CLAUDE_35_HAIKU = "claude-3-5-haiku-latest"  # Latest fastest model
    CLAUDE_3_OPUS = "claude-3-opus-latest"  # Most capable Claude 3
    
    # Azure OpenAI
    AZURE_GPT_4O = "azure/gpt-4o"
    AZURE_GPT_4 = "azure/gpt-4"
    AZURE_GPT_35 = "azure/gpt-35-turbo"
    
    # Open Source Models
    MISTRAL_LARGE = "mistral/mistral-large"
    MIXTRAL = "mistral/mixtral-8x7b"
    LLAMA2_70B = "meta-llama/llama-2-70b-chat" 