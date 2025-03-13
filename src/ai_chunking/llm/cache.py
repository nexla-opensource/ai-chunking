"""Caching utilities for LLM responses."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Type, Union, Dict
import logging
from enum import Enum

from diskcache import Cache, FanoutCache
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class CacheBackend(str, Enum):
    """Supported cache backend types."""
    DISK = "disk"
    MEMORY = "memory"
    FANOUT = "fanout"  # Multi-process safe disk cache

class CacheConfig:
    """Configuration for LLM response caching."""
    
    def __init__(
        self,
        enabled: bool = False,
        backend: CacheBackend = CacheBackend.DISK,
        directory: Optional[str] = None,
        ttl: Optional[int] = None,
        shards: int = 8,  # Number of shards for fanout cache
        size_limit: Optional[int] = None,  # Cache size limit in bytes
        **kwargs: Any
    ):
        """Initialize cache configuration.
        
        Args:
            enabled: Whether caching is enabled
            backend: Cache backend type (disk, memory, or fanout)
            directory: Cache directory path. If None, uses ~/.ai_chunking/llm_cache
            ttl: Default time-to-live for cache entries in seconds
            shards: Number of shards for fanout cache (only used with fanout backend)
            size_limit: Maximum cache size in bytes (None for no limit)
            **kwargs: Additional backend-specific configuration
        """
        self.enabled = enabled
        self.backend = backend
        self.directory = directory or os.path.expanduser("~/.ai_chunking/llm_cache")
        self.ttl = ttl
        self.shards = shards
        self.size_limit = size_limit
        self.extra_config = kwargs

class LLMCache:
    """Cache for LLM responses with configurable backend."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize LLM cache.
        
        Args:
            config: Cache configuration. If None, uses environment variables.
        """
        self.config = config or self._load_config_from_env()
        
        if not self.config.enabled:
            logger.info("LLM caching is disabled")
            return
            
        # Create cache directory if needed
        if self.config.backend in (CacheBackend.DISK, CacheBackend.FANOUT):
            Path(self.config.directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate cache backend
        if self.config.backend == CacheBackend.FANOUT:
            self.cache = FanoutCache(
                directory=self.config.directory,
                shards=self.config.shards,
                size_limit=self.config.size_limit,
                **self.config.extra_config
            )
        elif self.config.backend == CacheBackend.DISK:
            self.cache = Cache(
                directory=self.config.directory,
                size_limit=self.config.size_limit,
                **self.config.extra_config
            )
        else:  # Memory cache
            self.cache = {}
            
        logger.info(f"Initialized {self.config.backend.value} cache at {self.config.directory}")
    
    @staticmethod
    def _load_config_from_env() -> CacheConfig:
        """Load cache configuration from environment variables.
        
        Environment variables:
            AI_CHUNKING_CACHE_ENABLED: Enable/disable caching (default: false)
            AI_CHUNKING_CACHE_BACKEND: Cache backend type (disk/memory/fanout)
            AI_CHUNKING_CACHE_DIR: Cache directory path
            AI_CHUNKING_CACHE_TTL: Default TTL in seconds
            AI_CHUNKING_CACHE_SIZE_LIMIT: Cache size limit in bytes
            AI_CHUNKING_CACHE_SHARDS: Number of shards for fanout cache
            
        Returns:
            CacheConfig instance
        """
        return CacheConfig(
            enabled=os.getenv("AI_CHUNKING_CACHE_ENABLED", "false").lower() == "true",
            backend=CacheBackend(os.getenv("AI_CHUNKING_CACHE_BACKEND", "disk").lower()),
            directory=os.getenv("AI_CHUNKING_CACHE_DIR"),
            ttl=int(os.getenv("AI_CHUNKING_CACHE_TTL")) if os.getenv("AI_CHUNKING_CACHE_TTL") else None,
            size_limit=int(os.getenv("AI_CHUNKING_CACHE_SIZE_LIMIT")) if os.getenv("AI_CHUNKING_CACHE_SIZE_LIMIT") else None,
            shards=int(os.getenv("AI_CHUNKING_CACHE_SHARDS", "8"))
        )
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float, **kwargs: Any) -> str:
        """Generate a unique cache key for the LLM request.
        
        Args:
            prompt: The input prompt
            model: The model name/identifier
            temperature: The sampling temperature
            **kwargs: Additional parameters that affect the response
            
        Returns:
            A unique hash string for the request parameters
        """
        # Create a dictionary of all parameters that affect the response
        params = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            **{k: v for k, v in kwargs.items() if not callable(v)}
        }
        
        # Convert to stable string representation and hash
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache based on backend type."""
        if not self.config.enabled:
            return None
            
        if self.config.backend == CacheBackend.MEMORY:
            return self.cache.get(key)
        return self.cache.get(key)
    
    def _set_in_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache based on backend type."""
        if not self.config.enabled:
            return
            
        ttl = ttl or self.config.ttl
        if self.config.backend == CacheBackend.MEMORY:
            self.cache[key] = value
        else:
            self.cache.set(key, value, expire=ttl)
    
    def cache_llm_response(self, ttl: Optional[int] = None) -> Callable:
        """Decorator to cache LLM responses.
        
        Args:
            ttl: Time-to-live in seconds for cache entries. None means use config default.
            
        Returns:
            Decorated function that uses caching
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(
                self_obj: Any,
                prompt: str,
                response_model: Optional[Type[T]] = None,
                **kwargs: Any
            ) -> Union[str, T]:
                if not self.config.enabled:
                    return await func(self_obj, prompt, response_model, **kwargs)
                
                # Generate cache key from request parameters
                cache_key = self._generate_cache_key(
                    prompt=prompt,
                    model=self_obj.model,
                    temperature=self_obj.temperature,
                    response_model=response_model.__name__ if response_model else None,
                    **kwargs
                )
                
                # Try to get from cache
                cached_response = self._get_from_cache(cache_key)
                if cached_response is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    if response_model:
                        return response_model.model_validate(cached_response)
                    return cached_response
                
                # If not in cache, call original function
                response = await func(self_obj, prompt, response_model, **kwargs)
                
                # Cache the response
                to_cache = response.model_dump() if isinstance(response, BaseModel) else response
                self._set_in_cache(cache_key, to_cache, ttl)
                logger.debug(f"Cached response for key: {cache_key}")
                
                return response
                
            return wrapper
        return decorator

# Global cache instance - disabled by default, enabled via environment variables
llm_cache = LLMCache() 