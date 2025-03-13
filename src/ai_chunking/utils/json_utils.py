"""Utility functions for JSON handling."""

import json
from typing import Any, Dict, Optional, Union, TypeVar, List

T = TypeVar('T')

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from text by scanning character by character.
    
    This function scans through the text character by character to find valid JSON objects,
    handling nested structures, escaped characters, and both object and array formats.
    It will return the first valid JSON object or array found in the text.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        Dict[str, Any]: The extracted JSON object
        
    Raises:
        ValueError: If no valid JSON object is found in the text or if JSON is malformed
        
    Examples:
        >>> extract_json_from_text('some text {"key": "value"} more text')
        {'key': 'value'}
        >>> extract_json_from_text('text {"nested": {"key": [1,2,3]}} text')
        {'nested': {'key': [1, 2, 3]}}
    """
    if not text:
        raise ValueError("Empty input text")
        
    start_idx = -1
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_char = ''
    
    for i, char in enumerate(text):
        # Handle escape sequences
        if escape_next:
            escape_next = False
            last_char = char
            continue
            
        if char == '\\' and not escape_next:
            escape_next = True
            last_char = char
            continue
            
        # Handle string boundaries
        if char == '"' and not escape_next:
            # Only toggle string state if quote is not escaped
            if last_char != '\\':
                in_string = not in_string
            last_char = char
            continue
            
        # Skip whitespace outside strings
        if not in_string and char.isspace():
            last_char = char
            continue
            
        # Process structural characters when not in a string
        if not in_string:
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if start_idx != -1 and brace_count == 0 and bracket_count == 0:
                    try:
                        json_str = text[start_idx:i+1].strip()
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Continue searching if this isn't valid JSON
                        continue
            elif char == '[':
                if start_idx == -1:
                    start_idx = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if start_idx != -1 and brace_count == 0 and bracket_count == 0:
                    try:
                        json_str = text[start_idx:i+1].strip()
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                        
        last_char = char
        
        # Check for malformed structure
        if brace_count < 0 or bracket_count < 0:
            raise ValueError("Malformed JSON: Unmatched closing brace/bracket")
            
    if start_idx != -1 and (brace_count > 0 or bracket_count > 0):
        raise ValueError(f"Malformed JSON: Unclosed {'braces' if brace_count > 0 else 'brackets'}")
            
    raise ValueError("No valid JSON object found in the text")

def safe_loads(json_str: str, default: Optional[T] = None) -> Union[Dict[str, Any], List[Any], T]:
    """
    Safely load a JSON string with error handling.
    
    Args:
        json_str: The JSON string to parse
        default: The default value to return if parsing fails
        
    Returns:
        The parsed JSON object or the default value if parsing fails
        
    Examples:
        >>> safe_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_loads('invalid json', default={})
        {}
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {} if default is None else default

def parse_json_response(response_text: str, key: Optional[str] = None, default: Optional[T] = None) -> Union[Dict[str, Any], List[Any], Any, T]:
    """
    Parse JSON from an LLM response text and optionally extract a specific key.
    
    This function first attempts to extract JSON from the text using extract_json_from_text,
    then optionally extracts a specific key from the resulting JSON object.
    
    Args:
        response_text: The text containing JSON to parse
        key: Optional key to extract from the parsed JSON
        default: Default value to return if parsing fails or key doesn't exist
        
    Returns:
        The parsed JSON object, or the value at the specified key, or the default value
        
    Examples:
        >>> parse_json_response('some text {"data": [1,2,3]} more text', key="data")
        [1, 2, 3]
        >>> parse_json_response('invalid', default=[])
        []
    """
    try:
        json_obj = extract_json_from_text(response_text)
        if key is not None:
            return json_obj.get(key, default)
        return json_obj
    except ValueError:
        return default if default is not None else {} 