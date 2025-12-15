"""
Text processing utilities for multimodal GenAI projects.
"""

import re
from typing import List, Dict, Optional
from pathlib import Path


def clean_text(text: str, remove_extra_spaces: bool = True) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text string
        remove_extra_spaces: Whether to remove extra whitespace
        
    Returns:
        Cleaned text string
    """
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    if remove_extra_spaces:
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def truncate_text(
    text: str, 
    max_length: int, 
    add_ellipsis: bool = True
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum character length
        add_ellipsis: Whether to add '...' at the end
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    if add_ellipsis:
        truncated = truncated[:-3] + '...'
    
    return truncated


def split_into_chunks(
    text: str, 
    chunk_size: int, 
    overlap: int = 0
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks


def count_tokens_approximate(text: str, chars_per_token: int = 4) -> int:
    """
    Approximate token count for text.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (default: 4 for English)
        
    Returns:
        Estimated token count
    """
    return len(text) // chars_per_token


def create_prompt_template(
    template: str, 
    variables: Dict[str, str]
) -> str:
    """
    Fill in a prompt template with variables.
    
    Args:
        template: Template string with {variable} placeholders
        variables: Dictionary of variable names and values
        
    Returns:
        Filled template string
    """
    return template.format(**variables)


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown-style text.
    
    Args:
        text: Input text containing code blocks
        language: Optional language filter (e.g., 'python', 'javascript')
        
    Returns:
        List of extracted code blocks
    """
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def read_text_file(file_path: Path, encoding: str = 'utf-8') -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to text file
        encoding: Text encoding
        
    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def save_text_file(
    text: str, 
    file_path: Path, 
    encoding: str = 'utf-8'
) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text content to save
        file_path: Destination file path
        encoding: Text encoding
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(text)
