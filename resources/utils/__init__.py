"""
Shared utilities for multimodal GenAI case studies.
"""

from .image_utils import (
    load_image,
    resize_image,
    image_to_numpy,
    numpy_to_image,
    batch_load_images,
    create_image_grid
)

from .text_utils import (
    clean_text,
    truncate_text,
    split_into_chunks,
    count_tokens_approximate,
    create_prompt_template,
    extract_code_blocks,
    read_text_file,
    save_text_file
)

__all__ = [
    # Image utilities
    'load_image',
    'resize_image',
    'image_to_numpy',
    'numpy_to_image',
    'batch_load_images',
    'create_image_grid',
    # Text utilities
    'clean_text',
    'truncate_text',
    'split_into_chunks',
    'count_tokens_approximate',
    'create_prompt_template',
    'extract_code_blocks',
    'read_text_file',
    'save_text_file',
]
