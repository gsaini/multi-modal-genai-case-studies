"""
Image processing utilities for multimodal GenAI projects.
"""

from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from pathlib import Path


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    return Image.open(image_path).convert('RGB')


def resize_image(
    image: Image.Image, 
    size: Tuple[int, int], 
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: PIL Image object
        size: Target size as (width, height)
        maintain_aspect: If True, maintains aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image object
        
    Returns:
        Numpy array of shape (H, W, C)
    """
    return np.array(image)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array of shape (H, W, C)
        
    Returns:
        PIL Image object
    """
    return Image.fromarray(array.astype('uint8'))


def batch_load_images(image_paths: List[Union[str, Path]]) -> List[Image.Image]:
    """
    Load multiple images from file paths.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of PIL Image objects
    """
    return [load_image(path) for path in image_paths]


def create_image_grid(
    images: List[Image.Image], 
    grid_size: Tuple[int, int] = None,
    cell_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        grid_size: Grid dimensions as (rows, cols). Auto-calculated if None
        cell_size: Size of each cell in the grid
        
    Returns:
        Combined grid image
    """
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    cell_width, cell_height = cell_size
    
    # Create blank canvas
    grid_image = Image.new(
        'RGB', 
        (cols * cell_width, rows * cell_height),
        color='white'
    )
    
    # Paste images into grid
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        # Resize image to cell size
        img_resized = resize_image(img, cell_size, maintain_aspect=True)
        
        # Calculate position to center the image in cell
        x = col * cell_width + (cell_width - img_resized.width) // 2
        y = row * cell_height + (cell_height - img_resized.height) // 2
        
        grid_image.paste(img_resized, (x, y))
    
    return grid_image
