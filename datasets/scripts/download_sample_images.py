"""
Example script to download a sample dataset.
Customize this for your specific dataset needs.
"""

import requests
from pathlib import Path
from PIL import Image
import json


def download_coco_samples(output_dir: Path, num_samples: int = 10):
    """
    Download a few sample images from COCO dataset.
    
    Args:
        output_dir: Directory to save samples
        num_samples: Number of samples to download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_samples} COCO sample images...")
    
    # Example image URLs (replace with actual COCO URLs or use COCO API)
    sample_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # person
        # Add more URLs as needed
    ]
    
    metadata = []
    
    for idx, url in enumerate(sample_urls[:num_samples]):
        try:
            print(f"Downloading image {idx+1}/{num_samples}...")
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Save image
            image_path = output_dir / f"sample_{idx:03d}.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Verify image
            img = Image.open(image_path)
            width, height = img.size
            
            # Store metadata
            metadata.append({
                "filename": image_path.name,
                "url": url,
                "width": width,
                "height": height,
            })
            
            print(f"  ✓ Saved: {image_path.name} ({width}x{height})")
            
        except Exception as e:
            print(f"  ✗ Failed to download {url}: {e}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Downloaded {len(metadata)} samples to {output_dir}")
    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Download samples to datasets/samples/coco/
    output_path = Path(__file__).parent.parent / "samples" / "coco"
    download_coco_samples(output_path, num_samples=5)
