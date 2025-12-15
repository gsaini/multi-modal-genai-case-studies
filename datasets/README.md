# Datasets

This directory contains sample datasets and data preparation scripts for multimodal GenAI experiments.

## Structure

```
datasets/
├── raw/              # Original unprocessed datasets
├── processed/        # Cleaned and preprocessed data
├── scripts/          # Data preparation and augmentation scripts
└── samples/          # Small sample datasets for testing
```

## Dataset Guidelines

### Adding a New Dataset

1. **Documentation**: Create a README with:
   - Dataset name and version
   - Source and download link
   - License information
   - Format and structure description
   - Sample usage code

2. **Organization**:
   - Place raw data in `raw/dataset-name/`
   - Save processed data in `processed/dataset-name/`
   - Include preprocessing scripts in `scripts/`

3. **Size Considerations**:
   - For large datasets (>100MB), add to `.gitignore`
   - Include download scripts instead
   - Provide small samples for testing

### Common Dataset Types

#### Image Datasets
- COCO, ImageNet subsets
- Custom image collections
- Image-text pairs

#### Text Datasets
- Instruction datasets
- QA pairs
- Document collections

#### Audio Datasets
- Speech recordings
- Audio-text alignments
- Music datasets

#### Video Datasets
- Short clips
- Video-text descriptions
- Action recognition data

## Best Practices

1. **Always document the source and license**
2. **Include data statistics and exploratory analysis**
3. **Provide data loading utilities**
4. **Keep samples small for version control**
5. **Use standard formats (CSV, JSON, Parquet)**

## Download Scripts

Example script to download a dataset:
```python
# scripts/download_coco_sample.py
import requests
from pathlib import Path

def download_coco_sample():
    # Implementation here
    pass
```
