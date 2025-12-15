# Quick Start Guide

## Setting Up Your Workspace

### 1. Initial Setup

```bash
# Navigate to the repository
cd multi-modal-genai-case-studies

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install common dependencies
pip install torch transformers pillow numpy pandas matplotlib jupyter
```

### 2. Exploring Existing Case Studies

```bash
# Navigate to a case study
cd case-studies/browbake

# Install case-specific dependencies
pip install -r requirements.txt  # (create this file if needed)

# Open the notebook
jupyter notebook Notebook.ipynb
```

### 3. Creating a New Case Study

```bash
# Copy the template
cp -r templates/case-study-template case-studies/my-new-study

# Navigate to your new case study
cd case-studies/my-new-study

# Edit the README.md with your case study details
# Edit Notebook.ipynb to implement your experiments

# Create requirements.txt with your dependencies
# Run and iterate!
```

## Common Tasks

### Running a Quick Experiment

```bash
# Create a new notebook in the notebooks/ folder
cd notebooks
jupyter notebook my_experiment.ipynb
```

### Using Shared Utilities

```python
# Add the resources directory to your path
import sys
sys.path.append('../resources')

# Import utilities
from utils import load_image, resize_image, clean_text
```

### Working with Datasets

```bash
# Create a dataset directory
mkdir -p datasets/raw/my-dataset

# Add a download script
touch datasets/scripts/download_my_dataset.py

# Document your dataset
echo "# My Dataset" > datasets/raw/my-dataset/README.md
```

## Recommended Workflow

### For Learning

1. Start with the [common workflows](resources/docs/common-workflows.md) documentation
2. Create an experimental notebook in `notebooks/`
3. Try different models and approaches
4. Document interesting findings

### For Case Studies

1. Define your objectives clearly
2. Copy the case study template
3. Gather or prepare your dataset
4. Implement step by step with documentation
5. Analyze results and document learnings
6. Share your findings in the README

## Tips for Success

### 1. Start Small

- Use small datasets for initial testing
- Start with simple models before complex ones
- Test on a few samples before full runs

### 2. Document as You Go

- Add comments to your code
- Take notes on what works and what doesn't
- Screenshot interesting outputs
- Record performance metrics

### 3. Use Version Control

```bash
# Initialize git (if not already done)
git init

# Add files incrementally
git add case-studies/my-study/README.md
git commit -m "Add new case study: my-study"

# Push to remote (if configured)
git push origin main
```

### 4. Manage Resources

- Monitor GPU memory usage
- Clear cache between experiments
- Save intermediate results
- Use checkpointing for long runs

### 5. Stay Organized

- Name files descriptively
- Keep related files together
- Update README files regularly
- Clean up old experiments

## Example: Running Your First Vision-Language Model

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
url = "https://images.unsplash.com/photo-1537151625747-768eb6cf92b2"
image = Image.open(requests.get(url, stream=True).raw)

# Generate caption
inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print(f"Caption: {caption}")
```

## Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size
batch_size = 1

# Use smaller model variant
model_name = "model-name-small"  # instead of "model-name-large"

# Clear cache
import torch
torch.cuda.empty_cache()
```

### Import Errors

```bash
# Reinstall package
pip uninstall package-name
pip install package-name

# Check installed version
pip show package-name

# Install specific version
pip install package-name==1.2.3
```

### Slow Inference

```python
# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Use half precision
model.half()

# Batch processing
# Process multiple items at once
```

## Resources

- **Documentation**: See [resources/docs/](resources/docs/) for detailed guides
- **Utilities**: Use [resources/utils/](resources/utils/) for common functions
- **Templates**: Find templates in [templates/](templates/)
- **Examples**: Check existing case studies in [case-studies/](case-studies/)

## Getting Help

- Check the [best practices guide](resources/docs/best-practices.md)
- Review [common workflows](resources/docs/common-workflows.md)
- Look at existing case studies for examples
- Search Hugging Face documentation
- Check model cards on Hugging Face Hub

## Next Steps

1. ✅ Read this guide
2. ✅ Set up your environment
3. 📝 Try the example code above
4. 🚀 Create your first experiment notebook
5. 🎯 Start a case study using the template
6. 📚 Document and share your findings

Happy experimenting! 🎉
