# Common Multimodal GenAI Workflows

## 1. Image Captioning

### Basic Workflow
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and process image
image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")

# Generate caption
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)
```

### Use Cases
- Accessibility (alt text generation)
- Content moderation
- Search and retrieval
- Creative storytelling

## 2. Visual Question Answering (VQA)

### Basic Workflow
```python
from transformers import ViltProcessor, ViltForQuestionAnswering

# Load model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Prepare inputs
question = "What color is the car?"
image = Image.open("car.jpg")
inputs = processor(image, question, return_tensors="pt")

# Get answer
outputs = model(**inputs)
logits = outputs.logits
predicted_idx = logits.argmax(-1).item()
answer = model.config.id2label[predicted_idx]
```

### Use Cases
- Educational tools
- Product information
- Medical diagnosis support
- Customer service

## 3. Text-to-Image Generation

### Basic Workflow (Stable Diffusion)
```python
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A serene lake at sunset with mountains in the background"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

### Advanced Techniques
- Negative prompts for better control
- Image-to-image generation
- Inpainting and outpainting
- ControlNet for precise control
- LoRA fine-tuning

## 4. Document Understanding

### OCR + Layout Analysis
```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# Load model for document understanding
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Process document
image = Image.open("document.jpg")
encoding = processor(image, return_tensors="pt")

# Extract entities
outputs = model(**encoding)
predictions = outputs.logits.argmax(-1)
```

### Use Cases
- Invoice processing
- Form extraction
- Document classification
- Table extraction

## 5. Multimodal RAG (Retrieval-Augmented Generation)

### Basic Workflow
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import ImageCaptionLoader

# Create vector store with image captions
loader = ImageCaptionLoader(path="images/")
documents = loader.load()

# Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Query with multimodal context
query = "Show me images with cats"
results = vectorstore.similarity_search(query, k=5)
```

### Components
- Vision encoder for images
- Text encoder for queries
- Vector database for storage
- LLM for response generation

## 6. Audio-Visual Processing

### Speech-to-Text with Visual Context
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Process audio
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# Transcribe
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
```

### Applications
- Video subtitling
- Meeting transcription
- Accessibility features
- Content analysis

## 7. Image Segmentation + Description

### Workflow
```python
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import torch

# Load segmentation model
processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Segment based on text prompt
image = Image.open("image.jpg")
texts = ["cat", "dog", "person"]
inputs = processor(text=texts, images=[image]*len(texts), return_tensors="pt")

# Get segmentation masks
outputs = model(**inputs)
logits = outputs.logits
```

## 8. Video Understanding

### Frame Extraction + Analysis
```python
import cv2
from transformers import pipeline

# Extract key frames
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# Analyze frames
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
frames = extract_frames("video.mp4")
captions = [captioner(frame)[0]['generated_text'] for frame in frames]
```

## 9. Multimodal Search

### Building a Multimodal Search System
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load CLIP model for multimodal embeddings
model = SentenceTransformer('clip-ViT-B-32')

# Embed images and text
image_embeddings = model.encode([Image.open(f) for f in image_files])
text_embeddings = model.encode(descriptions)

# Search with text query
query = "red sports car"
query_embedding = model.encode(query)

# Find similar images
similarities = np.dot(image_embeddings, query_embedding)
top_k = np.argsort(similarities)[-5:]
```

## 10. Creative Applications

### Style Transfer
```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Transform image with prompt
init_image = Image.open("photo.jpg")
prompt = "oil painting, impressionist style"
image = pipe(prompt=prompt, image=init_image, strength=0.75).images[0]
```

### Applications
- Artistic rendering
- Photo enhancement
- Character design
- Scene generation

## Performance Tips

### Batch Processing
```python
# Process multiple images efficiently
images = [Image.open(f) for f in image_files]
inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
```

### Model Optimization
```python
# Quantization for faster inference
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    "model-name",
    quantization_config=quantization_config
)
```

## Resources

- **Hugging Face Transformers**: Pre-trained models and pipelines
- **LangChain**: Building multimodal applications
- **Diffusers**: Text-to-image and image-to-image models
- **CLIP**: Connecting vision and language
- **LLaVA**: Large Language and Vision Assistant
