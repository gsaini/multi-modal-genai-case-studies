# Multi-Modal GenAI Case Studies

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-8A2BE2?style=for-the-badge&logo=stability-ai&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A personal collection of multimodal generative AI case studies, experiments, and learning materials. This repository serves as a practical workspace for exploring, reproducing, and sharing compact examples that demonstrate common multimodal GenAI workflows.

## 📋 Repository Structure

```
.
├── case-studies/          # Complete case study implementations
│   └── browbake/         # Vision-language model case study
├── datasets/             # Sample datasets and data preparation scripts
├── notebooks/            # Experimental and learning notebooks
├── templates/            # Templates for new case studies
└── resources/            # Shared utilities, configs, and documentation
```

## 🎯 Case Studies

### [Browbake](case-studies/browbake/)

Vision-language model implementation exploring multimodal AI capabilities.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab or VS Code with Jupyter extension
- GPU recommended for vision models (CUDA support)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/multi-modal-genai-case-studies.git
cd multi-modal-genai-case-studies
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (per case study):

```bash
cd case-studies/browbake
pip install -r requirements.txt
```

## 📚 Topics Covered

- **Vision-Language Models**: Image captioning, VQA, visual reasoning
- **Text-to-Image**: Stable Diffusion, DALL-E workflows
- **Audio-Visual**: Speech recognition with visual context
- **Document Understanding**: OCR, layout analysis, document QA
- **Multimodal RAG**: Retrieval-augmented generation with multiple modalities

## 🛠️ Creating a New Case Study

Use the template in `templates/case-study-template/`:

1. Copy the template to `case-studies/your-study-name/`
2. Update the README with your case study details
3. Develop your notebook following the template structure
4. Add dataset information and requirements
5. Document results and learnings

## 📖 Resources

- **Shared Utilities**: Common functions in `resources/utils/`
- **Configuration Templates**: Model configs in `resources/configs/`
- **Documentation**: Best practices and patterns in `resources/docs/`

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to open an issue or PR.

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Useful Links

- [Hugging Face Models](https://huggingface.co/models)
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Multimodal Learning Papers](https://paperswithcode.com/task/multimodal-learning)
