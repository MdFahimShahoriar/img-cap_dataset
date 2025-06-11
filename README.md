# Image Captioning Dataset (img-cap_dataset)

A comprehensive dataset for training and evaluating image captioning models, containing high-quality image-caption pairs for computer vision and natural language processing research.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Statistics](#dataset-statistics)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Preprocessing](#preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Baseline Models](#baseline-models)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## 🔍 Overview

The img-cap_dataset is designed to facilitate research in automatic image captioning, providing a large-scale collection of images paired with descriptive captions. This dataset can be used for:

- Training image captioning models
- Evaluating caption generation quality
- Research in vision-language understanding
- Multi-modal learning experiments
- Transfer learning applications

## 📊 Dataset Statistics

| Metric | Count |
|--------|-------|
| Total Images | [UPDATE_WITH_ACTUAL_COUNT] |
| Total Captions | [UPDATE_WITH_ACTUAL_COUNT] |
| Average Captions per Image | [UPDATE_WITH_ACTUAL_COUNT] |
| Average Caption Length | [UPDATE_WITH_ACTUAL_COUNT] words |
| Image Resolution Range | [UPDATE_WITH_ACTUAL_RANGE] |
| Dataset Size | [UPDATE_WITH_ACTUAL_SIZE] GB |

**Splits:**
- Training: [UPDATE_WITH_ACTUAL_COUNT] images
- Validation: [UPDATE_WITH_ACTUAL_COUNT] images  
- Test: [UPDATE_WITH_ACTUAL_COUNT] images

## 📁 Dataset Structure

```
img-cap_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img_val_001.jpg
│   │   └── ...
│   └── test/
│       ├── img_test_001.jpg
│       └── ...
├── annotations/
│   ├── train_captions.json
│   ├── val_captions.json
│   └── test_captions.json
├── metadata/
│   ├── image_info.json
│   └── dataset_statistics.json
├── scripts/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
└── requirements.txt
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup
1. Clone the repository:
```bash
git clone https://github.com/codewith-pavel/img-cap_dataset.git
cd img-cap_dataset
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
# Option 1: Direct download (if hosted)
python scripts/download_dataset.py

# Option 2: Manual download
# Follow the instructions in DOWNLOAD.md
```

## 💻 Usage

### Basic Data Loading

```python
import json
from PIL import Image
import os

class ImageCaptionDataset:
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.images_dir, ann['image_filename'])
        image = Image.open(image_path).convert('RGB')
        caption = ann['caption']
        return image, caption

# Load training data
train_dataset = ImageCaptionDataset(
    images_dir='images/train',
    annotations_file='annotations/train_captions.json'
)
```

### Using with PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create custom dataset class with transforms
class ImageCaptionDatasetPyTorch(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.images_dir, ann['image_filename'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, ann['caption']

# Create DataLoader
dataset = ImageCaptionDatasetPyTorch(
    'images/train', 
    'annotations/train_captions.json', 
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 📋 Data Format

### Annotation Files Structure

The annotation files (`train_captions.json`, `val_captions.json`, `test_captions.json`) follow this format:

```json
[
    {
        "image_id": "001",
        "image_filename": "img_001.jpg",
        "caption": "A descriptive caption of the image content",
        "caption_id": "001_1",
        "width": 640,
        "height": 480
    },
    {
        "image_id": "001",
        "image_filename": "img_001.jpg", 
        "caption": "Alternative caption for the same image",
        "caption_id": "001_2",
        "width": 640,
        "height": 480
    }
]
```

### Image Metadata

```json
{
    "image_id": "001",
    "filename": "img_001.jpg",
    "width": 640,
    "height": 480,
    "split": "train",
    "source": "dataset_source",
    "license": "license_info"
}
```

## 🔧 Preprocessing

### Data Cleaning and Preprocessing Scripts

```python
# Example preprocessing pipeline
from scripts.preprocessing import (
    resize_images,
    clean_captions,
    create_vocabulary,
    tokenize_captions
)

# Resize all images to consistent dimensions
resize_images(input_dir='images/raw', output_dir='images/processed', size=(224, 224))

# Clean and normalize captions
clean_captions(
    input_file='annotations/raw_captions.json',
    output_file='annotations/clean_captions.json'
)

# Create vocabulary from captions
vocab = create_vocabulary('annotations/clean_captions.json', min_freq=5)

# Tokenize captions
tokenize_captions(
    annotations_file='annotations/clean_captions.json',
    vocab_file='metadata/vocabulary.json',
    output_file='annotations/tokenized_captions.json'
)
```

## 📈 Evaluation Metrics

The dataset supports standard image captioning evaluation metrics:

- **BLEU (1-4)**: Measures n-gram overlap between generated and reference captions
- **METEOR**: Considers stemming, synonymy, and word order
- **ROUGE-L**: Longest common subsequence-based metric
- **CIDEr**: Consensus-based evaluation metric
- **SPICE**: Semantic evaluation using scene graphs

### Evaluation Script

```python
from scripts.evaluation import evaluate_captions

# Evaluate model predictions
results = evaluate_captions(
    predictions_file='results/predictions.json',
    references_file='annotations/test_captions.json'
)

print(f"BLEU-4: {results['bleu_4']:.3f}")
print(f"METEOR: {results['meteor']:.3f}")
print(f"CIDEr: {results['cider']:.3f}")
```

## 🏗️ Baseline Models

We provide implementations and pretrained models for several baseline approaches:

### Available Models
- **CNN-RNN**: ResNet + LSTM baseline
- **Show, Attend and Tell**: Attention-based model
- **Transformer**: Vision Transformer + GPT-style decoder
- **CLIP-based**: Using CLIP features for captioning

### Training a Baseline Model

```python
from scripts.train_baseline import train_model

# Train CNN-RNN baseline
model = train_model(
    model_type='cnn_rnn',
    train_data='annotations/train_captions.json',
    val_data='annotations/val_captions.json',
    epochs=50,
    batch_size=32,
    learning_rate=1e-4
)
```

## 📚 Examples and Tutorials

Check out the `examples/` directory for:
- Data exploration and visualization
- Training custom models
- Fine-tuning pretrained models
- Evaluation and analysis scripts

## 🔗 Related Datasets

- [MS COCO Captions](https://cocodataset.org/#captions-2015)
- [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/)
- [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)
- [Visual Genome](https://visualgenome.org/)

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{img_cap_dataset_2024,
    title={Image Captioning Dataset: A Large-Scale Collection for Vision-Language Research},
    author={Pavel, CodeWith and Contributors},
    year={2024},
    url={https://github.com/codewith-pavel/img-cap_dataset},
    version={1.0}
}
```

## 📜 License

This dataset is released under the [MIT License](LICENSE). Please check individual image licenses if applicable.

**Note**: Some images may have specific usage restrictions. Please verify licensing before commercial use.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute:
1. Fork the repository
2. Create a feature branch
3. Add your improvements (new data, better preprocessing, evaluation metrics)
4. Submit a pull request

### Areas for Contribution:
- Additional preprocessing scripts
- New evaluation metrics
- Baseline model implementations
- Data visualization tools
- Documentation improvements

## 📞 Contact

- **Author**: CodeWith Pavel
- **Email**: [UPDATE_WITH_ACTUAL_EMAIL]
- **GitHub**: [@codewith-pavel](https://github.com/codewith-pavel)
- **Issues**: [Report bugs or request features](https://github.com/codewith-pavel/img-cap_dataset/issues)

## 🙏 Acknowledgments

- Thanks to all contributors who helped collect and annotate the data
- Built with inspiration from existing captioning datasets
- Special thanks to the computer vision and NLP research community

## 📋 Changelog

### Version 1.0 (Current)
- Initial release with [X] images and captions
- Basic preprocessing and evaluation scripts
- Baseline model implementations

---

**Note**: This is a living document. Please check back for updates and improvements to the dataset and documentation.
