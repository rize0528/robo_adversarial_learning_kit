# VLM Adversarial Patch Finder

> 🚧 **Development Status**: 30% Complete (3/10 tasks finished)  
> A research framework for generating and demonstrating adversarial patches against Vision-Language Models (VLMs)

## 🎯 Project Overview

The VLM Adversarial Patch Finder is an AI security demonstration system designed for exhibitions that showcases how simple adversarial patches can compromise vision-language models in robotics applications. The system generates optimized adversarial patches and demonstrates their real-time attack effects on VLMs, educating stakeholders about critical AI security vulnerabilities.

## ✅ Completed Components

### 1. Environment & Model Setup (Task #3)
- **VLM Integration**: HuggingFace Transformers with Gemma-3 4B support
- **Memory Management**: Apple Silicon MPS acceleration with 16GB system limit
- **Model Loading**: Automatic fallback system and CUDA memory management
- **Testing**: 21/21 environment tests passing

### 2. Data Pipeline (Task #10) 
- **Image Processing**: Support for JPG, PNG, BMP, TIFF, TIF, WebP formats
- **Preprocessing**: VLM-compatible resize, normalize, and tensor conversion
- **Data Augmentation**: Rotation, brightness, contrast, saturation, hue, blur, noise
- **PyTorch Integration**: Custom Dataset and DataLoader with batch processing
- **Testing**: 70/70 data pipeline tests passing

### 3. Loss Functions (Task #5)
- **Core Framework**: Abstract LossFunction base class with GPU optimization
- **Attack Types**: 
  - TargetedLoss (4 targeting modes)
  - NonTargetedLoss (4 suppression modes)  
  - CompositeLoss (5 composition strategies)
- **Factory Pattern**: Easy configuration and instantiation
- **Testing**: 128/128 loss function tests passing

## 🏗️ Architecture

```
src/
├── data/
│   ├── dataset.py         # PyTorch Dataset/DataLoader
│   └── preprocessing.py   # Image processing pipeline
├── losses/
│   ├── base.py           # Abstract loss function framework
│   ├── targeted.py       # Targeted attack losses
│   ├── non_targeted.py   # Non-targeted attack losses
│   ├── composite.py      # Multi-objective composition
│   └── factory.py        # Loss function factory
├── models/
│   ├── vlm_loader.py     # Base VLM loader
│   └── gemma_vlm.py      # Gemma-3 implementation
└── utils/
    ├── image_utils.py    # Image processing utilities
    └── memory_utils.py   # System memory monitoring
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 16GB+ RAM (for VLM models)
- CUDA support (optional, MPS for Apple Silicon)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rize0528/robo_adversarial_learning_kit.git
cd robo_adversarial_learning_kit
```

2. **Switch to development branch**
```bash
# Switch to epic worktree for latest features
cd ../epic-vlm-adversarial-patch-finder
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage Examples

#### 1. Load and Test VLM Model

```python
from src.models.gemma_vlm import Gemma3VLM
from src.utils.memory_utils import check_memory_usage

# Check system memory
memory_info = check_memory_usage()
print(f"Available memory: {memory_info['available_gb']:.1f}GB")

# Load VLM model
vlm = Gemma3VLM()
model = vlm.load_model()

# Test basic inference
test_prompt = "Describe this image:"
result = vlm.generate_text(test_prompt)
print(result)
```

#### 2. Image Data Processing

```python
from src.data.preprocessing import ImagePreprocessor
from src.data.dataset import AdversarialDataset
import torch

# Initialize preprocessor
preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225]
)

# Create dataset
dataset = AdversarialDataset(
    image_paths=['image1.jpg', 'image2.png'],
    preprocessor=preprocessor
)

# Create DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

# Process batch
for batch in dataloader:
    images = batch['image']
    print(f"Batch shape: {images.shape}")
    break
```

#### 3. Loss Function Setup

```python
from src.losses.factory import LossFactory

# Create loss function factory
factory = LossFactory()

# Create targeted loss
targeted_loss = factory.create_loss_function(
    'targeted',
    mode='classification',
    target_class=10,
    confidence_threshold=0.8
)

# Create composite loss for multi-objective attack
composite_loss = factory.create_loss_function(
    'composite',
    strategy='weighted_sum',
    components=[
        {'type': 'targeted', 'weight': 0.7, 'target_class': 5},
        {'type': 'non_targeted', 'weight': 0.3, 'mode': 'confidence_reduction'}
    ]
)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/test_models.py -v          # VLM model tests
pytest tests/test_preprocessing.py -v   # Data pipeline tests  
pytest tests/test_loss_base.py -v       # Loss function tests
```

## 📊 Development Progress

| Component | Status | Tests | Description |
|-----------|--------|-------|-------------|
| Environment & Model Setup | ✅ Complete | 21/21 | VLM loading and inference |
| Data Pipeline | ✅ Complete | 70/70 | Image processing and datasets |
| Loss Functions | ✅ Complete | 128/128 | Adversarial loss implementations |
| Patch Optimization Core | 🔄 Pending | - | Gradient-based optimization |
| Camera Integration | 🔄 Pending | - | Live webcam feed processing |
| Demo Interface | 🔄 Pending | - | Gradio web application |
| Exhibition Presets | 🔄 Pending | - | Healthcare/manufacturing scenarios |
| Patch Generation Script | 🔄 Pending | - | End-to-end patch creation |
| Testing & Validation | 🔄 Pending | - | System reliability testing |
| Documentation & Packaging | 🔄 Pending | - | Deployment documentation |

## 🛠️ Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
models:
  gemma_3_4b:
    model_id: "google/gemma-3-4b-it"
    torch_dtype: "bfloat16"
    device_map: "auto"
    memory_limit_gb: 8
    
  gemma_3_9b:
    model_id: "google/gemma-2-9b-it"
    torch_dtype: "bfloat16"
    device_map: "auto"
    memory_limit_gb: 12

device_priority: ["mps", "cuda", "cpu"]
```

### Loss Function Configuration

```python
# Example configuration for different attack scenarios
loss_configs = {
    "healthcare_scenario": {
        "type": "composite",
        "strategy": "weighted_sum",
        "components": [
            {"type": "targeted", "weight": 0.6, "target_class": "normal"},
            {"type": "non_targeted", "weight": 0.4, "mode": "detection_suppression"}
        ]
    },
    "manufacturing_scenario": {
        "type": "targeted", 
        "mode": "confidence",
        "target_class": "defective",
        "confidence_threshold": 0.9
    }
}
```

## 🧪 Research Applications

### Supported Attack Types

1. **Targeted Attacks**: Force specific misclassifications
   - Classification targeting
   - Confidence manipulation
   - Margin-based attacks
   - Likelihood maximization

2. **Non-targeted Attacks**: General model suppression  
   - Confidence reduction
   - Entropy maximization
   - Logit minimization
   - Detection suppression

3. **Multi-objective Attacks**: Combined strategies
   - Weighted sum composition
   - Adaptive weight adjustment
   - Hierarchical optimization
   - Pareto optimal solutions

### Performance Benchmarks

- **Inference Speed**: <5 seconds per frame on CPU
- **Memory Usage**: <16GB peak consumption  
- **Attack Success Rate**: Target >70% effectiveness
- **System Reliability**: 8-hour continuous operation

## 🔧 Development

### Project Structure

- **Main Repository**: Core project files and documentation
- **Epic Worktree**: `../epic-vlm-adversarial-patch-finder/` - Active development
- **Task Tracking**: `.claude/epics/vlm-adversarial-patch-finder/` - Progress tracking

### GitHub Issues

- **Epic Issue**: [#1](https://github.com/rize0528/robo_adversarial_learning_kit/issues/1)
- **Completed Tasks**: [#3](https://github.com/rize0528/robo_adversarial_learning_kit/issues/3), [#5](https://github.com/rize0528/robo_adversarial_learning_kit/issues/5), [#10](https://github.com/rize0528/robo_adversarial_learning_kit/issues/10)
- **Active Tasks**: [#2](https://github.com/rize0528/robo_adversarial_learning_kit/issues/2), [#4](https://github.com/rize0528/robo_adversarial_learning_kit/issues/4), [#6-9](https://github.com/rize0528/robo_adversarial_learning_kit/issues/6), [#11](https://github.com/rize0528/robo_adversarial_learning_kit/issues/11)

### Contributing

1. Work in the epic worktree: `cd ../epic-vlm-adversarial-patch-finder`  
2. Pick an open task from the GitHub issues
3. Follow the existing code patterns and testing standards
4. Ensure all tests pass before submitting changes

## ⚠️ Security Notice

This project is designed for **defensive security research and education**. The adversarial patches generated should only be used to:

- Demonstrate AI security vulnerabilities  
- Educate stakeholders about model robustness
- Develop defensive countermeasures
- Academic research in AI safety

**Do not use this system for malicious purposes.**

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Acknowledgments

- HuggingFace Transformers for VLM integration
- PyTorch for neural network framework
- OpenCV for image processing capabilities

---

**Status**: Active Development | **Version**: 0.3.0 | **Updated**: 2025-09-08