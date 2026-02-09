# CUB-200 Bird Classification
This project implements deep learning models for fine-grained visual recognition, specifically classifying 200 bird species from the CUB-200 dataset. It features two state-of-the-art architectures with advanced training techniques including cyclical learning rates, gradual unfreezing, and evaluation metrics.

**Key Features:**
- **Dual Architecture**: ResNet50 and EfficientNet-B3 with custom classification heads
- **Two-Stage Training**: Gradual unfreezing strategy for optimal transfer learning
- **Learning Rate Optimization**: LR Finder + Triangular Cyclical LR for super-convergence
- **Evaluation Metrics**: Top-1, Top-5, and per-class accuracy evaluation
- **Advanced Augmentation**: Multi-stage data augmentation pipeline

## Dataset
**CUB-200-2011 (Caltech-UCSD Birds)**
- 200 bird species
- 11,788 images total
- Split: ~5,994 training, ~5,794 test images
- Fine-grained classification challenge with high intra-class variation

## Architecture
### ResNet50
- **Input Size**: 224×224
- **Base Model**: Pre-trained on ImageNet
- **Custom Head**: 
  - Global Average Pooling
  - Dropout (0.5)
  - Fully Connected (2048 → 200)
- **Features**: 2048-dimensional embeddings

### EfficientNet-B3
- **Input Size**: 300×300
- **Base Model**: Pre-trained on ImageNet
- **Custom Head**:
  - Global Average Pooling
  - Dropout (0.5)
  - Fully Connected (1536 → 200)
- **Features**: 1536-dimensional embeddings

## Training Strategy
### Two-Stage Transfer Learning
**Stage 1: Head Training (20 epochs)**
- Freeze all backbone layers
- Train only custom classification head
- Faster convergence on new task

**Stage 2: Gradual Unfreezing (30 epochs)**
- Unfreeze backbone layers progressively
- Fine-tune entire network
- Preserve low-level features while adapting high-level representations

### Learning Rate Scheduling
**LR Range Test**
- Automated learning rate discovery
- Exponential sweep from 1e-6 to 1.0
- Identifies optimal LR range for training

**Triangular Cyclical LR**
- Base LR: 1e-4 (refined by LR Finder)
- Max LR: 1e-2 (refined by LR Finder)
- Step size: 4 epochs per half-cycle
- Enables super-convergence and improved generalization

### Data Augmentation
**Training Augmentation:**
- Random resized crop
- Random horizontal flip (50%)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation: ±20%)
- Normalization (ImageNet statistics)

**Validation/Test:**
- Center crop
- Normalization only

### Configuration
```python
BATCH_SIZE = 32
EPOCHS_STAGE1 = 20      # Head training
EPOCHS_STAGE2 = 30      # Fine-tuning
VAL_SPLIT = 0.2         # 80-20 train-val split
BASE_LR = 1e-4          # Cyclical LR minimum
MAX_LR = 1e-2           # Cyclical LR maximum
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4
PATIENCE = 10           # Early stopping
```

## Evaluation Metrics
**Top-1 Accuracy**
- Standard classification accuracy
- Percentage of correctly predicted first choices

**Top-5 Accuracy**
- Correct class in top 5 predictions
- Useful for fine-grained classification

**Average Per-Class Accuracy**
- Mean accuracy across all 200 classes
- Handles class imbalance
- More robust metric for imbalanced datasets

## Results
### Model Performance
| Model | Top-1 Acc | Top-5 Acc | Avg Per-Class Acc | Training Time |
|-------|-----------|-----------|-------------------|---------------|
| ResNet50 | TBD% | TBD% | TBD% | ~X hours |
| EfficientNet-B3 | TBD% | TBD% | TBD% | ~X hours |

## Implementation Details
### Model Components
**Custom Classification Head:**
```python
nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(features, num_classes)
)
```

**Optimizer:** Adam with weight decay (1e-4)

**Loss Function:** CrossEntropyLoss

**Scheduler:** CyclicLR (triangular mode)

### Training Loop Features
- Automatic mixed precision (AMP) for faster training
- Gradient clipping for stability
- Model checkpointing every 5 epochs
- Early stopping with patience=10
- Progress tracking with tqdm
- Comprehensive metric logging

### Reproducibility
All random seeds are fixed:
```python
RANDOM_SEED = 42
- Python random
- NumPy
- PyTorch (CPU & CUDA)
- CuDNN (deterministic mode)
```

## Key Techniques
### Transfer Learning
- Leverage ImageNet pre-trained weights
- Domain adaptation for bird species
- Preserve low-level visual features

### Cyclical Learning Rates
- Faster convergence than fixed LR
- Better generalization
- Escape local minima
- Based on Leslie Smith's research

### Gradual Unfreezing
- Prevents catastrophic forgetting
- Preserves pre-trained features
- Progressive adaptation to new domain

### Learning Rate Finder
- Automated hyperparameter tuning
- Identifies optimal LR range
- Based on loss curve analysis

## Troubleshooting
**Poor Convergence:**
- Run LR Finder to adjust learning rates
- Increase `EPOCHS_STAGE1` for better head initialization
- Check data augmentation isn't too aggressive

**Overfitting:**
- Increase `DROPOUT_RATE`
- Add more data augmentation
- Increase `WEIGHT_DECAY`
- Reduce model capacity

## Advanced Features
### LR Range Test Visualization
- Plots loss vs learning rate
- Identifies optimal LR range
- Suggests base_lr and max_lr values

### Per-Class Analysis
- Identifies difficult species
- Confusion matrix visualization
- Class-specific accuracy metrics

### Model Comparison
- Side-by-side performance metrics
- Efficiency analysis (params, FLOPs)
- Inference speed benchmarks
---

*Educational project for fine-grained visual recognition using transfer learning*
