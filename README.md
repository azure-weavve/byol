# BYOL-based Wafer Pattern Clustering System

Self-supervised learning system for semiconductor wafer defect pattern clustering using Bootstrap Your Own Latent (BYOL).

## Project Overview

This project implements a BYOL-based approach for learning wafer defect patterns with the following goals:

1. **Latent vector distance ≈ Pattern similarity**: Similar patterns should be close in latent space
2. **High clustering quality**: Latent vectors should cluster well using HDBSCAN
3. **Rotation invariance**: D4 rotations of the same pattern should cluster together

### Why BYOL?

Advantages over VQ-VAE + Contrastive Learning:
- No negative samples needed (avoids false negatives)
- Continuous latent space (no discrete codebook limitations)
- Flexible batch sizes (works well with 12GB VRAM, batch_size=256)
- Natural compatibility with D4 augmentation

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.4.0 (FIXED - do not upgrade!)
- CUDA compatible with PyTorch 1.4.0
- 12GB VRAM GPU (recommended)

### Setup

```bash
# Install PyTorch 1.4.0 (adjust CUDA version as needed)
pip install torch==1.4.0 torchvision==0.5.0

# Install other dependencies
pip install -r requirements.txt
```

## Project Structure

```
BYOL/
├── CLAUDE_BYOL.md              # Detailed implementation guide
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── main_byol_training.py       # Main training script
│
├── models/
│   ├── __init__.py
│   ├── encoder.py              # WaferEncoder (ResNet-18 based)
│   ├── projector.py            # Projector & Predictor
│   └── byol.py                 # Complete BYOL model
│
├── utils/
│   ├── __init__.py
│   ├── augmentation.py         # D4 + wafer augmentations
│   ├── train_byol.py           # Training functions
│   ├── evaluation.py           # Evaluation metrics
│   └── byol_monitor.py         # Training monitor
│
├── functions/
│   └── __init__.py
│
└── experiments/
    └── __init__.py
```

## Quick Start

### 1. Prepare Data

Replace the dummy dataloader in `main_byol_training.py` with your actual wafer data:

```python
# TODO: Implement actual data loading
# Expected format: (N, 1, H, W) grayscale wafer maps
# Values: 0 (good/non-wafer), 1 (defect)
```

### 2. Train Model

```bash
# Basic training
python main_byol_training.py

# With custom parameters
python main_byol_training.py --epochs 100 --batch_size 256 --lr 0.0001

# Resume from checkpoint
python main_byol_training.py --resume checkpoints/checkpoint_epoch_50.pth
```

### 3. Monitor Training

Training logs and plots are saved to `logs/`:
- `training_curves.png`: Loss, LR, tau, collapse metrics
- `evaluation_metrics.png`: Retrieval, clustering, rotation invariance
- `latent_space_epoch_*.png`: t-SNE visualizations
- `history.json`: Complete training history

### 4. Evaluate Model

```python
from models.byol import BYOL
from utils.evaluation import evaluate_all
from utils.train_byol import extract_features

# Load model
model = BYOL(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
metrics, labels = evaluate_all(model, dataloader, device)
```

## Key Features

### D4 Dihedral Group Augmentation

Critical for rotation invariance! 8 symmetry transformations:
- Identity, 90°, 180°, 270° rotations
- Horizontal, vertical, diagonal flips

```python
from utils.augmentation import D4Transform

# Apply random D4 transform
transformed, transform_id = D4Transform.random_transform(wafer_map)

# Get all 8 transforms
all_transforms = D4Transform.get_all_transforms(wafer_map)
```

### BYOL Architecture

```
Online Network:  encoder → projector → predictor
Target Network:  encoder → projector (EMA updated)

Loss: symmetric_loss(pred_online, proj_target)
```

### Evaluation Metrics

1. **Retrieval Quality**
   - Precision@k, Recall@k, MRR
   - Average distance to top-k neighbors

2. **Clustering Quality**
   - Silhouette Score (target: ≥ 0.5)
   - Calinski-Harabasz Index
   - Davies-Bouldin Index
   - Noise ratio (HDBSCAN)

3. **Rotation Invariance**
   - Cosine similarity within D4 group (target: ≥ 0.95)
   - Cluster consistency (D4 transforms in same cluster)

## Training Configuration

Default hyperparameters (see `main_byol_training.py`):

```python
# Model
encoder_dim = 512
projector_out = 256
use_radial_encoding = True  # Wafer center distance encoding
use_attention = True        # Self-attention for global patterns

# Training
batch_size = 256           # Max for 12GB VRAM
epochs = 100
base_lr = 0.0001
weight_decay = 0.01

# BYOL
tau_base = 0.996          # Initial EMA momentum
tau_max = 0.999           # Final EMA momentum (cosine schedule)

# Augmentation
augmentation_type = 'strong'  # D4 + crop + noise
```

## Performance Targets

| Metric | Target | VQ-VAE (baseline) |
|--------|--------|-------------------|
| Retrieval Precision@5 | ≥ 70% | ~3% |
| Silhouette Score | ≥ 0.5 | ~0.4 |
| Rotation Invariance (D4) | ≥ 95% | TBD |
| Cluster Count | 10-50 | - |
| Noise Ratio (HDBSCAN) | < 20% | - |

## PyTorch 1.4.0 Compatibility

**Important limitations:**
```python
# ❌ Not available in PyTorch 1.4.0
torch.quantile()
nn.SyncBatchNorm
torch.cuda.amp

# ✅ Use these instead
torch.kthvalue()       # For quantile
nn.BatchNorm2d         # For batch norm
# No automatic mixed precision
```

## Troubleshooting

### Collapse Detection

**Symptoms:** Loss → 0, all features identical

**Solutions:**
1. Verify predictor exists (critical!)
2. Increase tau (0.999)
3. Check BatchNorm is in training mode
4. Reduce learning rate

### Training Instability

**Symptoms:** Loss oscillations, NaN values

**Solutions:**
1. Reduce learning rate (1e-4 → 1e-5)
2. Add gradient clipping
3. Increase warmup period

### Low Clustering Quality

**Symptoms:** Silhouette < 0.3, high noise ratio

**Solutions:**
1. Ensure D4 augmentation is enabled (critical!)
2. Train longer (100+ epochs)
3. Adjust HDBSCAN `min_cluster_size`
4. Increase projector output dimension

### CUDA Out of Memory

**Solutions:**
1. Reduce batch_size (256 → 128)
2. Use gradient accumulation
3. Delete unused tensors + `torch.cuda.empty_cache()`

## Testing Components

Each module has a test function:

```bash
# Test encoder
python models/encoder.py

# Test projector and predictor
python models/projector.py

# Test BYOL model
python models/byol.py

# Test augmentation
python utils/augmentation.py

# Test evaluation
python utils/evaluation.py

# Test monitor
python utils/byol_monitor.py
```

## Citation

If you use this code, please cite the BYOL paper:

```bibtex
@inproceedings{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please refer to `CLAUDE_BYOL.md` for detailed implementation guidance.
