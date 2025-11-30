# CLAUDE.md - BYOL 기반 웨이퍼 패턴 클러스터링 시스템

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Project Overview

### 1.1 프로젝트 목적
반도체 EDS(Electric Die Sorting) 웨이퍼맵의 불량 패턴을 학습하여:
1. **Latent vector 거리 ≈ 패턴 유사도**: 유사한 패턴일수록 latent space에서 가까워야 함
2. **Clustering 품질**: Latent vectors를 HDBSCAN으로 clustering했을 때 유사 패턴끼리 묶여야 함
3. **Rotation Invariance**: Edge-Left와 Edge-Right 등 회전된 동일 패턴은 같은 cluster로 분류

### 1.2 왜 BYOL인가?
기존 VQ-VAE + Contrastive Loss 대비 장점:
- **Negative sample 불필요**: 유사 패턴이 negative로 잘못 학습되는 문제 회피
- **연속적 latent space**: VQ의 discrete codebook 한계 극복
- **Batch size 유연성**: 12GB VRAM에서 batch_size=256 안정적
- **D4 augmentation 친화적**: 회전/반전이 자연스럽게 "같은 패턴" 정의

### 1.3 최종 목표
- **Retrieval Precision@5 ≥ 70%**: Query 패턴과 유사한 top-5 결과
- **Silhouette Score ≥ 0.5**: HDBSCAN clustering 품질
- **Rotation Invariance ≥ 95%**: 동일 패턴의 8가지 D4 변환이 같은 cluster

---

## 2. Technical Constraints

### 2.1 하드웨어 제약
```
GPU: NVIDIA GPU with 12GB VRAM (maximum)
PyTorch: 1.4.0 (고정, 업그레이드 불가)
CUDA: PyTorch 1.4.0 호환 버전
```

### 2.2 PyTorch 1.4.0 호환성 주의사항
```python
# ❌ 사용 불가 (1.4.0 이후 추가됨)
torch.quantile()           # → torch.kthvalue() 사용
F.normalize(..., dim=1)    # 일부 버전 문제 → 수동 구현
nn.SyncBatchNorm           # → nn.BatchNorm2d 사용
torch.cuda.amp             # → 없음, 수동 half precision

# ✅ 사용 가능
torch.nn.functional.cosine_similarity()
torch.optim.AdamW (존재하지만 확인 필요)
F.interpolate()
torch.distributed (기본 기능)
```

### 2.3 데이터 제약
```
총 데이터: ~61,000 wafer maps (4개 제품군 합산)
Train/Valid split: 80/20
Label: 없음 (unsupervised)
Wafer map 크기: 가변 → 128×128로 resize
값 범위: 0 (non-wafer/good), 1 (bad chip/defect)
```

---

## 3. BYOL Architecture

### 3.1 전체 구조
```
                    ┌─────────────────────────────────────┐
                    │           Input Image x             │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │  Augment t   │               │  Augment t'  │
            │   (view 1)   │               │   (view 2)   │
            └──────────────┘               └──────────────┘
                    │                               │
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │   Online     │               │   Target     │
            │   Encoder    │               │   Encoder    │
            │   f_θ        │               │   f_ξ        │
            └──────────────┘               └──────────────┘
                    │                               │
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │   Online     │               │   Target     │
            │  Projector   │               │  Projector   │
            │   g_θ        │               │   g_ξ        │
            └──────────────┘               └──────────────┘
                    │                               │
                    ▼                               │
            ┌──────────────┐                       │
            │  Predictor   │                       │
            │   q_θ        │                       │
            └──────────────┘                       │
                    │                               │
                    ▼                               ▼
                   z_θ ─────── L2 Loss ─────────► z'_ξ
                              (+ symmetric)
                    
            Target network: ξ ← τ·ξ + (1-τ)·θ  (EMA update)
```

### 3.2 Component 상세

#### 3.2.1 Encoder (f_θ, f_ξ)
```python
# ResNet-18 기반 경량화 버전 (12GB VRAM 고려)
class WaferEncoder(nn.Module):
    """
    Input: (B, 1, 128, 128) - grayscale wafer map
    Output: (B, 512) - feature vector
    
    Architecture:
    - Conv stem: 1 → 64 channels
    - ResNet blocks: 64 → 128 → 256 → 512
    - Global Average Pooling
    - Output: 512-dim feature
    """
    
    # Optional components (from VQ-VAE):
    # - RadialPositionalEncoder: 웨이퍼 중심 거리 정보
    # - SelfAttention2D: 전역 패턴 관계 학습
```

#### 3.2.2 Projector (g_θ, g_ξ)
```python
class Projector(nn.Module):
    """
    Input: (B, 512)
    Output: (B, 256)
    
    Architecture:
    - Linear(512, 1024)
    - BatchNorm1d(1024)
    - ReLU
    - Linear(1024, 256)
    """
```

#### 3.2.3 Predictor (q_θ) - Online only
```python
class Predictor(nn.Module):
    """
    Input: (B, 256)
    Output: (B, 256)
    
    Architecture:
    - Linear(256, 1024)
    - BatchNorm1d(1024)
    - ReLU
    - Linear(1024, 256)
    
    Note: Target network에는 없음!
          이것이 collapse 방지의 핵심
    """
```

### 3.3 Loss Function
```python
def byol_loss(p, z):
    """
    Args:
        p: predictor output from online network (B, 256)
        z: projector output from target network (B, 256)
    
    Returns:
        loss: mean squared error after L2 normalization
    
    Formula:
        L = 2 - 2 * <p, z> / (||p|| * ||z||)
        = 2 * (1 - cosine_similarity(p, z))
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

# Symmetric loss (both views)
loss = byol_loss(p1, z2.detach()) + byol_loss(p2, z1.detach())
```

### 3.4 EMA Update
```python
def update_target_network(online_net, target_net, tau=0.996):
    """
    Exponential Moving Average update
    
    Args:
        tau: momentum coefficient (0.996 ~ 0.999)
             높을수록 target이 천천히 변함
    
    Update rule:
        ξ ← τ·ξ + (1-τ)·θ
    """
    for online_params, target_params in zip(
        online_net.parameters(), target_net.parameters()
    ):
        target_params.data = tau * target_params.data + (1 - tau) * online_params.data
```

---

## 4. Data Augmentation Strategy

### 4.1 D4 Dihedral Group (Rotation Invariance 핵심)
```python
class D4Augmentation:
    """
    8가지 대칭 변환 (웨이퍼 패턴에 필수!)
    
    Transformations:
    0: Identity
    1: 90° rotation
    2: 180° rotation  
    3: 270° rotation
    4: Horizontal flip
    5: Vertical flip
    6: Diagonal flip (transpose)
    7: Anti-diagonal flip
    
    Usage:
        view1 = random_d4_transform(x)
        view2 = random_d4_transform(x)  # 다른 변환
    """
```

### 4.2 추가 Augmentation (Optional)
```python
class WaferAugmentation:
    """
    웨이퍼 특화 augmentation
    
    Spatial:
    - Random crop & resize (0.8 ~ 1.0 scale)
    - Small rotation (±5°) - D4 외 미세 회전
    
    Intensity:
    - Gaussian noise (σ=0.02)
    - Random erasing (defect 일부 제거, p=0.1)
    
    NOT recommended:
    - Color jittering (grayscale이므로)
    - Heavy blur (패턴 경계 중요)
    """
```

### 4.3 Augmentation 구현
```python
def get_byol_augmentation(view_type='strong'):
    """
    Args:
        view_type: 'strong' or 'weak'
    
    Returns:
        augmentation function
    
    Implementation (PyTorch 1.4.0 compatible):
    - torch.rot90() for rotations
    - torch.flip() for flips
    - F.interpolate() for resize
    - Manual implementation for others
    """
```

---

## 5. Training Pipeline

### 5.1 Training Loop
```python
def train_byol_epoch(model, dataloader, optimizer, device, tau):
    """
    1 epoch 학습
    
    Steps per batch:
    1. Load batch x
    2. Generate two views: v1 = aug(x), v2 = aug(x)
    3. Online forward: p1, p2 = online_network(v1, v2)
    4. Target forward: z1, z2 = target_network(v1, v2)
    5. Compute symmetric loss
    6. Backward & optimize online network only
    7. EMA update target network
    
    Returns:
        avg_loss, learning_rate
    """
```

### 5.2 Hyperparameters
```python
# Model
encoder_dim = 512
projector_dim = 256
predictor_hidden = 1024

# Training
batch_size = 256          # 12GB VRAM 최대
epochs = 100              # BYOL은 긴 학습 필요
base_lr = 0.0001          # AdamW 기준
weight_decay = 0.01

# BYOL specific
tau_base = 0.996          # EMA momentum 시작값
tau_max = 0.999           # EMA momentum 최종값 (cosine schedule)

# Augmentation
use_d4 = True             # 필수!
use_crop = True
crop_scale = (0.8, 1.0)
use_noise = True
noise_std = 0.02

# Learning rate schedule
scheduler = CosineAnnealingWarmUpRestarts(
    T_0=30,
    T_mult=1,
    eta_max=0.001,
    T_up=5,
    gamma=0.9
)
```

### 5.3 Tau (EMA Momentum) Schedule
```python
def get_tau(epoch, total_epochs, tau_base=0.996, tau_max=0.999):
    """
    Cosine schedule for tau
    
    - 초기: tau 낮음 → target이 빠르게 업데이트
    - 후기: tau 높음 → target이 안정적
    
    Formula:
        tau = tau_max - (tau_max - tau_base) * (1 + cos(π * epoch / total_epochs)) / 2
    """
    return tau_max - (tau_max - tau_base) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
```

---

## 6. Evaluation & Monitoring

### 6.1 Evaluation Metrics

#### 6.1.1 Retrieval Quality
```python
def evaluate_retrieval(model, dataloader, device, k=5):
    """
    Query 패턴과 유사한 top-k 결과 평가
    
    Metrics:
    - Precision@k: top-k 중 실제 유사 패턴 비율
    - Recall@k: 전체 유사 패턴 중 top-k에 포함된 비율
    - MRR (Mean Reciprocal Rank)
    
    Note: Ground truth 없으므로 pattern similarity 기반 평가
    - IoU > 0.7 → 같은 패턴으로 간주
    - D4 변환 → 같은 패턴으로 간주
    """
```

#### 6.1.2 Clustering Quality
```python
def evaluate_clustering(features, min_cluster_size=50):
    """
    HDBSCAN clustering 후 품질 평가
    
    Metrics:
    - Silhouette Score: cluster 분리도 (-1 ~ 1, 높을수록 좋음)
    - Calinski-Harabasz Index: cluster 밀집도
    - Davies-Bouldin Index: cluster 간 거리 (낮을수록 좋음)
    - Noise ratio: HDBSCAN -1 label 비율 (낮을수록 좋음)
    - Number of clusters: 발견된 패턴 유형 수
    """
```

#### 6.1.3 Rotation Invariance
```python
def evaluate_rotation_invariance(model, test_samples, device):
    """
    D4 변환에 대한 불변성 평가
    
    Process:
    1. 샘플 선택
    2. 8가지 D4 변환 적용
    3. 모든 변환의 embedding 추출
    4. 같은 샘플의 변환들이 얼마나 가까운지 측정
    
    Metrics:
    - Mean cosine similarity within D4 group
    - Variance of embeddings within D4 group
    - Cluster consistency (같은 cluster 할당 비율)
    """
```

### 6.2 Monitoring System
```python
class BYOLMonitor:
    """
    학습 중 모니터링
    
    Per-epoch tracking:
    - Loss (train/valid)
    - Learning rate
    - Tau (EMA momentum)
    - Feature statistics (mean, std, collapse 감지)
    - Embedding space uniformity
    
    Periodic evaluation (every N epochs):
    - Retrieval metrics
    - Clustering metrics
    - Rotation invariance
    - t-SNE/UMAP visualization
    
    Collapse detection:
    - Feature std < threshold → warning
    - All features similar → critical
    """
```

### 6.3 Visualization
```python
def visualize_latent_space(features, labels=None, method='tsne'):
    """
    Latent space 시각화
    
    Methods:
    - t-SNE: 로컬 구조 보존
    - UMAP: 글로벌 구조 보존 (권장)
    
    Color coding:
    - Cluster assignment (HDBSCAN)
    - Pattern type (if available)
    - Defect density
    """

def visualize_similar_patterns(query_idx, model, dataloader, k=10):
    """
    Query 패턴과 top-k 유사 패턴 시각화
    
    Layout:
    - Left: Query wafer map
    - Right: Top-k similar maps with similarity scores
    """
```

---

## 7. File Structure

```
project/
├── CLAUDE.md                    # 이 파일
├── main_byol_training.py        # 메인 학습 스크립트
│
├── models/
│   ├── __init__.py
│   ├── byol.py                  # BYOL 전체 모델
│   ├── encoder.py               # WaferEncoder (ResNet-18 based)
│   ├── projector.py             # Projector & Predictor
│   └── models_vqvae_dual.py     # (기존) 비교용
│
├── utils/
│   ├── __init__.py
│   ├── augmentation.py          # D4 + Wafer augmentations
│   ├── dataloader_utils.py      # (기존) 데이터 로딩
│   ├── train_byol.py            # BYOL 학습 함수
│   ├── byol_monitor.py          # 모니터링 시스템
│   ├── evaluation.py            # 평가 함수들
│   ├── checkpoint_utils.py      # (기존) 체크포인트 관리
│   └── density_aware_filter.py  # (기존) 전처리 필터
│
├── functions/
│   ├── __init__.py
│   └── functions.py             # (기존) LR scheduler 등
│
└── experiments/
    ├── compare_vqvae_byol.py    # VQ-VAE vs BYOL 비교
    └── ablation_study.py        # Augmentation 효과 분석
```

---

## 8. Implementation Plan

### Phase 1: Core Implementation (Week 1)

#### Step 1.1: Encoder 구현
```python
# models/encoder.py

class WaferEncoder(nn.Module):
    """
    ResNet-18 기반 경량 encoder
    
    특이사항:
    - Input: 1 channel (grayscale)
    - Optional: RadialPositionalEncoder (from VQ-VAE)
    - Optional: SelfAttention2D (from VQ-VAE)
    - Output: 512-dim feature
    
    Memory estimation:
    - Parameters: ~11M
    - Forward (B=256): ~2GB
    """
    
    def __init__(self, 
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128)):
        ...
```

#### Step 1.2: Projector & Predictor 구현
```python
# models/projector.py

class MLP(nn.Module):
    """Projector와 Predictor 공용 MLP"""
    
class Projector(MLP):
    """512 → 256, BN 포함"""
    
class Predictor(MLP):
    """256 → 256, Online network 전용"""
```

#### Step 1.3: BYOL 모델 통합
```python
# models/byol.py

class BYOL(nn.Module):
    """
    전체 BYOL 모델
    
    Components:
    - online_encoder
    - online_projector
    - online_predictor
    - target_encoder (no grad)
    - target_projector (no grad)
    
    Methods:
    - forward(x): 학습용 (loss 계산)
    - encode(x): 추론용 (feature 추출)
    - update_target(): EMA update
    """
```

### Phase 2: Training Pipeline (Week 1-2)

#### Step 2.1: Augmentation 구현
```python
# utils/augmentation.py

class D4Transform:
    """D4 dihedral group 변환"""
    
class WaferAugmentation:
    """웨이퍼 특화 augmentation"""
    
class BYOLAugmentation:
    """BYOL용 two-view augmentation"""
```

#### Step 2.2: Training Loop 구현
```python
# utils/train_byol.py

def train_byol_epoch(model, dataloader, optimizer, device, tau):
    """1 epoch 학습"""
    
def validate_byol_epoch(model, dataloader, device):
    """Validation loss 계산"""
```

#### Step 2.3: 메인 학습 스크립트
```python
# main_byol_training.py

def train_byol_wafer(config):
    """전체 학습 파이프라인"""
    
def main():
    """Entry point"""
```

### Phase 3: Evaluation & Monitoring (Week 2)

#### Step 3.1: 모니터링 시스템
```python
# utils/byol_monitor.py

class BYOLMonitor:
    """학습 중 모니터링"""
```

#### Step 3.2: 평가 함수
```python
# utils/evaluation.py

def evaluate_retrieval(...)
def evaluate_clustering(...)
def evaluate_rotation_invariance(...)
```

### Phase 4: Experiments & Comparison (Week 3)

#### Step 4.1: VQ-VAE vs BYOL 비교
```python
# experiments/compare_vqvae_byol.py

def compare_models():
    """
    동일 데이터셋에서:
    1. VQ-VAE 모델 로드
    2. BYOL 모델 로드
    3. 동일 metrics로 비교
    """
```

#### Step 4.2: Ablation Study
```python
# experiments/ablation_study.py

def ablation_augmentation():
    """
    Augmentation 효과 분석:
    - D4 only
    - D4 + crop
    - D4 + crop + noise
    """
```

---

## 9. Key Implementation Details

### 9.1 Collapse 방지
```python
# BYOL의 collapse 방지 메커니즘

# 1. Predictor의 존재 (가장 중요!)
# - Online network에만 있음
# - 비대칭성이 collapse 방지

# 2. EMA update (천천히)
# - tau = 0.996 ~ 0.999
# - Target이 너무 빨리 변하면 collapse

# 3. BatchNorm의 역할
# - Feature 정규화
# - Batch 내 다양성 유지

# Collapse 감지
def detect_collapse(features):
    """
    Collapse 징후:
    1. Feature std < 0.01
    2. All cosine similarities > 0.99
    3. Loss가 0에 수렴
    """
    std = features.std(dim=0).mean()
    if std < 0.01:
        return True, f"Low std: {std:.4f}"
    return False, "OK"
```

### 9.2 메모리 최적화
```python
# 12GB VRAM 제약 하에서 최적화

# 1. Gradient checkpointing (필요시)
# - Encoder 중간 activation 재계산
# - 메모리 ↓, 속도 ↓

# 2. Mixed precision (PyTorch 1.4.0에서 수동)
# - 불가능하면 batch_size 줄이기

# 3. Target network no_grad
with torch.no_grad():
    target_output = target_network(x)

# 4. Batch size 조정
# - 256: 안정적 (권장)
# - 128: 메모리 부족시
# - 512: 여유 있으면 (학습 안정성 ↑)

# Memory estimation (batch_size=256):
# - Encoder: ~2GB
# - Projector/Predictor: ~0.5GB
# - Gradients: ~3GB
# - Data: ~1GB
# - Total: ~8GB (여유 있음)
```

### 9.3 기존 코드 재사용
```python
# VQ-VAE에서 가져올 컴포넌트

# 1. RadialPositionalEncoder (models/models_vqvae_ema.py)
from models.models_vqvae_ema import RadialPositionalEncoder

# 2. SelfAttention2D (models/models_vqvae_ema.py)
from models.models_vqvae_ema import SelfAttention2D

# 3. 데이터 로딩 (utils/dataloader_utils.py)
from utils.dataloader_utils import prepare_clean_data, create_dataloaders

# 4. 전처리 필터 (utils/density_aware_filter.py)
from utils.density_aware_filter import DensityAwareWaferMapFilter

# 5. LR Scheduler (functions/functions.py)
from functions.functions import CosineAnnealingWarmUpRestarts

# 6. Checkpoint 관리 (utils/checkpoint_utils.py)
from utils.checkpoint_utils import load_checkpoint, check_training_status
```

---

## 10. Expected Results

### 10.1 학습 곡선 예상
```
Epoch 1-10:   Loss 급격히 감소, feature collapse 주의
Epoch 10-30:  Loss 안정화, cluster 형성 시작
Epoch 30-60:  Retrieval 성능 향상
Epoch 60-100: 수렴, fine-grained separation
```

### 10.2 성능 목표
```
Metric                          Target      VQ-VAE (현재)
─────────────────────────────────────────────────────────
Retrieval Precision@5           ≥ 70%       ~3%
Silhouette Score                ≥ 0.5       ~0.4
Rotation Invariance (D4)        ≥ 95%       측정 필요
Cluster Count                   10-50       -
Noise Ratio (HDBSCAN)           < 20%       -
```

### 10.3 Downstream 활용
```python
# 학습 완료 후 활용

# 1. 유사 패턴 검색
similar_indices = search_similar(query_embedding, all_embeddings, k=10)

# 2. 패턴 클러스터링
cluster_labels = hdbscan.fit_predict(all_embeddings)

# 3. 새 패턴 감지
is_novel = detect_novel_pattern(new_embedding, cluster_centers)

# 4. 시계열 트렌드 분석
trend = analyze_pattern_trend(embeddings_by_date)
```

---

## 11. Troubleshooting Guide

### 11.1 Collapse 발생시
```
증상: Loss → 0, 모든 feature 동일
원인: Predictor 없음, tau 너무 낮음, BN 문제

해결:
1. Predictor 확인 (필수!)
2. tau 높이기 (0.999로)
3. BatchNorm 확인 (eval 모드 아닌지)
4. Learning rate 낮추기
```

### 11.2 학습 불안정
```
증상: Loss 진동, NaN 발생
원인: LR 너무 높음, gradient explosion

해결:
1. Learning rate 낮추기 (1e-4 → 1e-5)
2. Gradient clipping 추가
3. Warmup 기간 늘리기
```

### 11.3 Clustering 품질 낮음
```
증상: Silhouette < 0.3, noise 많음
원인: Augmentation 부족, 학습 부족

해결:
1. D4 augmentation 확인 (필수!)
2. 더 오래 학습 (100 epoch+)
3. HDBSCAN min_cluster_size 조정
4. Projector 출력 차원 늘리기
```

### 11.4 메모리 부족
```
증상: CUDA out of memory
원인: Batch size 너무 큼

해결:
1. Batch size 줄이기 (256 → 128)
2. Gradient accumulation 사용
3. 불필요한 텐서 del + torch.cuda.empty_cache()
```

---

## 12. Commands Reference

### 12.1 학습 시작
```bash
# 새 학습
python main_byol_training.py

# 체크포인트에서 재개
python main_byol_training.py --resume path/to/checkpoint.pth

# 특정 config로 학습
python main_byol_training.py --config configs/byol_v1.yaml
```

### 12.2 평가
```bash
# 전체 평가
python -c "from utils.evaluation import evaluate_all; evaluate_all('path/to/model.pth')"

# 개별 평가
python -c "from utils.evaluation import evaluate_retrieval; evaluate_retrieval(...)"
```

### 12.3 시각화
```bash
# t-SNE 시각화
python -c "from utils.visualization import plot_tsne; plot_tsne('path/to/model.pth')"

# 유사 패턴 시각화
python -c "from utils.visualization import show_similar; show_similar(query_idx=100)"
```

---

## 13. References

### 논문
- BYOL: Bootstrap Your Own Latent (Grill et al., 2020)
- MoCo v2: Improved Baselines with Momentum Contrastive Learning (Chen et al., 2020)
- SimSiam: Exploring Simple Siamese Representation Learning (Chen & He, 2021)

### 기존 코드
- VQ-VAE 구현: `models/models_vqvae_dual.py`
- 데이터 로딩: `utils/dataloader_utils.py`
- 전처리: `utils/density_aware_filter.py`

---

## 14. Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2024-XX-XX | 1.0 | Initial BYOL implementation plan |

---

**Note**: 이 문서는 Claude Code가 BYOL 구현 시 참조하는 가이드입니다. 구현 진행에 따라 업데이트됩니다.
