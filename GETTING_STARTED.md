# Getting Started with BYOL Wafer Pattern Clustering

이 가이드는 BYOL 시스템을 빠르게 시작하는 방법을 설명합니다.

## 1. 환경 설정

### 1.1 PyTorch 1.4.0 설치 (필수!)

```bash
# CUDA 10.1 기준
pip install torch==1.4.0 torchvision==0.5.0

# CPU 전용
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 1.2 추가 라이브러리 설치

```bash
pip install -r requirements.txt
```

**주의:** NumPy 버전 경고가 발생할 수 있지만, 코드는 정상 작동합니다.

## 2. 프로젝트 구조 확인

```
BYOL/
├── models/              # BYOL 모델 컴포넌트
│   ├── encoder.py      # WaferEncoder (11M params)
│   ├── projector.py    # Projector & Predictor
│   └── byol.py         # 전체 BYOL 모델
│
├── utils/              # 유틸리티 함수
│   ├── augmentation.py # D4 + 웨이퍼 augmentation
│   ├── train_byol.py   # 학습 함수
│   ├── evaluation.py   # 평가 함수
│   └── byol_monitor.py # 모니터링 시스템
│
└── main_byol_training.py  # 메인 학습 스크립트
```

## 3. 컴포넌트 테스트

각 모듈이 제대로 작동하는지 확인하세요:

```bash
# Encoder 테스트
python models/encoder.py
# 출력: Total parameters: 11,548,769

# Projector & Predictor 테스트
python models/projector.py

# BYOL 모델 테스트
python models/byol.py
# 출력: Total parameters: ~25M

# Augmentation 테스트
python utils/augmentation.py

# Evaluation 테스트
python utils/evaluation.py

# Monitor 테스트
python utils/byol_monitor.py
```

## 4. 데이터 준비

### 4.1 데이터 포맷

웨이퍼 맵은 다음 형식이어야 합니다:
- Shape: `(N, 1, H, W)` - 배치, 채널, 높이, 너비
- Values: `0` (정상/비웨이퍼), `1` (불량)
- Size: 128×128로 resize 권장

### 4.2 데이터 로더 구현

`main_byol_training.py`의 `create_dummy_dataloader` 함수를 실제 데이터로 교체:

```python
def create_real_dataloader(data_path, batch_size=256):
    """
    실제 웨이퍼 데이터 로딩

    Args:
        data_path: 데이터 디렉토리 경로
        batch_size: 배치 크기

    Returns:
        train_loader, val_loader
    """
    # TODO: 실제 데이터 로딩 로직 구현
    # 예시:
    # 1. 웨이퍼 맵 파일 읽기 (pkl, npy, csv 등)
    # 2. 0-1 정규화
    # 3. 128x128로 resize
    # 4. torch.Tensor로 변환
    # 5. DataLoader 생성

    pass
```

## 5. 학습 시작

### 5.1 기본 학습

```bash
python main_byol_training.py
```

기본 설정:
- Epochs: 100
- Batch size: 256 (12GB VRAM 기준)
- Learning rate: 0.0001
- Augmentation: strong (D4 + crop + noise)

### 5.2 커스텀 파라미터

```bash
# Epoch 수 조정
python main_byol_training.py --epochs 150

# Batch size 조정 (VRAM 부족시)
python main_byol_training.py --batch_size 128

# Learning rate 조정
python main_byol_training.py --lr 0.0005

# 복합 사용
python main_byol_training.py --epochs 150 --batch_size 128 --lr 0.0005
```

### 5.3 체크포인트에서 재개

```bash
python main_byol_training.py --resume checkpoints/checkpoint_epoch_50.pth
```

## 6. 학습 모니터링

### 6.1 실시간 모니터링

터미널에서 실시간으로 다음 정보가 출력됩니다:
- Train/Validation Loss
- Learning Rate
- Tau (EMA momentum)
- Collapse Detection (Feature std, Cosine similarity)

### 6.2 로그 파일

`logs/` 디렉토리에 다음 파일이 생성됩니다:

```
logs/
├── training_curves.png           # Loss, LR, Tau 그래프
├── evaluation_metrics.png        # 평가 지표 그래프
├── latent_space_epoch_*.png     # t-SNE 시각화
├── final_latent_space.png       # 최종 latent space
└── history.json                 # 전체 학습 기록
```

### 6.3 체크포인트

`checkpoints/` 디렉토리에 저장됩니다:

```
checkpoints/
├── best_model.pth               # Best validation loss
├── final_model.pth              # 최종 모델
└── checkpoint_epoch_*.pth       # 주기적 체크포인트
```

## 7. 평가

### 7.1 학습 중 평가

매 `eval_frequency` epoch마다 자동으로 평가:
- Retrieval metrics (Precision@5, avg distance)
- Clustering metrics (Silhouette, CH, DB)
- Rotation invariance (D4 cosine similarity)

### 7.2 최종 평가

학습 완료 후 자동으로 comprehensive evaluation 수행:

```python
from models.byol import BYOL
from utils.evaluation import evaluate_all, print_evaluation_results

# 모델 로드
model = BYOL(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 평가
metrics, labels = evaluate_all(model, val_loader, device)
print_evaluation_results(metrics)
```

### 7.3 Latent Vector 추출

```python
from utils.train_byol import extract_features

# Feature 추출
features, _ = extract_features(model, dataloader, device, use_target=True)

# Clustering
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=50)
labels = clusterer.fit_predict(features.numpy())

# 유사 패턴 검색
from utils.evaluation import compute_pairwise_distances
distances = compute_pairwise_distances(features, metric='euclidean')
nearest = distances.argsort(dim=1)[:, 1:6]  # Top-5 nearest
```

## 8. 문제 해결

### 8.1 Collapse 발생

**증상:** Loss가 0에 수렴, 모든 feature가 동일

**해결책:**
```python
# config 수정
config['tau_base'] = 0.999  # tau 증가
config['base_lr'] = 0.00005  # LR 감소
config['T_up'] = 10          # Warmup 증가
```

### 8.2 CUDA Out of Memory

**해결책:**
```bash
# Batch size 줄이기
python main_byol_training.py --batch_size 128

# 또는 gradient accumulation 사용 (코드 수정 필요)
```

### 8.3 낮은 Clustering 품질

**해결책:**
1. D4 augmentation 확인 (`use_d4=True` 필수!)
2. 더 오래 학습 (100+ epochs)
3. HDBSCAN `min_cluster_size` 조정
4. Projector dimension 증가

## 9. 성능 목표

| Metric | Target |
|--------|--------|
| Retrieval Precision@5 | ≥ 70% |
| Silhouette Score | ≥ 0.5 |
| Rotation Invariance | ≥ 95% |
| Cluster Count | 10-50 |
| Noise Ratio | < 20% |

## 10. 다음 단계

1. **실제 데이터로 학습**
   - `create_dummy_dataloader` → `create_real_dataloader` 교체
   - 데이터 전처리 확인 (0-1 정규화, resize)

2. **Hyperparameter Tuning**
   - Learning rate schedule 조정
   - Augmentation strength 조정
   - Tau schedule 조정

3. **비교 실험**
   - VQ-VAE vs BYOL 성능 비교
   - Ablation study (D4 효과, attention 효과 등)

4. **Production 배포**
   - Feature extraction 최적화
   - 실시간 유사 패턴 검색 시스템 구축
   - 패턴 트렌드 분석

## 참고 문서

- [CLAUDE_BYOL.md](CLAUDE_BYOL.md): 상세 구현 가이드
- [README.md](README.md): 프로젝트 개요
- BYOL 논문: Grill et al., 2020

## 질문/이슈

문제가 발생하면 다음을 확인하세요:
1. PyTorch 1.4.0 버전 확인
2. CUDA 호환성 확인
3. 데이터 포맷 확인
4. 로그 파일의 collapse detection 확인
