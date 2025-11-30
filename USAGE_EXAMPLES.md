# BYOL Training Usage Examples

## 기본 사용법

### 1. Dummy 데이터로 테스트 (데이터 없이)

```bash
# 기본 설정으로 실행 (dummy data 사용)
python main_byol_training.py

# epoch과 batch size 조정
python main_byol_training.py --epochs 50 --batch_size 128
```

### 2. 실제 웨이퍼 데이터로 학습

```bash
# 단일 데이터 파일 사용
python main_byol_training.py --data_path path/to/wafer_data.npz

# 데이터 이름 지정
python main_byol_training.py --data_path data.npz --data_name "Product_A"

# 필터링 적용
python main_byol_training.py --data_path data.npz --use_filter

# 밀도 기반 적응형 필터링 (권장!)
python main_byol_training.py --data_path data.npz --use_filter --use_density_aware
```

### 3. 고급 설정

```bash
# 전체 설정 예시
python main_byol_training.py \
    --data_path wafer_data.npz \
    --data_name "Product_ABC" \
    --use_filter \
    --use_density_aware \
    --epochs 150 \
    --batch_size 256 \
    --lr 0.0001

# 체크포인트에서 재개
python main_byol_training.py \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --epochs 200
```

## Python 코드에서 사용

### 방법 1: 단일 데이터 파일

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_byol_training import get_default_config, train_byol_wafer

# Config 설정
config = get_default_config()

# 데이터 경로 설정
config['data_configs'] = [
    {"path": "data/product_a.npz", "name": "Product_A"}
]

# 필터링 설정
config['use_filter'] = True
config['use_density_aware'] = True  # 밀도 기반 적응형 필터 (권장)
config['use_region_aware'] = False

# 학습 설정
config['epochs'] = 100
config['batch_size'] = 256
config['base_lr'] = 0.0001

# 학습 시작
train_byol_wafer(config)
```

### 방법 2: 여러 데이터 파일 합치기

```python
from main_byol_training import get_default_config, train_byol_wafer

config = get_default_config()

# 여러 데이터 파일 설정
config['data_configs'] = [
    {"path": "data/product_a.npz", "name": "Product_A"},
    {"path": "data/product_b.npz", "name": "Product_B"},
    {"path": "data/product_c.npz", "name": "Product_C"}
]

config['use_filter'] = True
config['use_density_aware'] = True
config['epochs'] = 150

train_byol_wafer(config)
```

### 방법 3: 직접 데이터 로드하여 사용

```python
import numpy as np
from utils.dataloader_utils import prepare_clean_data, create_dataloaders

# 1. 데이터 로드 및 정리
data_configs = [
    {"path": "data.npz", "name": "my_data"}
]

wafer_maps, labels, info = prepare_clean_data(
    data_configs,
    use_filter=True,
    filter_params=None,
    use_density_aware=True,
    use_region_aware=False
)

print(f"Loaded {len(wafer_maps)} wafer maps")

# 2. DataLoader 생성
train_loader, val_loader = create_dataloaders(
    wafer_maps=wafer_maps,
    labels=labels,
    batch_size=256,
    target_size=(128, 128),
    test_size=0.2,
    use_filter=False,  # 이미 필터링됨
    use_augmentation=False  # BYOL이 직접 augmentation 적용
)

# 3. 학습 (기존 코드 사용)
# ... BYOL 모델 생성 및 학습
```

## 데이터 형식

### NPZ 파일 형식

웨이퍼 데이터는 다음 형식의 `.npz` 파일이어야 합니다:

```python
import numpy as np

# 데이터 준비
wafer_maps = [...]  # List of 2D numpy arrays (H, W)
labels = [...]      # List of labels (str or int)

# 저장
np.savez(
    'wafer_data.npz',
    maps=wafer_maps,
    ids=labels  # 또는 'labels'
)
```

### 로드 후 자동 처리

`prepare_clean_data` 함수가 자동으로:
1. ✅ Object array → float32 변환
2. ✅ NaN, Inf 처리
3. ✅ 0-1 정규화
4. ✅ 필터링 (옵션)
5. ✅ 검증 (shape, dtype)

## 출력 파일

### 체크포인트 (checkpoints/)

```
checkpoints/
├── best_model.pth              # Best validation loss
├── final_model.pth             # 최종 모델
└── checkpoint_epoch_*.pth      # 주기적 체크포인트
```

### 로그 (logs/)

```
logs/
├── training_curves.png         # Loss, LR, Tau 그래프
├── evaluation_metrics.png      # 평가 지표
├── latent_space_epoch_*.png   # t-SNE 시각화
├── final_latent_space.png     # 최종 latent space
└── history.json               # 전체 학습 기록
```

## 학습 모니터링

### 실시간 출력

학습 중 다음 정보가 출력됩니다:

```
Epoch 1 [Train] [10/50] Loss: 3.2145, Tau: 0.9960
Epoch 1 [Train] [20/50] Loss: 3.1892, Tau: 0.9960
...
============================================================
Epoch 1 Summary
============================================================
Train Loss:      3.1234
Val Loss:        3.0987
Learning Rate:   1.000e-04
Tau (EMA):       0.996000
Time:            45.23s

Collapse Detection:
  Feature Std:   0.234567
  Avg Cos Sim:   0.123456
  Collapsed:     False
============================================================
```

### 주기적 평가 (eval_frequency마다)

```
Performing evaluation at epoch 10...
Extracting features...
Evaluating retrieval...
Evaluating clustering...
Evaluating rotation invariance...

============================================================
EVALUATION RESULTS
============================================================

Retrieval Metrics:
  avg_distance_top_k              : 0.1234
  std_distance_top_k              : 0.0567

Clustering Metrics:
  n_clusters                      : 23
  noise_ratio                     : 0.1234
  silhouette                      : 0.5678
  calinski_harabasz               : 1234.56
  davies_bouldin                  : 0.8765

Rotation Invariance Metrics:
  avg_cosine_similarity           : 0.9567
  std_cosine_similarity           : 0.0123
  ...
```

## 문제 해결

### 1. 데이터 로딩 실패

```bash
# 에러: Failed to load data, falling back to dummy data
# 해결: 데이터 경로 확인
ls -la path/to/wafer_data.npz

# NPZ 파일 내용 확인
python -c "import numpy as np; d = np.load('data.npz', allow_pickle=True); print(d.files)"
```

### 2. CUDA Out of Memory

```bash
# Batch size 줄이기
python main_byol_training.py --data_path data.npz --batch_size 128
```

### 3. 필터링 너무 강함

```python
# Python 코드에서 필터 파라미터 조정
config['use_filter'] = True
config['use_density_aware'] = False  # 일반 필터 사용

# 또는 필터 비활성화
config['use_filter'] = False
```

### 4. Collapse 발생

```python
# Tau 증가, LR 감소
config['tau_base'] = 0.999
config['base_lr'] = 0.00005
config['T_up'] = 10  # Warmup 증가
```

## 성능 팁

### 최적 설정 (12GB VRAM 기준)

```python
config = {
    'batch_size': 256,          # 최대
    'wafer_size': 128,          # 128x128 권장
    'use_density_aware': True,  # 적응형 필터 사용
    'base_lr': 0.0001,
    'tau_base': 0.996,
    'tau_max': 0.999,
    'augmentation_type': 'strong',
    'eval_frequency': 10,
}
```

### 빠른 실험 (테스트용)

```python
config = {
    'batch_size': 64,
    'epochs': 20,
    'eval_frequency': 5,
    'n_samples': 1000,  # dummy data인 경우
}
```

### Production 설정

```python
config = {
    'batch_size': 256,
    'epochs': 200,
    'eval_frequency': 10,
    'early_stopping_patience': 30,
    'use_filter': True,
    'use_density_aware': True,
    'save_frequency': 10,
}
```

## 추가 참고

- [README.md](README.md): 프로젝트 개요
- [GETTING_STARTED.md](GETTING_STARTED.md): 시작 가이드
- [CLAUDE_BYOL.md](CLAUDE_BYOL.md): 상세 구현 가이드
