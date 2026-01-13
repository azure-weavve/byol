# Multi-Category Wafer Map 적용 가이드

## 개요

기존 이진 분류(0=정상/non-wafer, 1=불량)에서 **다중 카테고리(1~10) 불량 분류**로 전환하기 위한 수정 가이드입니다.
다중 카테고리는 사

---

## 데이터 구조 변경

### 기존 (Binary)
```python
wafer_map: (H, W)
- 0 = non-wafer 또는 good chip
- 1 = bad chip

모델 입력: (B, 1, H, W)
```

### 변경 후 (Multi-category)
```python
wafer_map: (H, W)
- 0 = non-wafer 또는 good chip
- 1~10 = 불량 카테고리 1~10

데이터 생성 시 변환: (10, H, W) - 10개 채널
- channel[0] = 카테고리 1번 불량 위치 (0 또는 1)
- channel[1] = 카테고리 2번 불량 위치
- ...
- channel[9] = 카테고리 10번 불량 위치

모델 입력: (B, 10, H, W)
```

### 변환 로직 예시
```python
# 원본 데이터: (H, W) with values 0~10
original_map = np.array([[0, 0, 3, 5],
                         [0, 1, 0, 0],
                         [2, 0, 0, 10]])

# 변환: (10, H, W)
multi_channel = np.zeros((10, H, W), dtype=np.float32)

for cat in range(1, 11):
    multi_channel[cat-1] = (original_map == cat).astype(np.float32)

# 결과:
# channel[0] = [[0,0,0,0], [0,1,0,0], [0,0,0,0]]  # 카테고리 1
# channel[1] = [[0,0,0,0], [0,0,0,0], [1,0,0,0]]  # 카테고리 2
# channel[2] = [[0,0,1,0], [0,0,0,0], [0,0,0,0]]  # 카테고리 3
# ...
```

---

## 수정 필요 파일 목록

### 1. `models/encoder.py`
**목적**: 입력 채널 수를 1 → 10으로 변경

#### 수정 위치
```python
class WaferEncoder(nn.Module):
    def __init__(self,
                 input_channels=1,  # ← 이 부분을 10으로 변경
                 output_dim=512,
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128),
                 layers=[2, 2, 2, 2]):
```

#### 수정 방법
- **Option A**: 기본값 자체를 변경
  ```python
  input_channels=10,  # 기본값 변경
  ```

- **Option B**: 파라미터는 유지하고, 호출 시 명시
  ```python
  # 기본값은 그대로 두고
  # BYOL 생성 시 input_channels=10 전달
  ```

**권장**: Option B (하위 호환성 유지)

---

### 2. `models/byol.py`
**목적**: Encoder 생성 시 input_channels 파라미터 전달

#### 수정 위치
```python
class BYOL(nn.Module):
    def __init__(self,
                 encoder_dim=512,
                 projector_hidden=1024,
                 projector_out=256,
                 predictor_hidden=1024,
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128),
                 tau=0.996):
```

#### 추가 파라미터
```python
class BYOL(nn.Module):
    def __init__(self,
                 encoder_dim=512,
                 projector_hidden=1024,
                 projector_out=256,
                 predictor_hidden=1024,
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128),
                 tau=0.996,
                 input_channels=10):  # ← 추가
```

#### Encoder 생성 수정
```python
# === Online Network (trainable) ===
self.encoder_online = WaferEncoder(
    input_channels=input_channels,  # ← 추가
    output_dim=encoder_dim,
    use_radial_encoding=use_radial_encoding,
    use_attention=use_attention,
    wafer_size=wafer_size
)

# === Target Network (EMA, no grad) ===
self.encoder_target = WaferEncoder(
    input_channels=input_channels,  # ← 추가
    output_dim=encoder_dim,
    use_radial_encoding=use_radial_encoding,
    use_attention=use_attention,
    wafer_size=wafer_size
)
```

---

### 3. `main_byol_training.py`
**목적**: Config에 input_channels 추가 및 BYOL 모델 생성 시 전달

#### 수정 위치 1: Config
```python
def get_default_config(path):
    config = {
        # ... 기존 설정들 ...
        
        # Model
        'encoder_dim': 512,
        'projector_hidden': 1024,
        'projector_out': 256,
        'predictor_hidden': 1024,
        'use_radial_encoding': True,
        'use_attention': True,
        'input_channels': 10,  # ← 추가
        
        # ... 나머지 설정들 ...
    }
    return config
```

#### 수정 위치 2: 모델 생성
```python
def train_byol_wafer(config):
    # ...
    
    # Create model
    print("\nCreating BYOL model...")
    model = BYOL(
        encoder_dim=config['encoder_dim'],
        projector_hidden=config['projector_hidden'],
        projector_out=config['projector_out'],
        predictor_hidden=config['predictor_hidden'],
        use_radial_encoding=config['use_radial_encoding'],
        use_attention=config['use_attention'],
        wafer_size=(config['wafer_size'], config['wafer_size']),
        tau=config['tau_base'],
        input_channels=config['input_channels']  # ← 추가
    ).to(device)
    
    # ...
```

---

### 4. `utils/augmentation.py`
**목적**: D4 변환이 10채널 모두에 올바르게 적용되는지 확인

#### 확인 사항
현재 D4Transform은 **채널 차원을 유지**하면서 spatial transformation만 적용하므로, 별도 수정 불필요합니다.

```python
# 현재 구현
def apply(x, transform_id):
    # x: (C, H, W) 또는 (B, C, H, W)
    # transform은 마지막 2개 차원(H, W)에만 적용
    
    if transform_id == 1:
        return torch.rot90(x, k=1, dims=(-2, -1))  # ✅ 모든 채널에 동일 적용
```

**검증 방법**:
```python
# 테스트 코드
x = torch.randn(10, 128, 128)  # 10채널
transformed, _ = D4Transform.random_transform(x)
print(transformed.shape)  # (10, 128, 128) 확인
```

---

### 5. `utils/dataloader_utils.py` (수정 불필요)
**사용자가 직접 데이터 생성 시 채널 변환 수행**

데이터 생성 예시:
```python
# 사용자 코드 예시
import numpy as np

def convert_to_multichannel(wafer_maps):
    """
    wafer_maps: List of (H, W) arrays with values 0~10
    Returns: List of (10, H, W) arrays
    """
    multi_channel_maps = []
    
    for wm in wafer_maps:
        H, W = wm.shape
        channels = np.zeros((10, H, W), dtype=np.float32)
        
        for cat in range(1, 11):
            channels[cat-1] = (wm == cat).astype(np.float32)
        
        multi_channel_maps.append(channels)
    
    return multi_channel_maps

# 데이터 생성
original_maps = load_your_data()  # (N, H, W) with values 0~10
multi_channel_maps = convert_to_multichannel(original_maps)

# NPZ 저장
np.savez('wafer_data_multichannel.npz',
         maps=multi_channel_maps,
         ids=labels)
```

---

## 수정 순서

### Step 1: Encoder 수정
```bash
# models/encoder.py
# input_channels 파라미터 확인 (기본값은 유지)
```

### Step 2: BYOL 수정
```bash
# models/byol.py
# input_channels 파라미터 추가
# Encoder 생성 시 전달
```

### Step 3: Main script 수정
```bash
# main_byol_training.py
# Config에 input_channels=10 추가
# 모델 생성 시 전달
```

### Step 4: 테스트
```python
# 간단한 테스트 코드
import torch
from models.byol import BYOL

# 10채널 입력 테스트
model = BYOL(
    encoder_dim=512,
    input_channels=10
)

x = torch.randn(4, 10, 128, 128)  # (B=4, C=10, H=128, W=128)
features = model.encode(x, use_target=True)
print(features.shape)  # (4, 512) 확인
```

---

## 주의사항

### 1. 기존 모델 Weight 호환 불가
- 입력 채널이 1 → 10으로 변경되므로 **처음부터 재학습 필수**
- 기존 체크포인트 사용 불가

### 2. 메모리 사용량 증가
```
기존: (B, 1, H, W) = (256, 1, 128, 128) ≈ 4MB per batch
변경: (B, 10, H, W) = (256, 10, 128, 128) ≈ 40MB per batch
```
- Batch size 조정 필요할 수 있음 (256 → 128 등)

### 3. RadialPositionalEncoder 주의
```python
# models/encoder.py 내부
if use_radial_encoding:
    self.radial_encoder = RadialPositionalEncoder(wafer_size, embedding_dim=16)
    input_channels += 16  # 10 → 26 채널로 증가!
```
- Radial encoding 사용 시 실제 입력은 **26채널**이 됨
- Conv stem이 올바르게 처리하는지 확인 필요

---

## 예상 결과

### 학습 효과
1. ✅ **카테고리 구분**: 같은 공간 패턴(edge-ring)이라도 카테고리가 다르면 다른 latent vector
2. ✅ **복합 패턴 학습**: 한 wafer에 여러 카테고리가 섞인 경우도 학습 가능
3. ✅ **Fine-grained clustering**: 기존보다 세밀한 패턴 분류 가능

### 평가 지표
- Silhouette Score: 카테고리별로 latent space에서 잘 분리되는지
- Rotation Invariance: D4 변환에도 불구하고 같은 카테고리는 가까운지
- Cluster Purity: 각 클러스터 내에서 카테고리 분포 확인

---

## 문제 발생 시 체크리스트

### 1. 모델 생성 실패
```
Error: Input channel mismatch
```
→ `input_channels=10`이 모든 단계에 전달되었는지 확인

### 2. 메모리 에러
```
RuntimeError: CUDA out of memory
```
→ Batch size 줄이기 (256 → 128 → 64)

### 3. 학습 불안정
- Loss가 NaN: Learning rate 낮추기 (1e-4 → 5e-5)
- Collapse 발생: Tau 높이기 (0.996 → 0.999)

---

## 참고사항

### 향후 확장 가능성
1. **가중치 적용**: 일부 카테고리가 더 중요하다면 loss에 가중치 추가
2. **Auxiliary task**: 카테고리 분포 예측 head 추가 (supervised)
3. **Attention 활용**: 어떤 채널(카테고리)에 모델이 집중하는지 시각화

---

**작성일**: 2025-01-13  
**목적**: Binary → Multi-category wafer map 전환  
**핵심**: 10채널 입력 (카테고리 1~10), 처음부터 재학습