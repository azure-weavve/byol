# 📋 Multi-Channel BYOL 전환 완료 보고서

## 1. 전체 변경 개요

### 목표
- Binary data (110k) + Category data (90k)를 **통합 학습**
- 기존 BYOL 구조 유지하면서 **multi-channel input** 지원
- **Option A**: Pure multi-channel BYOL (카테고리 학습이 자동으로 되도록)

### 핵심 아이디어
```
모든 데이터를 (n_categories+1, H, W) 형식으로 통일
- Channel 0: Spatial pattern (binary)
- Channel 1~n: Category-specific information

Binary data: Channel 1~n이 모두 0 (카테고리 정보 없음)
Category data: Channel 1~n에 각 카테고리 정보 포함
```

---

## 2. 파일별 수정사항

### 2-1. `utils/dataloader_utils.py`

#### 추가된 함수 (파일 상단에 추가)

**`convert_to_multichannel()`**
```python
def convert_to_multichannel(wafer_map, n_categories=10):
    """
    Binary (H, W) 또는 Category (n_cat, H, W) → (n_categories+1, H, W) 변환
    
    - Binary: channel 0만 채우고 나머지 0
    - Category: channel 0은 binary aggregation, channel 1~n은 각 카테고리
    """
```

**`detect_n_categories()`**
```python
def detect_n_categories(data_configs):
    """
    데이터 파일들을 스캔하여 최대 카테고리 개수 자동 감지
    
    Returns: 최대 카테고리 개수 (예: 10)
    """
```

#### 수정된 함수

**`prepare_clean_data()`**
- 🔴 카테고리 자동 감지: `n_categories = detect_n_categories(data_configs)`
- 🔴 Multi-channel 변환: `multi_channel_wm = convert_to_multichannel(wm, n_categories)`
- 🔴 Shape 검증 수정: `len(wm.shape) in [2, 3]` (binary or category)
- 🔴 필터링은 channel 0에만 적용
- 🔴 최종 반환: `List of (n_categories+1, H, W) arrays`

**`MultiSizeWaferDataset.__init__()`**
- 🔴 Shape 검증 수정: `len(wm.shape) == 3` (C, H, W)
- 🔴 사전 필터링도 channel 0에만 적용

**`MultiSizeWaferDataset.__getitem__()`**
- 🔴 Input이 이미 `(C, H, W)` multi-channel
- 🔴 Resize 시 channel 차원 유지
- `_apply_augmentation()`은 수정 불필요 (자동으로 모든 channel에 적용)

**`collate_fn()`**
- 🔴 Shape 검증: `len(data.shape) == 3` (C, H, W)
- 🔴 Dummy batch: `(1, 11, 128, 128)`

**`create_dataloaders()`**
- ✅ 수정 불필요 (내부에서 `MultiSizeWaferDataset` 사용)

---

### 2-2. `models/encoder.py`

**`WaferEncoder.__init__()`**
- 🔴 기본값 변경: `input_channels=11` (1 → 11)
- 🔴 RadialPositionalEncoder 사용 시: `11 + 16 = 27 channels`
- ✅ 나머지 구조는 모두 동일 (ResNet blocks, attention, pooling)

**`test_encoder()`**
- 🔴 테스트 input: `torch.randn(batch_size, 11, 128, 128)`

---

### 2-3. `models/byol.py`

**`BYOL.__init__()`**
- 🔴 파라미터 추가: `input_channels=11`
- 🔴 Online/Target encoder에 `input_channels` 전달
- ✅ 나머지 구조는 모두 동일

**`test_byol()`**
- 🔴 테스트 input: `torch.randn(batch_size, 11, 128, 128)`

---

### 2-4. `main_byol_training.py`

**`get_default_config()`**
- 🔴 Thetis 데이터 추가:
  ```python
  {"path": ".../thetis/thetis_map_data_goodbinmap.npz", "name": "Thetis"}
  ```
- 🔴 `input_channels` 제거 (자동 감지)

**`train_byol_wafer()`**
- 🔴 Auto-detect channels:
  ```python
  n_channels = wafer_maps[0].shape[0]  # (C, H, W)
  # Safety check 추가 (처음 10개 확인)
  ```
- 🔴 BYOL 생성 시 전달: `input_channels=n_channels`
- ✅ 나머지는 모두 동일

---

## 3. 데이터 준비 체크리스트

### 월요일에 할 일

#### Step 1: Thetis 데이터 확인
```python
# Thetis NPZ 파일 확인
import numpy as np

data = np.load('thetis_map_data_goodbinmap.npz', allow_pickle=True)
maps = data['maps']
labels = data['ids']  # or data['labels']

# 형식 확인
print(f"Total samples: {len(maps)}")
print(f"First sample shape: {maps[0].shape}")  # (n_cat, H, W) 이어야 함
print(f"First sample values: min={maps[0].min()}, max={maps[0].max()}")

# 카테고리 개수 확인
n_cat = maps[0].shape[0]
print(f"Number of categories: {n_cat}")

# 샘플 확인
for i in range(min(5, len(maps))):
    print(f"Sample {i}: shape={maps[i].shape}, sum={maps[i].sum()}")
```

**예상 형식**:
```python
maps[0].shape = (10, 26, 22)  # (n_categories, H, W)
maps[0][0] = [...] # Category 1 map
maps[0][1] = [...] # Category 2 map
...
maps[0][9] = [...] # Category 10 map

# 각 channel은 0 또는 1
# 한 chip은 최대 하나의 카테고리만 1
```

#### Step 2: 데이터 변환 테스트
```python
# 테스트 스크립트 작성
from utils.dataloader_utils import convert_to_multichannel, detect_n_categories

# 1. 카테고리 감지 테스트
configs = [
    {"path": "root.npz", "name": "Root"},
    {"path": "thetis.npz", "name": "Thetis"}
]
n_cat = detect_n_categories(configs)
print(f"Detected categories: {n_cat}")

# 2. 변환 테스트
test_binary = np.random.rand(26, 22) > 0.9  # Binary (H, W)
test_category = np.random.rand(10, 26, 22) > 0.9  # Category (10, H, W)

multi_binary = convert_to_multichannel(test_binary, n_categories=10)
multi_category = convert_to_multichannel(test_category, n_categories=10)

print(f"Binary → Multi: {test_binary.shape} → {multi_binary.shape}")
print(f"Category → Multi: {test_category.shape} → {multi_category.shape}")

# 검증
assert multi_binary.shape == (11, 26, 22)
assert multi_category.shape == (11, 26, 22)
print("✅ Conversion test passed!")
```

#### Step 3: 전체 파이프라인 테스트
```python
# 소규모 테스트 (데이터 일부만)
from utils.dataloader_utils import prepare_clean_data, create_dataloaders

# 소량 데이터로 테스트
test_configs = [
    {"path": "root.npz", "name": "Root"},
    {"path": "thetis.npz", "name": "Thetis"}
]

# 데이터 로드
wafer_maps, labels, info = prepare_clean_data(
    test_configs,
    use_filter=True,
    use_density_aware=False
)

print(f"Loaded: {len(wafer_maps)} samples")
print(f"Shape: {wafer_maps[0].shape}")  # (11, H, W) 확인
print(f"Channels: {wafer_maps[0].shape[0]}")

# DataLoader 생성
train_loader, val_loader = create_dataloaders(
    wafer_maps=wafer_maps,
    labels=labels,
    batch_size=4,  # 작게 시작
    target_size=(128, 128),
    test_size=0.2
)

# 1 batch 확인
for batch_data, batch_data_aug, batch_labels, batch_indices in train_loader:
    print(f"Batch data shape: {batch_data.shape}")  # (B, 11, 128, 128)
    print(f"Batch data aug shape: {batch_data_aug.shape if batch_data_aug is not None else None}")
    break

print("✅ Pipeline test passed!")
```

#### Step 4: 모델 테스트
```python
# 모델 생성 및 forward pass 테스트
from models.byol import BYOL
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BYOL(
    input_channels=11,
    encoder_dim=512,
    projector_hidden=1024,
    projector_out=256,
    predictor_hidden=1024,
    use_radial_encoding=True,
    use_attention=True,
    wafer_size=(128, 128),
    tau=0.996
).to(device)

# 테스트 데이터
view1 = torch.randn(2, 11, 128, 128).to(device)
view2 = torch.randn(2, 11, 128, 128).to(device)

# Forward pass
loss = model(view1, view2)
print(f"Loss: {loss.item()}")

# Embedding 추출
embeddings = model.get_embeddings(view1, use_target=True)
print(f"Embeddings shape: {embeddings.shape}")  # (2, 512)

print("✅ Model test passed!")
```

#### Step 5: 짧은 학습 테스트
```python
# 10 epoch만 돌려보기
python main_byol_training.py
# Config에서 epochs=10으로 수정해서 테스트
```

---

## 4. 이후 옵션들

### Option A (현재 구현): Pure Multi-channel BYOL

**현재 상태**:
- 11 channels input
- 기존 BYOL loss만 사용
- Category 학습이 자동으로 될 수도, 안 될 수도 있음

**평가 방법**:
```python
# 학습 후 평가
from utils.evaluation import evaluate_all

metrics, labels = evaluate_all(model, val_loader, device)

# 추가 분석: Binary vs Category 분리도
embeddings_all = extract_features(model, val_loader, device)

# Binary data embeddings
binary_indices = [...]  # Binary 데이터 인덱스
embeddings_binary = embeddings_all[binary_indices]

# Category data embeddings (카테고리별)
cat3_indices = [...]  # Category 3 데이터 인덱스
embeddings_cat3 = embeddings_all[cat3_indices]

# 거리 계산
dist_binary_vs_cat = torch.cdist(embeddings_binary.mean(0, keepdim=True), 
                                  embeddings_cat3.mean(0, keepdim=True))
print(f"Binary vs Cat3 distance: {dist_binary_vs_cat.item()}")
```

**성공 조건**:
- Binary cluster가 따로 생기지 않음
- 같은 category 내에서 pattern similarity 유지
- 다른 category는 적당히 분리

**실패 조건**:
- Binary vs Category로 나뉨 (카테고리 정보 무시)
- 카테고리가 전혀 반영 안 됨

---

### Option B: BYOL + Category-aware Positive Sampling

**Option A 실패 시** 적용할 방법

**핵심 아이디어**:
```python
# 기존 BYOL loss
byol_loss = byol_loss_function(view1, view2)

# Category-aware loss (90k만)
# Batch 내에서 같은 카테고리 찾기
if has_category:
    same_cat_wafer = find_same_category_in_batch(wafer)
    if same_cat_wafer is not None:
        cat_loss = byol_loss_function(
            embedding(wafer),
            embedding(same_cat_wafer).detach()
        )
    else:
        cat_loss = 0
else:
    cat_loss = 0

# Combined
total_loss = byol_loss + 0.5 * cat_loss
```

**장점**:
- BYOL 철학 유지 (no negative pairs)
- Category 학습 명시적
- 110k binary도 활용

**단점**:
- Batch 내 category diversity 필요
- Batch size 영향 큼

**구현 복잡도**: 중간

---

### Option C: BYOL + SupCon Loss

**Option A/B 모두 실패 시** 적용

**핵심 아이디어**:
```python
# BYOL loss (모든 데이터)
byol_loss = byol_loss_function(view1, view2)

# SupCon loss (90k만)
if has_category:
    supcon_loss = supervised_contrastive_loss(
        embeddings, 
        category_labels,
        temperature=0.07
    )
else:
    supcon_loss = 0

# Combined (5:5 비율)
total_loss = 0.5 * byol_loss + 0.5 * supcon_loss
```

**장점**:
- Category 학습 확실
- 같은 category는 당기고, 다른 category는 밀어냄

**단점**:
- Negative pairs 필요 (BYOL 철학 포기)
- Batch 내 category diversity 중요
- Collapse 위험 증가

**구현 복잡도**: 높음

---

## 5. 예상 학습 시나리오

### 시나리오 1: Option A 성공 (Best Case)
```
Epoch 10: Silhouette ~0.3
Epoch 30: Silhouette ~0.4
Epoch 50: Silhouette ~0.5 ✅

Clustering 결과:
- Cat3 Edge + Cat3 Edge: 가까움 ✅
- Cat3 Edge + Cat3 Center: 멀리 ✅
- Cat3 Edge + Cat7 Edge: 멀리 ✅
- Binary는 pattern에 따라 분산 ✅
```

### 시나리오 2: Option A 실패 (Category 무시)
```
Epoch 50: Silhouette ~0.5
하지만...

Clustering 결과:
- Binary cluster (110k)
- Category cluster (90k)
→ Pattern은 학습했지만 category는 무시 ❌

→ Option B로 전환 필요
```

### 시나리오 3: Option A 부분 성공 (Binary 분리)
```
Clustering 결과:
- Binary: pattern별로 cluster
- Category: pattern + category 모두 반영

하지만 Binary vs Category 간 gap 존재

→ 실사용 가능하지만 Option B로 개선 고려
```

---

## 6. 디버깅 체크리스트

### 데이터 로딩 단계
- [ ] Thetis 파일 존재 및 형식 확인 `(n_cat, H, W)`
- [ ] `detect_n_categories()` 정상 작동 (10 반환)
- [ ] `convert_to_multichannel()` 정상 작동
- [ ] `prepare_clean_data()` 후 shape `(11, H, W)` 확인
- [ ] Binary와 Category 데이터 모두 로드 확인

### DataLoader 단계
- [ ] `MultiSizeWaferDataset` 생성 성공
- [ ] Batch shape `(B, 11, 128, 128)` 확인
- [ ] Augmentation 적용 확인 (optional)

### 모델 단계
- [ ] `WaferEncoder` input channels 11 확인
- [ ] BYOL forward pass 성공
- [ ] Loss 계산 정상 (NaN 아님)
- [ ] Embedding 추출 성공 `(B, 512)`

### 학습 단계
- [ ] GPU 메모리 사용량 확인 (12GB 이내)
- [ ] Loss 감소 확인
- [ ] Feature std 확인 (collapse 없음)
- [ ] Checkpoint 저장 확인

---

## 7. 최종 체크

### 변경 전후 비교

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| Input | (B, 1, H, W) | (B, 11, H, W) |
| 데이터 | Binary 61k | Binary 110k + Category 90k |
| Encoder | input_channels=1 | input_channels=11 |
| Conv1 | Conv2d(1, 64) | Conv2d(11, 64) |
| RadialEncoder | +16 ch → 17 total | +16 ch → 27 total |
| 학습 방식 | BYOL only | BYOL only (Option A) |
| 목표 | Pattern similarity | Pattern + Category |

### 호환성
- ✅ 기존 BYOL 구조 유지
- ✅ 기존 평가 코드 사용 가능
- ✅ 기존 checkpoint 관리 코드 사용 가능
- ❌ 기존 binary 모델과 weight 호환 불가 (input channel 다름)

---

## 8. 요약

### 핵심 변경사항
1. **데이터**: Binary (H, W) + Category (n, H, W) → 모두 (11, H, W)
2. **Encoder**: input_channels 1 → 11
3. **학습**: 기존 BYOL 그대로 (Option A)

### 다음 단계
1. 월요일: Thetis 데이터 확인 및 변환
2. 테스트: 소규모 학습 (10 epochs)
3. 평가: Category 학습 여부 확인
4. 필요시: Option B/C로 전환

### 성공 지표
- Silhouette ≥ 0.5
- Category별 cluster 분리
- Binary data도 pattern에 따라 분산
- Binary vs Category gap 최소화

---

## 9. 문의 및 이슈

문제 발생 시 확인할 사항:
1. 데이터 형식이 맞는지 (Binary: (H,W), Category: (n,H,W))
2. NPZ 파일이 올바르게 생성되었는지
3. 카테고리 개수가 자동 감지되는지
4. Multi-channel 변환이 정상 작동하는지
5. 모델 input shape이 맞는지

---

**작성일**: 2025-01-16  
**버전**: 1.0  
**작성자**: BYOL Multi-channel 전환 프로젝트
