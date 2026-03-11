# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

반도체 웨이퍼 불량 패턴 클러스터링을 위한 BYOL(Bootstrap Your Own Latent) 기반 자기지도 학습 시스템. 입력은 다중 채널 웨이퍼 맵이며, 목표는 유사한 패턴이 latent space에서 가깝게 위치하도록 학습하는 것.

## 주요 명령어

### 학습 실행
```bash
# 기본 학습 (main 함수 내 path 설정 필요)
python main_byol_training.py

# 체크포인트에서 재개: get_default_config()의 resume_path 값 설정 후 실행
```

### 개별 모듈 테스트
```bash
python models/encoder.py       # WaferEncoder 테스트
python models/projector.py     # Projector/Predictor 테스트
python models/byol.py          # BYOL 전체 모델 테스트
python utils/augmentation.py   # D4 + defect dropout 테스트
python utils/evaluation.py     # 평가 지표 테스트
python utils/byol_monitor.py   # 모니터 테스트
```

### 패키지 설치
```bash
pip install torch==1.4.0 torchvision==0.5.0  # CUDA 버전에 맞게 조정
pip install -r requirements.txt
```

## 아키텍처

### 데이터 흐름
1. **원본 데이터**: `.npz` 파일, `maps` 키 (H, W) 정수 배열(값 0~12), `ids` 키
2. **변환** (`dataloader_utils.convert_to_multichannel`): (H,W) → (13, H, W)
   - channel[0]: 전체 불량 위치 (값 > 0)
   - channel[1~12]: 카테고리별 one-hot
3. **모델 입력**: (B, 13, H, W) — 학습 루프에서 augmentation 적용 후 입력
4. **출력**: (B, 512) feature vector

### 채널 구성 (`n_spatial_channels=13`)
- **Spatial 채널** (13개): 불량 카테고리 데이터
- **Radial encoding 채널** (16개, 선택적): `RadialPositionalEncoder`가 추가 concat → 실제 모델 입력 29채널
- `defect_dropout`은 spatial 채널(앞 13개)에만 적용

### 모델 구조
```
Online:  encoder_online → projector_online → predictor
Target:  encoder_target → projector_target  (EMA 업데이트, no grad)

Loss: symmetric_byol_loss(pred_online_v1, proj_target_v2, pred_online_v2, proj_target_v1)
```

**WaferEncoder** (`models/encoder.py`): ResNet-18 기반, ~11M 파라미터
- `RadialPositionalEncoder`: 웨이퍼 중심 거리 인코딩 (embedding_dim=16)
- `SelfAttention2D`: 전역 패턴 포착 (layer4 출력에 적용)
- `input_channels` 파라미터가 실제 채널 수를 결정 (`use_radial_encoding=True`이면 내부에서 +16)

### 학습 설정 (`get_default_config`)
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `batch_size` | 256 | 12GB VRAM 기준, 384 테스트 가능 |
| `tau_base` / `tau_max` | 0.996 / 0.999 | EMA momentum (cosine 스케줄) |
| `eval_frequency` | 5 | N 에폭마다 평가 |
| `n_spatial_channels` | 13 | Spatial 채널 수 |
| `variance_weight` | 0.05 | Feature std 정규화 가중치 |
| `uniformity_weight` | 0.005 | Uniformity loss 가중치 |

### 복합 점수 (Best model 기준)
```
composite = knn_consistency × 0.5 + silhouette × 0.3 + avg_cos_sim × (-0.2)
```
- `avg_cos_sim`은 낮을수록 좋음 (collapse 방지)

## 핵심 유틸리티

| 파일 | 역할 |
|------|------|
| `utils/batch_augmentation.py` | 배치 단위 벡터화 augmentation (C4 + defect dropout) |
| `utils/augmentation.py` | D4 변환 정의, `BYOLAugmentation` (개별 샘플용) |
| `utils/dataloader_utils.py` | `.npz` 로드 → multi-channel 변환 → DataLoader 생성 |
| `utils/train_byol.py` | `train_byol_epoch`, `validate_byol_epoch`, `extract_features`, `save/load_checkpoint` |
| `utils/evaluation.py` | `evaluate_all`, kNN consistency, silhouette, rotation invariance |
| `utils/byol_monitor.py` | 학습 곡선 플롯, history 저장, 조기 종료 |

## PyTorch 1.4.0 제약사항

```python
# ❌ 사용 불가
torch.quantile()  → torch.kthvalue() 사용
nn.SyncBatchNorm  → nn.BatchNorm2d 사용
torch.cuda.amp    → 사용 불가 (mixed precision 없음)
```

## 체크포인트 및 출력

- `checkpoints/best_model.pth`: composite score 최고 모델
- `checkpoints/temp_checkpoint.pth`: 매 에폭 저장 (evaluation 미완료 복구용 `pending_evaluation` 플래그 포함)
- `logs/history.json`: 전체 학습 히스토리
- `logs/training_curves.png`, `logs/evaluation_metrics.png`

resume 시 `config['resume_path']`에 체크포인트 경로 지정. `temp_checkpoint.pth`를 지정하면 평가 미완료 에폭을 자동 재시도.

## Collapse 감지

학습 중 `detect_collapse`로 매 에폭 확인:
- `feat_std` < 임계값: feature 다양성 소실
- `avg_cos_sim` → 1: 모든 출력이 동일

**해결책**: predictor 존재 확인, tau 증가(0.999), learning rate 감소.
