"""
train_byol.py 변경 가이드

아래는 train_byol_epoch / validate_byol_epoch 함수에서
for-loop augmentation을 배치 augmentation으로 교체하는 변경 사항입니다.

변경 포인트가 명확하게 표시되어 있으므로,
기존 train_byol.py에서 해당 부분만 교체하시면 됩니다.
"""


# =============================================================================
# [변경 1] train_byol_epoch - augmentation 루프 제거
# =============================================================================
#
# 위치: train_byol_epoch() 함수 내부
#
# ──── 기존 코드 (삭제) ────
#
#   images = images.to(device)
#   batch_size = images.size(0)
#
#   # Generate two augmented views
#   view1_list = []
#   view2_list = []
#
#   for i in range(batch_size):
#       v1, v2 = augmentation(images[i])
#       view1_list.append(v1)
#       view2_list.append(v2)
#
#   view1 = torch.stack(view1_list).to(device)
#   view2 = torch.stack(view2_list).to(device)
#
# ──── 변경 코드 (교체) ────
#
#   images = images.to(device)
#   batch_size = images.size(0)
#
#   # Generate two augmented views (batch vectorized)
#   view1 = augmentation(images)
#   view2 = augmentation(images)
#


# =============================================================================
# [변경 2] validate_byol_epoch - augmentation 루프 제거
# =============================================================================
#
# 위치: validate_byol_epoch() 함수 내부, with torch.no_grad() 블록
#
# ──── 기존 코드 (삭제) ────
#
#   images = images.to(device)
#   batch_size = images.size(0)
#
#   # Generate two augmented views
#   view1_list = []
#   view2_list = []
#
#   for i in range(batch_size):
#       v1, v2 = augmentation(images[i])
#       view1_list.append(v1)
#       view2_list.append(v2)
#
#   view1 = torch.stack(view1_list).to(device)
#   view2 = torch.stack(view2_list).to(device)
#
# ──── 변경 코드 (교체) ────
#
#   images = images.to(device)
#   batch_size = images.size(0)
#
#   # Generate two augmented views (batch vectorized)
#   view1 = augmentation(images)
#   view2 = augmentation(images)
#


# =============================================================================
# [변경 3] main_byol_training.py - augmentation 생성 변경
# =============================================================================
#
# 위치: train_byol_wafer() 함수 내부
#
# ──── 기존 코드 ────
#
#   from utils.augmentation import get_byol_augmentation
#   augmentation = get_byol_augmentation(config['augmentation_type'])
#
# ──── 변경 코드 ────
#
#   from utils.batch_augmentation import get_batch_byol_augmentation
#   augmentation = get_batch_byol_augmentation(
#       config['augmentation_type'],
#       n_spatial_channels=config.get('n_spatial_channels', 13)
#   )
#
# 주의: 기존 BYOLAugmentation은 evaluation(rotation_invariance 등)에서
#       여전히 개별 샘플 단위로 사용되므로 삭제하지 마세요.


# =============================================================================
# [변경 4] __init__.py 에 배치 augmentation 추가 (선택사항)
# =============================================================================
#
# 위치: utils/__init__.py
#
# 기존 import에 추가:
#
#   from .batch_augmentation import (
#       BatchBYOLAugmentation,
#       get_batch_byol_augmentation
#   )
#
# __all__ 리스트에 추가:
#
#   'BatchBYOLAugmentation',
#   'get_batch_byol_augmentation',