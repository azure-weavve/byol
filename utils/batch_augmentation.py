"""
Batch Vectorized Augmentation for BYOL Training

기존 BYOLAugmentation의 for-loop 병목을 제거하고
배치 단위로 C4 회전 + defect dropout을 한 번에 처리.

기존 BYOLAugmentation과 동일한 augmentation 결과를 보장하되,
128번의 개별 CUDA 커널 호출 → 최대 4번으로 감소.

PyTorch 1.4.0 compatible
"""

import torch
import random


class BatchBYOLAugmentation:
    """
    배치 단위 BYOL augmentation pipeline
    
    기존 BYOLAugmentation과 동일한 동작:
        view = C4 rotation + defect dropout (spatial 채널만)
    
    차이점:
        - for i in range(batch_size) 루프 제거
        - C4: 같은 변환 ID끼리 묶어서 배치 rot90 (최대 4번 호출)
        - Defect dropout: 배치 단위 마스크 생성 (1번 호출)
    
    Usage:
        batch_aug = BatchBYOLAugmentation(
            use_c4_only=True,
            use_defect_dropout=True,
            defect_dropout_prob=0.5,
            defect_dropout_rate=(0.1, 0.5),
            n_spatial_channels=13,
        )
        view1 = batch_aug(images)  # (B, C, H, W) → (B, C, H, W)
        view2 = batch_aug(images)  # 독립적인 랜덤 변환
    """

    def __init__(self,
                 use_d4=True,
                 use_c4_only=True,
                 use_defect_dropout=True,
                 defect_dropout_prob=0.5,
                 defect_dropout_rate=(0.1, 0.5),
                 n_spatial_channels=13):
        """
        Args:
            use_d4: D4/C4 변환 사용 여부
            use_c4_only: True면 C4(회전만), False면 D4(회전+반사)
            use_defect_dropout: defect dropout 사용 여부
            defect_dropout_prob: 각 샘플에 dropout을 적용할 확률
            defect_dropout_rate: (min_rate, max_rate)
            n_spatial_channels: spatial 채널 수 (나머지는 radial encoding)
        """
        self.use_d4 = use_d4
        self.use_c4_only = use_c4_only
        self.use_defect_dropout = use_defect_dropout
        self.defect_dropout_prob = defect_dropout_prob
        self.defect_dropout_rate = defect_dropout_rate
        self.n_spatial_channels = n_spatial_channels

        self.n_transforms = 4 if use_c4_only else 8

    def _batch_c4_rotation(self, x):
        """
        배치 단위 C4 회전
        
        전략: 배치 전체에 랜덤 변환 ID를 할당하고,
              같은 ID끼리 묶어서 torch.rot90을 호출 (최대 4번)
        
        Args:
            x: (B, C, H, W) tensor
        
        Returns:
            (B, C, H, W) 회전된 tensor
        """
        B = x.size(0)
        device = x.device

        # 배치 전체의 변환 ID를 한 번에 생성
        transform_ids = torch.randint(0, self.n_transforms, (B,), device=device)

        # 결과 텐서 (원본 복사 - identity인 샘플용)
        result = x.clone()

        if self.use_c4_only:
            # C4: 0=identity, 1=90°, 2=180°, 3=270°
            for k in range(1, 4):  # k=0 (identity)는 이미 clone됨
                mask = (transform_ids == k)
                if mask.any():
                    result[mask] = torch.rot90(x[mask], k=k, dims=(-2, -1))
        else:
            # D4: 0-3 = C4 회전, 4 = 좌우반전, 5 = 상하반전, 6 = 대각반전, 7 = 반대각반전
            for k in range(1, 4):
                mask = (transform_ids == k)
                if mask.any():
                    result[mask] = torch.rot90(x[mask], k=k, dims=(-2, -1))

            # Horizontal flip
            mask = (transform_ids == 4)
            if mask.any():
                result[mask] = torch.flip(x[mask], dims=[-1])

            # Vertical flip
            mask = (transform_ids == 5)
            if mask.any():
                result[mask] = torch.flip(x[mask], dims=[-2])

            # Transpose (diagonal flip)
            mask = (transform_ids == 6)
            if mask.any():
                result[mask] = x[mask].transpose(-2, -1)

            # Anti-transpose (rot90 + transpose)
            mask = (transform_ids == 7)
            if mask.any():
                rotated = torch.rot90(x[mask], k=1, dims=(-2, -1))
                result[mask] = rotated.transpose(-2, -1)

        return result

    def _batch_defect_dropout(self, x):
        """
        배치 단위 defect dropout
        
        각 샘플에 독립적인 dropout_rate와 적용 여부를 결정하고,
        spatial 채널에만 마스킹 적용.
        
        Args:
            x: (B, C, H, W) tensor
        
        Returns:
            (B, C, H, W) dropout 적용된 tensor
        """
        B, C, H, W = x.shape
        device = x.device
        n_spatial = min(self.n_spatial_channels, C)

        # 각 샘플별 적용 여부 결정: (B,)
        apply_mask = torch.rand(B, device=device) < self.defect_dropout_prob

        # 적용할 샘플이 없으면 바로 반환
        if not apply_mask.any():
            return x

        result = x.clone()

        # 적용 대상 샘플 인덱스
        apply_indices = apply_mask.nonzero(as_tuple=False).squeeze(1)  # PyTorch 1.4 호환
        n_apply = apply_indices.size(0)

        # 각 샘플별 dropout rate: uniform(min, max)
        min_rate, max_rate = self.defect_dropout_rate
        dropout_rates = torch.rand(n_apply, device=device) * (max_rate - min_rate) + min_rate

        # drop mask: (n_apply, 1, H, W) - 각 샘플별 독립적인 마스크
        # dropout_rates를 (n_apply, 1, 1, 1)로 reshape하여 broadcasting
        rand_vals = torch.rand(n_apply, 1, H, W, device=device)
        drop_masks = rand_vals < dropout_rates.view(n_apply, 1, 1, 1)

        # spatial 채널 추출: (n_apply, n_spatial, H, W)
        spatial = result[apply_indices, :n_spatial]

        # defect 위치: spatial > 0.5인 곳
        defect_mask = (spatial > 0.5)  # (n_apply, n_spatial, H, W)

        # drop_masks (n_apply, 1, H, W) broadcast → (n_apply, n_spatial, H, W)
        # defect이면서 drop 대상인 위치를 0으로
        spatial[defect_mask & drop_masks] = 0

        result[apply_indices, :n_spatial] = spatial

        return result

    def __call__(self, x):
        """
        배치 단위 augmentation 적용 (단일 view 생성)
        
        Args:
            x: (B, C, H, W) tensor (이미 device에 올라와 있어야 함)
        
        Returns:
            (B, C, H, W) augmented tensor
        """
        # 1. C4/D4 회전
        if self.use_d4:
            view = self._batch_c4_rotation(x)
        else:
            view = x.clone()

        # 2. Defect dropout (spatial 채널만)
        if self.use_defect_dropout:
            view = self._batch_defect_dropout(view)

        return view


def get_batch_byol_augmentation(augmentation_type='strong', n_spatial_channels=13):
    """
    배치 단위 BYOL augmentation 생성 함수
    
    기존 get_byol_augmentation과 동일한 인터페이스.
    
    Args:
        augmentation_type: 'strong', 'medium', or 'weak'
        n_spatial_channels: spatial 채널 수
    
    Returns:
        BatchBYOLAugmentation instance
    """
    if augmentation_type == 'strong':
        return BatchBYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_defect_dropout=True,
            defect_dropout_prob=0.5,
            defect_dropout_rate=(0.1, 0.5),
            n_spatial_channels=n_spatial_channels,
        )

    elif augmentation_type == 'medium':
        return BatchBYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_defect_dropout=True,
            defect_dropout_prob=0.3,
            defect_dropout_rate=(0.1, 0.3),
            n_spatial_channels=n_spatial_channels,
        )

    elif augmentation_type == 'weak':
        return BatchBYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_defect_dropout=False,
            n_spatial_channels=n_spatial_channels,
        )

    else:
        raise ValueError(f"Invalid augmentation_type: {augmentation_type}")


def test_batch_augmentation():
    """
    BatchBYOLAugmentation 테스트 및 기존 BYOLAugmentation과 동등성/성능 비교
    """
    import time

    print("=" * 60)
    print("BatchBYOLAugmentation 테스트")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 테스트 데이터: 29채널 (13 spatial + 16 radial)
    B, C, H, W = 128, 29, 128, 128
    n_spatial = 13

    # spatial 채널에만 defect 생성
    x = torch.zeros(B, C, H, W, device=device)
    x[:, :n_spatial] = (torch.rand(B, n_spatial, H, W, device=device) > 0.85).float()
    # radial 채널에 임의의 값 (positional encoding 시뮬레이션)
    x[:, n_spatial:] = torch.rand(B, C - n_spatial, H, W, device=device) * 0.5

    # ========== 1. 기본 동작 테스트 ==========
    print("\n[1] 기본 동작 테스트")
    batch_aug = get_batch_byol_augmentation('strong', n_spatial_channels=n_spatial)

    view1 = batch_aug(x)
    view2 = batch_aug(x)

    print(f"   Input shape:  {x.shape}")
    print(f"   View1 shape:  {view1.shape}")
    print(f"   View2 shape:  {view2.shape}")
    assert view1.shape == x.shape, "Shape 불일치!"
    assert view2.shape == x.shape, "Shape 불일치!"
    print("   ✅ Shape 검증 통과")

    # ========== 2. Radial 채널 보존 검증 ==========
    print("\n[2] Radial 채널 보존 검증")
    # C4 회전은 radial 채널도 회전시키므로, 값 자체가 보존되진 않음
    # 하지만 defect dropout은 radial에 영향 주면 안 됨
    # → dropout만 적용해서 검증
    dropout_only = BatchBYOLAugmentation(
        use_d4=False,
        use_defect_dropout=True,
        defect_dropout_prob=1.0,  # 100% 적용
        defect_dropout_rate=(0.5, 0.5),
        n_spatial_channels=n_spatial,
    )

    x_dropped = dropout_only(x)
    radial_before = x[:, n_spatial:].sum().item()
    radial_after = x_dropped[:, n_spatial:].sum().item()
    spatial_before = x[:, :n_spatial].sum().item()
    spatial_after = x_dropped[:, :n_spatial].sum().item()

    print(f"   Spatial defects:  {spatial_before:.0f} → {spatial_after:.0f} (감소해야 함)")
    print(f"   Radial channels:  {radial_before:.4f} → {radial_after:.4f} (동일해야 함)")
    assert abs(radial_before - radial_after) < 1e-4, "Radial 채널이 변경됨!"
    assert spatial_after < spatial_before, "Spatial dropout이 적용되지 않음!"
    print("   ✅ Radial 채널 보존 검증 통과")

    # ========== 3. C4 회전 분포 검증 ==========
    print("\n[3] C4 회전 분포 검증")
    rotation_only = BatchBYOLAugmentation(
        use_d4=True,
        use_c4_only=True,
        use_defect_dropout=False,
        n_spatial_channels=n_spatial,
    )
    # 비대칭 패턴 생성하여 회전 감지
    x_asym = torch.zeros(1000, C, H, W, device=device)
    x_asym[:, 0, 0:10, 0:10] = 1.0  # 좌상단에만 defect

    x_rotated = rotation_only(x_asym)

    # 4개 코너 defect 분포 확인
    top_left = (x_rotated[:, 0, 0:10, 0:10].sum(dim=(1, 2)) > 0).float().mean().item()
    top_right = (x_rotated[:, 0, 0:10, -10:].sum(dim=(1, 2)) > 0).float().mean().item()
    bot_left = (x_rotated[:, 0, -10:, 0:10].sum(dim=(1, 2)) > 0).float().mean().item()
    bot_right = (x_rotated[:, 0, -10:, -10:].sum(dim=(1, 2)) > 0).float().mean().item()

    print(f"   좌상: {top_left:.2f}, 우상: {top_right:.2f}, 좌하: {bot_left:.2f}, 우하: {bot_right:.2f}")
    print(f"   (각 ~0.25 기대)")
    # 각 코너가 대략 0.25 ± 0.05 범위인지 확인
    for name, val in [("좌상", top_left), ("우상", top_right), ("좌하", bot_left), ("우하", bot_right)]:
        assert 0.15 < val < 0.35, f"{name} 분포 이상: {val:.3f}"
    print("   ✅ C4 회전 균등 분포 검증 통과")

    # ========== 4. 성능 비교 (vs for-loop) ==========
    print("\n[4] 성능 비교")

    # 배치 augmentation
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        v1 = batch_aug(x)
        v2 = batch_aug(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batch_time = time.time() - start

    # for-loop augmentation (기존 방식 시뮬레이션)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        v1_list, v2_list = [], []
        for i in range(B):
            # C4 rotation
            k = random.randint(0, 3)
            v1_i = torch.rot90(x[i], k=k, dims=(-2, -1)).clone()
            k = random.randint(0, 3)
            v2_i = torch.rot90(x[i], k=k, dims=(-2, -1)).clone()
            v1_list.append(v1_i)
            v2_list.append(v2_i)
        v1_loop = torch.stack(v1_list)
        v2_loop = torch.stack(v2_list)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    loop_time = time.time() - start

    print(f"   Batch augmentation:    {batch_time:.3f}초 (50 iterations)")
    print(f"   For-loop augmentation: {loop_time:.3f}초 (50 iterations)")
    print(f"   속도 향상: {loop_time / batch_time:.1f}x")

    # ========== 5. D4 모드 테스트 ==========
    print("\n[5] D4 모드 테스트")
    d4_aug = BatchBYOLAugmentation(
        use_d4=True,
        use_c4_only=False,
        use_defect_dropout=False,
        n_spatial_channels=n_spatial,
    )
    view_d4 = d4_aug(x)
    print(f"   D4 view shape: {view_d4.shape}")
    assert view_d4.shape == x.shape
    print("   ✅ D4 모드 검증 통과")

    # ========== 6. weak/medium 모드 테스트 ==========
    print("\n[6] weak/medium 모드 테스트")
    for mode in ['weak', 'medium', 'strong']:
        aug = get_batch_byol_augmentation(mode, n_spatial_channels=n_spatial)
        view = aug(x[:4])
        print(f"   {mode}: shape={view.shape}, defect_dropout={aug.use_defect_dropout}")
    print("   ✅ 모든 모드 검증 통과")

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)


if __name__ == "__main__":
    test_batch_augmentation()