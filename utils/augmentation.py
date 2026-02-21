"""
Data Augmentation for Wafer Maps

Key components:
1. D4 Dihedral Group transformations (8 symmetries) - CRITICAL for rotation invariance
2. Wafer-specific augmentations (crop, noise, etc.)
3. BYOL two-view augmentation pipeline

PyTorch 1.4.0 compatible
"""

import torch
import torch.nn.functional as F
import random
import math


class D4Transform:
    """
    D4 Dihedral Group transformations

    8 symmetry transformations:
        0: Identity
        1: 90° rotation
        2: 180° rotation
        3: 270° rotation
        4: Horizontal flip
        5: Vertical flip
        6: Transpose (diagonal flip)
        7: Anti-transpose (anti-diagonal flip)

    Critical for rotation invariance in wafer patterns!
    """

    @staticmethod
    def apply(x, transform_id):
        """
        Apply D4 transformation

        Args:
            x: (B, C, H, W) or (C, H, W) tensor
            transform_id: 0-7 transformation ID

        Returns:
            transformed tensor
        """
        if transform_id == 0:
            # Identity
            return x

        elif transform_id == 1:
            # 90° rotation (counter-clockwise)
            return torch.rot90(x, k=1, dims=(-2, -1))

        elif transform_id == 2:
            # 180° rotation
            return torch.rot90(x, k=2, dims=(-2, -1))

        elif transform_id == 3:
            # 270° rotation (or 90° clockwise)
            return torch.rot90(x, k=3, dims=(-2, -1))

        elif transform_id == 4:
            # Horizontal flip
            return torch.flip(x, dims=[-1])

        elif transform_id == 5:
            # Vertical flip
            return torch.flip(x, dims=[-2])

        elif transform_id == 6:
            # Transpose (diagonal flip)
            return x.transpose(-2, -1)

        elif transform_id == 7:
            # Anti-transpose: rotate 90° then transpose
            # Equivalent to transpose then rotate -90°
            x = torch.rot90(x, k=1, dims=(-2, -1))
            return x.transpose(-2, -1)

        else:
            raise ValueError(f"Invalid transform_id: {transform_id}. Must be 0-7.")

    @staticmethod
    def random_transform(x):
        """
        Apply random D4 transformation

        Args:
            x: input tensor

        Returns:
            transformed tensor, transform_id
        """
        transform_id = random.randint(0, 7)
        return D4Transform.apply(x, transform_id), transform_id

    @staticmethod
    def get_all_transforms(x):
        """
        Get all 8 D4 transformations

        Args:
            x: (C, H, W) tensor

        Returns:
            list of 8 transformed tensors
        """
        return [D4Transform.apply(x, i) for i in range(8)]

    @staticmethod
    def random_c4_transform(x):
        """C4 only (rotation without reflection)"""
        transform_id = random.randint(0, 3)  # 0~3만
        return D4Transform.apply(x, transform_id), transform_id

    @staticmethod
    def get_c4_transforms(x):
        """Get 4 C4 transforms only"""
        return [D4Transform.apply(x, i) for i in range(4)]


class WaferAugmentation:
    """
    Wafer-specific augmentation pipeline

    Augmentations:
        - Random crop & resize
        - Gaussian noise
        - Random erasing (simulates missing defects)
        - Small rotation (±5°, in addition to D4)
        - Defect dropout (simulates defect intensity variation)
    """

    def __init__(self,
                 crop_scale=(0.8, 1.0),
                 noise_std=0.02,
                 erase_prob=0.1,
                 erase_scale=(0.02, 0.1),
                 small_rotation_deg=5):
        """
        Args:
            crop_scale: (min, max) scale for random crop
            noise_std: standard deviation of Gaussian noise
            erase_prob: probability of random erasing
            erase_scale: (min, max) scale of erased area
            small_rotation_deg: maximum rotation angle in degrees
        """
        self.crop_scale = crop_scale
        self.noise_std = noise_std
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale
        self.small_rotation_deg = small_rotation_deg

    def random_crop_resize(self, x, scale):
        """
        Random crop and resize back to original size

        Args:
            x: (C, H, W) tensor
            scale: (min_scale, max_scale)

        Returns:
            cropped and resized tensor
        """
        C, H, W = x.shape

        # Random scale
        s = random.uniform(scale[0], scale[1])
        crop_h = int(H * s)
        crop_w = int(W * s)

        # Random crop position
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # Crop
        x_cropped = x[:, top:top+crop_h, left:left+crop_w]

        # Resize back (PyTorch 1.4.0 compatible)
        x_resized = F.interpolate(
            x_cropped.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return x_resized

    def add_gaussian_noise(self, x, std):
        """
        Add Gaussian noise

        Args:
            x: input tensor
            std: standard deviation

        Returns:
            noisy tensor
        """
        noise = torch.randn_like(x) * std
        return x + noise

    def random_erase(self, x, prob, scale):
        """
        Random erasing (simulate missing defects)

        Args:
            x: (C, H, W) tensor
            prob: probability of erasing
            scale: (min_scale, max_scale) of erased area

        Returns:
            tensor with random area erased
        """
        if random.random() > prob:
            return x

        C, H, W = x.shape

        # Random erase size
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(0.3, 1.0 / 0.3)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)

            # Erase with zeros (or mean value)
            x_erased = x.clone()
            x_erased[:, top:top+h, left:left+w] = 0
            return x_erased

        return x

    def small_rotation(self, x, max_deg):
        """
        Small rotation (±max_deg degrees)

        Args:
            x: (C, H, W) tensor
            max_deg: maximum rotation angle

        Returns:
            rotated tensor
        """
        # Random angle
        angle = random.uniform(-max_deg, max_deg)

        # Convert to radians
        theta = math.radians(angle)

        # Rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Affine transformation matrix (2x3)
        # For PyTorch 1.4.0, we use affine_grid + grid_sample
        affine_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        # Create grid and sample
        C, H, W = x.shape
        grid = F.affine_grid(affine_matrix, [1, C, H, W], align_corners=False)
        x_rotated = F.grid_sample(
            x.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0)

        return x_rotated

    def defect_dropout(self, x, dropout_rate_range=(0.1, 0.5), n_spatial_channels=13):
        """
        Defect 픽셀을 확률적으로 제거하여 강도 변화에 대한 invariance 학습.
        Spatial 채널(앞쪽 n_spatial_channels)에만 적용하고,
        Radial positional encoding 채널(뒤쪽)은 건드리지 않음.

        Args:
            x: (C, H, W) tensor, binary wafer map
            dropout_rate_range: (min_rate, max_rate) 범위에서 uniform sampling
            n_spatial_channels: spatial 채널 수 (기본 13)
                                 나머지 채널은 radial encoding으로 간주하여 dropout 미적용

        Returns:
            dropout이 적용된 tensor
        """
        dropout_rate = random.uniform(dropout_rate_range[0], dropout_rate_range[1])

        x = x.clone()
        C, H, W = x.shape

        # 실제 spatial 채널 수 결정 (채널 수가 n_spatial_channels보다 적을 경우 대비)
        n_spatial = min(n_spatial_channels, C)

        spatial = x[:n_spatial]  # (n_spatial, H, W)

        # drop_mask: (1, H, W) → spatial 채널 전체에 동일한 마스크 broadcast
        # 채널마다 다른 마스크를 쓰면 같은 위치 defect가 채널별로 들쭉날쭉해짐
        drop_mask = torch.rand(1, H, W, dtype=x.dtype, device=x.device) < dropout_rate

        defect_mask = (spatial > 0.5)           # (n_spatial, H, W)
        spatial[defect_mask & drop_mask] = 0    # broadcast: drop_mask (1,H,W) → (n_spatial,H,W)

        x[:n_spatial] = spatial
        return x

    def __call__(self, x, use_crop=True, use_noise=True, use_erase=True, use_rotation=False):
        """
        Apply augmentation pipeline

        Args:
            x: (C, H, W) tensor
            use_crop: apply random crop
            use_noise: apply Gaussian noise
            use_erase: apply random erasing
            use_rotation: apply small rotation

        Returns:
            augmented tensor
        """
        if use_crop:
            x = self.random_crop_resize(x, self.crop_scale)

        if use_noise:
            x = self.add_gaussian_noise(x, self.noise_std)

        if use_erase:
            x = self.random_erase(x, self.erase_prob, self.erase_scale)

        if use_rotation:
            x = self.small_rotation(x, self.small_rotation_deg)

        # Clamp to valid range [0, 1] for binary wafer maps
        x = torch.clamp(x, 0, 1)

        return x


class BYOLAugmentation:
    """
    BYOL two-view augmentation pipeline

    Creates two augmented views of the same image:
        view1 = D4(C4) + defect_dropout
        view2 = D4(C4) + defect_dropout (다른 랜덤 값)
    """

    def __init__(self,
                 use_d4=True,
                 use_c4_only=True,
                 use_crop=False,
                 use_noise=False,
                 use_erase=False,
                 use_rotation=False,
                 use_defect_dropout=True,
                 defect_dropout_prob=0.5,
                 defect_dropout_rate=(0.1, 0.5),
                 n_spatial_channels=13,
                 crop_scale=(0.8, 1.0),
                 noise_std=0.02,
                 erase_prob=0.1,
                 erase_scale=(0.02, 0.1),
                 small_rotation_deg=5):
        """
        Args:
            use_d4: D4 변환 사용 여부 (CRITICAL)
            use_c4_only: True면 C4(회전만), False면 D4(회전+반사) 사용
            use_crop: random crop 사용 여부 (wafer map에는 비권장)
            use_noise: Gaussian noise 사용 여부 (binary map에는 비권장)
            use_erase: random erase 사용 여부 (패턴 왜곡 우려로 비권장)
            use_rotation: small rotation 사용 여부
            use_defect_dropout: defect dropout 사용 여부
            defect_dropout_prob: defect dropout 적용 확률 (view당)
            defect_dropout_rate: (min, max) dropout rate range
            n_spatial_channels: spatial 채널 수 (나머지는 radial encoding으로 간주)
        """
        self.use_d4 = use_d4
        self.use_c4_only = use_c4_only
        self.use_crop = use_crop
        self.use_noise = use_noise
        self.use_erase = use_erase
        self.use_rotation = use_rotation
        self.use_defect_dropout = use_defect_dropout
        self.defect_dropout_prob = defect_dropout_prob
        self.defect_dropout_rate = defect_dropout_rate
        self.n_spatial_channels = n_spatial_channels

        self.wafer_aug = WaferAugmentation(
            crop_scale=crop_scale,
            noise_std=noise_std,
            erase_prob=erase_prob,
            erase_scale=erase_scale,
            small_rotation_deg=small_rotation_deg
        )

    def _apply_single_view(self, x):
        """
        단일 view에 augmentation 적용

        Args:
            x: (C, H, W) tensor

        Returns:
            augmented tensor
        """
        view = x.clone()

        # D4 / C4 rotation
        if self.use_d4:
            if self.use_c4_only:
                view, _ = D4Transform.random_c4_transform(view)
            else:
                view, _ = D4Transform.random_transform(view)

        # 기존 augmentation (기본 비활성화)
        view = self.wafer_aug(
            view,
            use_crop=self.use_crop,
            use_noise=self.use_noise,
            use_erase=self.use_erase,
            use_rotation=self.use_rotation
        )

        # Defect dropout (spatial 채널만)
        if self.use_defect_dropout and random.random() < self.defect_dropout_prob:
            view = self.wafer_aug.defect_dropout(
                view,
                dropout_rate_range=self.defect_dropout_rate,
                n_spatial_channels=self.n_spatial_channels
            )

        return view

    def __call__(self, x):
        """
        Generate two augmented views

        Args:
            x: (C, H, W) tensor

        Returns:
            view1, view2: two augmented views (서로 독립적인 랜덤 변환)
        """
        view1 = self._apply_single_view(x)
        view2 = self._apply_single_view(x)
        return view1, view2


def get_byol_augmentation(augmentation_type='strong', n_spatial_channels=13):
    """
    Get BYOL augmentation pipeline

    Args:
        augmentation_type: 'strong', 'medium', or 'weak'
        n_spatial_channels: spatial 채널 수 (모델 설정에 맞게 전달)

    Returns:
        BYOLAugmentation instance
    """
    if augmentation_type == 'strong':
        return BYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_crop=False,
            use_noise=False,
            use_erase=False,
            use_rotation=False,
            use_defect_dropout=True,
            defect_dropout_prob=0.5,
            defect_dropout_rate=(0.1, 0.5),
            n_spatial_channels=n_spatial_channels,
        )

    elif augmentation_type == 'medium':
        return BYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_crop=False,
            use_noise=False,
            use_erase=False,
            use_rotation=False,
            use_defect_dropout=True,
            defect_dropout_prob=0.3,
            defect_dropout_rate=(0.1, 0.3),
            n_spatial_channels=n_spatial_channels,
        )

    elif augmentation_type == 'weak':
        return BYOLAugmentation(
            use_d4=True,
            use_c4_only=True,
            use_crop=False,
            use_noise=False,
            use_erase=False,
            use_rotation=False,
            use_defect_dropout=False,
            n_spatial_channels=n_spatial_channels,
        )

    else:
        raise ValueError(f"Invalid augmentation_type: {augmentation_type}")


def test_augmentation():
    """Test augmentation functions"""
    print("Testing augmentation...")

    # 29채널 (13 spatial + 16 radial) 테스트
    C, H, W = 29, 128, 128
    x = torch.zeros(C, H, W)
    # spatial 채널에만 임의 defect 생성
    x[:13] = (torch.rand(13, H, W) > 0.85).float()

    # Test D4 transformations
    print("\nTesting D4 transformations...")
    all_transforms = D4Transform.get_all_transforms(x)
    print(f"Generated {len(all_transforms)} D4 transformations")

    x_transformed, transform_id = D4Transform.random_transform(x)
    print(f"Random D4 transform ID: {transform_id}")
    print(f"Transformed shape: {x_transformed.shape}")

    # Test defect dropout (spatial only)
    print("\nTesting defect dropout (spatial only)...")
    wafer_aug = WaferAugmentation()
    before_defects_spatial = x[:13].sum().item()
    before_defects_radial  = x[13:].sum().item()

    x_dropped = wafer_aug.defect_dropout(x, dropout_rate_range=(0.3, 0.3), n_spatial_channels=13)

    after_defects_spatial = x_dropped[:13].sum().item()
    after_defects_radial  = x_dropped[13:].sum().item()

    print(f"Spatial defects  before: {before_defects_spatial:.0f}, after: {after_defects_spatial:.0f}")
    print(f"Radial  channels before: {before_defects_radial:.0f},  after: {after_defects_radial:.0f}  (변화 없어야 함)")
    assert before_defects_radial == after_defects_radial, "Radial 채널이 변경됨!"
    assert after_defects_spatial < before_defects_spatial, "Spatial dropout이 적용되지 않음!"

    # Test BYOL augmentation
    print("\nTesting BYOL augmentation (strong)...")
    byol_aug = get_byol_augmentation('strong', n_spatial_channels=13)
    view1, view2 = byol_aug(x)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    # Test batch processing
    print("\nTesting batch augmentation...")
    batch = torch.zeros(4, C, H, W)
    batch[:, :13] = (torch.rand(4, 13, H, W) > 0.85).float()

    view1_batch, view2_batch = [], []
    for i in range(batch.size(0)):
        v1, v2 = byol_aug(batch[i])
        view1_batch.append(v1)
        view2_batch.append(v2)

    view1_batch = torch.stack(view1_batch)
    view2_batch = torch.stack(view2_batch)
    print(f"Batch view 1 shape: {view1_batch.shape}")
    print(f"Batch view 2 shape: {view2_batch.shape}")

    print("\nAugmentation test passed!")


if __name__ == "__main__":
    test_augmentation()