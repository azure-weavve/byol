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
        view1 = D4 + wafer_aug
        view2 = D4 + wafer_aug (different transformation)
    """

    def __init__(self,
                 use_d4=True,
                 use_crop=True,
                 use_noise=True,
                 use_erase=True,
                 use_rotation=False,
                 crop_scale=(0.8, 1.0),
                 noise_std=0.02,
                 erase_prob=0.1,
                 erase_scale=(0.02, 0.1),
                 small_rotation_deg=5,
                 use_c4_only=True):
        """
        Args:
            use_d4: use D4 transformations (CRITICAL!)
            use_crop: use random crop
            use_noise: use Gaussian noise
            use_erase: use random erasing
            use_rotation: use small rotation
            ... other parameters passed to WaferAugmentation
        """
        self.use_d4 = use_d4
        self.use_crop = use_crop
        self.use_noise = use_noise
        self.use_erase = use_erase
        self.use_rotation = use_rotation

        self.wafer_aug = WaferAugmentation(
            crop_scale=crop_scale,
            noise_std=noise_std,
            erase_prob=erase_prob,
            erase_scale=erase_scale,
            small_rotation_deg=small_rotation_deg
        )

    def __call__(self, x):
        """
        Generate two augmented views

        Args:
            x: (C, H, W) tensor

        Returns:
            view1, view2: two augmented views
        """
        # View 1
        view1 = x.clone()
        if self.use_d4:
            if self.use_c4_only:
                view1, _ = D4Transform.random_c4_transform(view1)
            else:
                view1, _ = D4Transform.random_transform(view1)
        view1 = self.wafer_aug(
            view1,
            use_crop=self.use_crop,
            use_noise=self.use_noise,
            use_erase=self.use_erase,
            use_rotation=self.use_rotation
        )

        # View 2 (different transformation)
        view2 = x.clone()
        if self.use_d4:
            if self.use_c4_only:
                view2, _ = D4Transform.random_c4_transform(view2)
            else:
                view2, _ = D4Transform.random_transform(view2)
        view2 = self.wafer_aug(
            view2,
            use_crop=self.use_crop,
            use_noise=self.use_noise,
            use_erase=self.use_erase,
            use_rotation=self.use_rotation
        )

        return view1, view2


def get_byol_augmentation(augmentation_type='strong'):
    """
    Get BYOL augmentation pipeline

    Args:
        augmentation_type: 'strong', 'medium', or 'weak'

    Returns:
        BYOLAugmentation instance
    """
    if augmentation_type == 'strong':
        return BYOLAugmentation(
            use_d4=True,
            use_crop=True,
            use_noise=True,
            use_erase=True,
            use_rotation=False,
            crop_scale=(0.8, 1.0),
            noise_std=0.02,
            erase_prob=0.1
        )

    elif augmentation_type == 'medium':
        return BYOLAugmentation(
            use_d4=True,
            use_crop=True,
            use_noise=True,
            use_erase=False,
            use_rotation=False,
            crop_scale=(0.85, 1.0),
            noise_std=0.01
        )

    elif augmentation_type == 'weak':
        return BYOLAugmentation(
            use_d4=True,
            use_crop=False,
            use_noise=False,
            use_erase=False,
            use_rotation=False
        )

    else:
        raise ValueError(f"Invalid augmentation_type: {augmentation_type}")


def test_augmentation():
    """Test augmentation functions"""
    print("Testing augmentation...")

    # Create sample wafer map
    x = torch.rand(1, 128, 128)

    # Test D4 transformations
    print("\nTesting D4 transformations...")
    all_transforms = D4Transform.get_all_transforms(x)
    print(f"Generated {len(all_transforms)} D4 transformations")

    # Test random D4
    x_transformed, transform_id = D4Transform.random_transform(x)
    print(f"Random D4 transform ID: {transform_id}")
    print(f"Transformed shape: {x_transformed.shape}")

    # Test wafer augmentation
    print("\nTesting wafer augmentation...")
    wafer_aug = WaferAugmentation()
    x_aug = wafer_aug(x)
    print(f"Augmented shape: {x_aug.shape}")

    # Test BYOL augmentation
    print("\nTesting BYOL augmentation...")
    byol_aug = get_byol_augmentation('strong')
    view1, view2 = byol_aug(x)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    # Test batch processing
    print("\nTesting batch augmentation...")
    batch = torch.rand(4, 1, 128, 128)
    view1_batch = []
    view2_batch = []

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