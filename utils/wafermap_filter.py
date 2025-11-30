import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class WaferMapFilter:
    """
    웨이퍼맵에서 random defect를 제거하고 의미있는 패턴만 유지하는 필터
    
    주요 전략:
    1. Small isolated defects 제거 (1-2개 chip)
    2. 패턴의 모서리 보존 (edge-preserving)
    3. Batch 처리로 속도 최적화
    """
    
    def __init__(self, 
                 min_component_size: int = 3,
                 opening_kernel_size: int = 3,
                 closing_kernel_size: int = 5,
                 edge_preserve_strength: float = 0.7):
        """
        Args:
            min_component_size: 유지할 최소 연결 컴포넌트 크기 (chip 개수)
            opening_kernel_size: Opening 연산 커널 크기 (작은 noise 제거)
            closing_kernel_size: Closing 연산 커널 크기 (구멍 메우기)
            edge_preserve_strength: 모서리 보존 강도 (0~1, 높을수록 원본 유지)
        """
        self.min_component_size = min_component_size
        self.opening_kernel_size = opening_kernel_size
        self.closing_kernel_size = closing_kernel_size
        self.edge_preserve_strength = edge_preserve_strength
        
    def filter_single_map(self, wafer_map: np.ndarray, density_category: str = None) -> np.ndarray:
        """
        단일 웨이퍼맵 필터링 (numpy 기반)
        
        Args:
            wafer_map: (H, W) binary map (0: good/non-wafer, 1: bad chip)
        
        Returns:
            filtered_map: (H, W) 필터링된 맵
        """
        if wafer_map.sum() == 0:
            return wafer_map.copy()
        
        original = wafer_map.copy()
        
        # Low density: 이웃 기반 필터링
        if density_category == 'very_low' or (self.opening_kernel_size == 0 and 
                                        self.closing_kernel_size == 0 and 
                                        self.edge_preserve_strength == 0.0):
            return self._filter_isolated_defects(wafer_map, neighbor_threshold=1)
        
        # Step 1: Morphological Opening (작은 noise 제거)
        if self.opening_kernel_size == 0:
            opened = wafer_map.copy()
        else:
            struct = ndimage.generate_binary_structure(2, 1)
            struct = ndimage.iterate_structure(struct, self.opening_kernel_size // 2)
            opened = ndimage.binary_opening(wafer_map, structure=struct)
        
        # Step 2: Connected Component Analysis (크기 기반 필터링)
        labeled, num_features = ndimage.label(opened)
        
        if num_features == 0:
            return np.zeros_like(wafer_map)
        
        # 각 컴포넌트의 크기 계산
        component_sizes = ndimage.sum(opened, labeled, range(1, num_features + 1))
        
        # 최소 크기 이상인 컴포넌트만 유지
        valid_components = np.where(component_sizes >= self.min_component_size)[0] + 1
        
        if len(valid_components) == 0:
            return np.zeros_like(wafer_map)
        
        # 유효한 컴포넌트만 남기기
        mask = np.isin(labeled, valid_components)
        filtered = opened * mask
        
        # Step 3: Morphological Closing (패턴 내부 구멍 메우기)
        if self.closing_kernel_size == 0:
            closed = filtered.copy()
        else:
            struct_close = ndimage.generate_binary_structure(2, 1)
            struct_close = ndimage.iterate_structure(struct_close, self.closing_kernel_size // 2)
            closed = ndimage.binary_closing(filtered, structure=struct_close)
        
        # Step 4: Edge-preserving reconstruction
        # 원본에서 발견된 edge를 부분적으로 복원
        edges = self._detect_edges(original)
        edge_pixels = original * edges
        
        # 필터링된 맵과 edge 픽셀을 결합
        result = closed.astype(float)
        if self.edge_preserve_strength != 0.0:
            result = result + self.edge_preserve_strength * edge_pixels
            result = (result > 0.5).astype(np.float32)            
        
        return result
    
    def _detect_edges(self, binary_map: np.ndarray) -> np.ndarray:
        """
        패턴의 edge 검출
        
        Args:
            binary_map: (H, W) binary map
        
        Returns:
            edge_map: (H, W) edge map
        """
        # Sobel edge detection
        sx = ndimage.sobel(binary_map.astype(float), axis=0)
        sy = ndimage.sobel(binary_map.astype(float), axis=1)
        edge_magnitude = np.hypot(sx, sy)
        
        # Normalize and threshold
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()
        
        edges = (edge_magnitude > 0.3).astype(np.float32)
        
        return edges
    
    def _filter_isolated_defects(self, wafer_map: np.ndarray, 
                             neighbor_threshold: int = 2) -> np.ndarray:
        """
        고립된 defect만 제거 (이웃이 적은 픽셀 제거)
        
        Args:
            wafer_map: (H, W) binary map
            neighbor_threshold: 최소 이웃 defect 수 (8-이웃 기준)
                            2: 자신 포함 3개 이상 뭉쳐있으면 유지
                            1: 자신 포함 2개 이상 (완전 고립만 제거)
        
        Returns:
            filtered_map: 고립된 defect 제거된 맵
        """
        from scipy.ndimage import convolve
        
        # 8-이웃의 defect 개수 카운팅
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],  # 중심은 0 (자신 제외)
                        [1, 1, 1]], dtype=np.float32)
        
        neighbor_count = convolve(wafer_map.astype(float), kernel, mode='constant', cval=0)
        
        # 이웃이 threshold 이상인 defect만 유지
        filtered = wafer_map * (neighbor_count >= neighbor_threshold)
        
        return filtered.astype(np.float32)
    
    def filter_batch_numpy(self, wafer_maps: np.ndarray) -> np.ndarray:
        """
        배치 웨이퍼맵 필터링 (numpy 병렬 처리)
        
        Args:
            wafer_maps: (B, H, W) 또는 (B, 1, H, W) batch
        
        Returns:
            filtered_maps: (B, H, W) 또는 (B, 1, H, W) 필터링된 배치
        """
        original_shape = wafer_maps.shape
        
        # (B, 1, H, W) -> (B, H, W)
        if len(original_shape) == 4 and original_shape[1] == 1:
            wafer_maps = wafer_maps.squeeze(1)
        
        batch_size = wafer_maps.shape[0]
        filtered_batch = np.zeros_like(wafer_maps)
        
        for i in range(batch_size):
            filtered_batch[i] = self.filter_single_map(wafer_maps[i])
        
        # 원래 shape 복원
        if len(original_shape) == 4:
            filtered_batch = filtered_batch[:, np.newaxis, :, :]
        
        return filtered_batch
    
    def filter_batch_torch(self, wafer_maps: torch.Tensor, 
                          device: Optional[str] = None) -> torch.Tensor:
        """
        배치 웨이퍼맵 필터링 (torch 텐서 입력/출력)
        
        Args:
            wafer_maps: (B, 1, H, W) torch tensor
            device: 결과 텐서의 device
        
        Returns:
            filtered_maps: (B, 1, H, W) 필터링된 배치
        """
        if device is None:
            device = wafer_maps.device
        
        # Torch -> Numpy
        wafer_maps_np = wafer_maps.cpu().numpy()
        
        # 필터링
        filtered_np = self.filter_batch_numpy(wafer_maps_np)
        
        # Numpy -> Torch
        filtered_torch = torch.from_numpy(filtered_np).float().to(device)
        
        return filtered_torch
    
    def filter_batch_torch_fast(self, wafer_maps: torch.Tensor) -> torch.Tensor:
        """
        더 빠른 배치 필터링 (torch 네이티브 연산 활용)
        Morphological operations만 사용 (component analysis 생략)
        
        Args:
            wafer_maps: (B, 1, H, W) torch tensor
        
        Returns:
            filtered_maps: (B, 1, H, W) 필터링된 배치
        """
        device = wafer_maps.device
        batch_size = wafer_maps.size(0)
        
        # Step 1: Opening (erosion + dilation)
        # Erosion
        kernel_erode = torch.ones(1, 1, self.opening_kernel_size, 
                                  self.opening_kernel_size).to(device)
        eroded = -F.max_pool2d(-wafer_maps, 
                               kernel_size=self.opening_kernel_size,
                               stride=1, 
                               padding=self.opening_kernel_size // 2)
        
        # Dilation
        kernel_dilate = torch.ones(1, 1, self.opening_kernel_size, 
                                   self.opening_kernel_size).to(device)
        opened = F.max_pool2d(eroded, 
                             kernel_size=self.opening_kernel_size,
                             stride=1, 
                             padding=self.opening_kernel_size // 2)
        
        # Step 2: Closing (dilation + erosion)
        # Dilation
        dilated = F.max_pool2d(opened, 
                              kernel_size=self.closing_kernel_size,
                              stride=1, 
                              padding=self.closing_kernel_size // 2)
        
        # Erosion
        closed = -F.max_pool2d(-dilated, 
                              kernel_size=self.closing_kernel_size,
                              stride=1, 
                              padding=self.closing_kernel_size // 2)
        
        # Step 3: Edge preservation
        # Sobel edge detection (torch 구현)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
        edges_x = F.conv2d(wafer_maps, sobel_x, padding=1)
        edges_y = F.conv2d(wafer_maps, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize edges
        edges = edges / (edges.max() + 1e-8)
        edge_mask = (edges > 0.3).float()
        edge_pixels = wafer_maps * edge_mask
        
        # Combine
        result = closed + self.edge_preserve_strength * edge_pixels
        result = (result > 0.5).float()
        
        return result


def visualize_filter_comparison(original: np.ndarray, 
                                filtered: np.ndarray,
                                title: str = "Filter Comparison"):
    """
    필터링 전후 비교 시각화
    
    Args:
        original: (H, W) 원본 맵
        filtered: (H, W) 필터링된 맵
        title: 플롯 제목
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original\n(Defects: {int(original.sum())})')
    axes[0].axis('off')
    
    # Filtered
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title(f'Filtered\n(Defects: {int(filtered.sum())})')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original - filtered)
    axes[2].imshow(diff, cmap='Reds')
    axes[2].set_title(f'Removed\n(Defects: {int(diff.sum())})')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# 사용 예시
if __name__ == "__main__":
    print("="*60)
    print("🧹 Wafer Map Filter Test")
    print("="*60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    test_map = np.zeros((128, 128), dtype=np.float32)
    
    # 1. 의미있는 패턴 (큰 덩어리)
    test_map[40:60, 40:80] = 1
    test_map[70:90, 50:70] = 1
    
    # 2. Random noise (1-2개 픽셀)
    noise_positions = np.random.randint(0, 128, (50, 2))
    for pos in noise_positions:
        test_map[pos[0], pos[1]] = 1
    
    print(f"\n원본 맵:")
    print(f"  전체 defect 수: {int(test_map.sum())}")
    
    # 필터 초기화
    filter_obj = WaferMapFilter(
        min_component_size=5,
        opening_kernel_size=3,
        closing_kernel_size=5,
        edge_preserve_strength=0.7
    )
    
    # 단일 맵 필터링
    filtered_map = filter_obj.filter_single_map(test_map)
    
    print(f"\n필터링 후:")
    print(f"  남은 defect 수: {int(filtered_map.sum())}")
    print(f"  제거된 defect 수: {int(test_map.sum() - filtered_map.sum())}")
    print(f"  제거 비율: {(1 - filtered_map.sum()/test_map.sum())*100:.1f}%")
    
    # 배치 테스트
    print(f"\n{'='*60}")
    print(f"배치 처리 테스트")
    print(f"{'='*60}")
    
    batch_size = 32
    test_batch = np.random.rand(batch_size, 1, 128, 128) > 0.95
    test_batch = test_batch.astype(np.float32)
    
    print(f"\n배치 크기: {batch_size}")
    print(f"평균 defect 수 (필터링 전): {test_batch.sum(axis=(1,2,3)).mean():.1f}")
    
    # Numpy 배치 필터링
    import time
    start = time.time()
    filtered_batch = filter_obj.filter_batch_numpy(test_batch)
    numpy_time = time.time() - start
    
    print(f"\nNumpy 배치 필터링:")
    print(f"  처리 시간: {numpy_time:.3f}초")
    print(f"  평균 defect 수 (필터링 후): {filtered_batch.sum(axis=(1,2,3)).mean():.1f}")
    
    # Torch 배치 필터링
    test_batch_torch = torch.from_numpy(test_batch).float()
    if torch.cuda.is_available():
        test_batch_torch = test_batch_torch.cuda()
        device_name = "GPU"
    else:
        device_name = "CPU"
    
    start = time.time()
    filtered_batch_torch = filter_obj.filter_batch_torch_fast(test_batch_torch)
    torch_time = time.time() - start
    
    print(f"\nTorch 배치 필터링 ({device_name}):")
    print(f"  처리 시간: {torch_time:.3f}초")
    print(f"  평균 defect 수 (필터링 후): {filtered_batch_torch.sum(dim=(1,2,3)).mean():.1f}")
    print(f"  속도 향상: {numpy_time/torch_time:.1f}x")
    
    print(f"\n{'='*60}")