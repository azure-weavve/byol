import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_closing, generate_binary_structure, distance_transform_edt, gaussian_filter, grey_dilation

class RegionAwareWaferMapFilter:
    """각도 + 거리 기반 2D Region 분류 필터 (NumPy & PyTorch 지원)"""
    
    def __init__(self, 
                 n_sectors=12,
                 n_rings=3,
                 sector_density_threshold=0.03,
                 closing_kernel_size=15,
                 min_region_size=30):
        self.n_sectors = n_sectors
        self.n_rings = n_rings
        self.sector_density_threshold = sector_density_threshold
        self.closing_kernel_size = closing_kernel_size
        self.min_region_size = min_region_size
    
    # 🆕 Helper functions for type conversion
    @staticmethod
    def _to_numpy(x):
        """Tensor or ndarray → numpy"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    @staticmethod
    def _to_same_type(result, reference):
        """결과를 reference와 같은 타입으로 변환"""
        if isinstance(reference, torch.Tensor):
            return torch.from_numpy(result).to(reference.device).type(reference.dtype)
        return result
    
    @staticmethod
    def _astype_float(x):
        """NumPy/Torch 모두 지원하는 float 변환"""
        if isinstance(x, torch.Tensor):
            return x.float()
        return x.astype(np.float32)
    
    def create_region_map(self, wafer_map):
        """
        웨이퍼를 (각도 × 거리) 그리드로 분할
        
        Args:
            wafer_map: (H, W) numpy array or torch.Tensor
        
        Returns:
            region_indices: (H, W) numpy array
            wafer_mask: (H, W) numpy array
        """
        # 🔴 NumPy로 변환
        wafer_map_np = self._to_numpy(wafer_map)
        
        H, W = wafer_map_np.shape
        center_y, center_x = H / 2, W / 2
        
        # 각도 계산
        y_grid, x_grid = np.ogrid[:H, :W]
        angles = np.arctan2(y_grid - center_y, x_grid - center_x)
        angles_normalized = (angles + np.pi) / (2 * np.pi)
        sector_indices = (angles_normalized * self.n_sectors).astype(int) % self.n_sectors
        
        # 거리 계산
        distances = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        max_radius = min(H, W) / 2 - 5
        wafer_mask = distances <= max_radius
        
        # Ring 계산
        normalized_distance = distances / max_radius
        ring_indices = (normalized_distance * self.n_rings).astype(int)
        ring_indices = np.clip(ring_indices, 0, self.n_rings - 1)
        
        # Region ID
        region_indices = sector_indices * self.n_rings + ring_indices
        region_indices = region_indices * wafer_mask
        
        return region_indices, wafer_mask
    
    def detect_clustering_regions(self, wafer_map):
        """
        각도 + 거리 기반 몰림 영역 감지
        
        Args:
            wafer_map: (H, W) numpy array or torch.Tensor
        
        Returns:
            region_mask: (H, W) numpy array
            region_info: list of dict
        """
        # 🔴 NumPy로 변환
        wafer_map_np = self._to_numpy(wafer_map)
        
        region_indices, wafer_mask = self.create_region_map(wafer_map_np)
        
        total_regions = self.n_sectors * self.n_rings
        region_mask = np.zeros_like(wafer_map_np)
        region_info = []
        
        for region_id in range(total_regions):
            region_pixels_mask = (region_indices == region_id) & wafer_mask
            region_pixels = region_pixels_mask.sum()
            
            if region_pixels == 0:
                continue
            
            region_defects = (wafer_map_np * region_pixels_mask).sum()
            region_density = region_defects / region_pixels
            
            sector_id = region_id // self.n_rings
            ring_id = region_id % self.n_rings
            
            region_info.append({
                'region_id': region_id,
                'sector_id': sector_id,
                'ring_id': ring_id,
                'density': region_density,
                'defects': region_defects,
                'pixels': region_pixels
            })
            
            if region_density >= self.sector_density_threshold and \
               region_defects >= self.min_region_size:
                region_mask[region_pixels_mask] = 1
        
        return region_mask, region_info
    
    def fill_clustering_regions(self, wafer_map, region_mask, 
                           fill_strength=1.0, decay_power=2.0,
                           blend_sigma=1.0):
        """
        원본 + 채운 부분을 하나의 덩어리로 blend

        Args:
            fill_strength: 채우기 강도 (0.5~1.5)
            decay_power: 거리 감쇠 지수 (0.3~1.0)
            blend_sigma: 전체 덩어리 smoothing 강도 (1.0~3.0)
        """
        original_type = wafer_map
        wafer_map_np = self._to_numpy(wafer_map)
        region_mask_np = self._to_numpy(region_mask)

        if region_mask_np.sum() == 0:
            return original_type

        clustering_defects = wafer_map_np * region_mask_np
        if clustering_defects.sum() == 0:
            return original_type

        # 1. 원본 불량 마스크
        original_mask = (clustering_defects > 0).astype(np.float32)

        # 2. Closing으로 최종 영역 결정
        struct = generate_binary_structure(2, 2)
        struct = ndimage.iterate_structure(struct, self.closing_kernel_size // 2)
        filled_mask = binary_closing(original_mask > 0, structure=struct).astype(np.float32)

        # 3. 새로 채워지는 영역
        newly_filled = (filled_mask > 0) & (original_mask == 0)

        if newly_filled.sum() == 0:
            return original_type

        # 4. Max intensity
        max_intensity = np.percentile(clustering_defects[original_mask > 0], 95)

        # 5. Distance transform
        distance = distance_transform_edt(original_mask == 0)

        # 6. Probability 계산
        max_fill_distance = self.closing_kernel_size / 2
        normalized_distance = np.clip(distance / max_fill_distance, 0, 1)
        probability = 1 - normalized_distance ** decay_power
        probability = np.clip(probability * fill_strength, 0, 1)

        # 7. 채운 부분의 값 생성
        filled_values = max_intensity * probability

        # 8. 🔴 원본 + 채운 부분을 하나의 맵으로 합치기
        combined_map = wafer_map_np.copy()
        combined_map[newly_filled] = filled_values[newly_filled]

        # 9. 🔴 전체 덩어리에 Gaussian blur 적용 (하나로 blend!)
        # filled_mask 영역 전체를 blur
        blurred_full = gaussian_filter(combined_map, sigma=blend_sigma)

        # 10. 🔴 덩어리 영역에만 적용 (외부는 원본 유지)
        result = wafer_map_np.copy()
        result[filled_mask > 0] = blurred_full[filled_mask > 0]

        return self._to_same_type(result, original_type)
    
    def filter_single_map(self, wafer_map, verbose=False):
        """
        전체 파이프라인
        
        Args:
            wafer_map: (H, W) numpy array or torch.Tensor
            verbose: bool
        
        Returns:
            filled_map: same type as wafer_map
        """
        # 🔴 원본 타입 기억
        original_type = wafer_map
        
        # 몰림 영역 감지 (내부에서 numpy로 변환)
        region_mask, region_info = self.detect_clustering_regions(wafer_map)
        
        if verbose:
            ring_names = ['Inner', 'Middle', 'Outer'] if self.n_rings == 3 else \
                         [f'Ring{i}' for i in range(self.n_rings)]
            
            print(f"\n🔍 Region Analysis ({self.n_sectors} sectors × {self.n_rings} rings):")
            
            detected_regions = []
            for info in region_info:
                if info['density'] >= self.sector_density_threshold and \
                   info['defects'] >= self.min_region_size:
                    sector = info['sector_id']
                    ring = info['ring_id']
                    detected_regions.append(
                        f"Sector {sector:2d}-{ring_names[ring]:6s}: "
                        f"density={info['density']:.4f} ({int(info['defects']):3d}/{info['pixels']:4d}) ✅"
                    )
            
            if detected_regions:
                for msg in detected_regions:
                    print(f"   {msg}")
            else:
                print(f"   ⚠️  No clustering regions detected")
        
        if region_mask.sum() == 0:
            return original_type
        
        # 몰림 영역 채우기 (내부에서 타입 변환)
        filled = self.fill_clustering_regions(wafer_map, region_mask)
        
        if verbose:
            # 🔴 NumPy로 변환해서 통계 계산
            wafer_np = self._to_numpy(wafer_map)
            filled_np = self._to_numpy(filled)
            
            original_defects = wafer_np.sum()
            filled_defects = filled_np.sum()
            added = filled_defects - original_defects
            
            print(f"\n📊 Filling Results:")
            print(f"   Original: {int(original_defects):4d} defects")
            print(f"   Filled:   {int(filled_defects):4d} defects")
            print(f"   Added:    {int(added):4d} ({added/original_defects*100:+.1f}%)")
        
        return filled