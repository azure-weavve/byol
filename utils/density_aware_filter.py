import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
from utils.wafermap_filter import WaferMapFilter


class DensityAwareWaferMapFilter:
    """
    밀도 기반 적응형 웨이퍼맵 필터
    
    핵심 아이디어:
    - Defect 밀도가 높음 (>40%) → 의미있는 패턴 → 거의 필터링 안함
    - Defect 밀도가 중간 (10-40%) → 선택적 필터링
    - Defect 밀도가 낮음 (<10%) → Random noise 가능성 → 강하게 필터링
    """
    
    def __init__(self, 
                 density_thresholds=(0.12, 0.2, 0.35, 0.5),
                 filter_configs=None):
        """
        Args:
            density_thresholds: (low, medium, high) 밀도 임계값
            filter_configs: 각 밀도 구간별 필터 설정 (None이면 기본값 사용)
        """
        self.density_thresholds = density_thresholds
        
        if filter_configs is None:
            # 기본 설정 (밀도별 4단계)
            self.filter_configs = {
                'very_low': {  # density <= 0.12
                    'min_component_size': 1,
                    'opening_kernel_size': 0,
                    'closing_kernel_size': 0,
                    'edge_preserve_strength': 0.0
                },
                'low': {  # density <= 0.2
                    'min_component_size': 2,
                    'opening_kernel_size': 0,
                    'closing_kernel_size': 0,
                    'edge_preserve_strength': 1.0
                },
                'medium': {  # 0.2 < density <= 0.35
                    'min_component_size': 3,
                    'opening_kernel_size': 2,
                    'closing_kernel_size': 0,
                    'edge_preserve_strength': 0.95
                },
                'high': {  # 0.35 < density <= 0.5
                    'min_component_size': 5,
                    'opening_kernel_size': 3,
                    'closing_kernel_size': 0,
                    'edge_preserve_strength': 0.9
                },
                'very_high': {  # density > 0.5
                    'min_component_size': 7,
                    'opening_kernel_size': 4,
                    'closing_kernel_size': 0,
                    'edge_preserve_strength': 0.8
                }
            }
        else:
            self.filter_configs = filter_configs
    
    def calculate_density(self, wafer_map: np.ndarray) -> float:
        """
        Defect 밀도 계산
        
        Args:
            wafer_map: (H, W) binary map
        
        Returns:
            density: defect 픽셀 비율 (0~1)
        """
        total_pixels = wafer_map.size
        defect_pixels = wafer_map.sum()
        density = defect_pixels / total_pixels
        return density
    
    def select_filter_config(self, density: float) -> dict:
        """
        밀도에 따른 필터 설정 선택
        
        Args:
            density: defect 밀도 (0~1)
        
        Returns:
            filter_config: 선택된 필터 설정
        """
        verylow_th, low_th, med_th, high_th = self.density_thresholds
        
        if density > high_th:
            return self.filter_configs['very_high'], 'very_high'
        elif density > med_th:
            return self.filter_configs['high'], 'high'
        elif density > low_th:
            return self.filter_configs['medium'], 'medium'
        elif density > verylow_th:
            return self.filter_configs['low'], 'low'
        else:
            return self.filter_configs['low'], 'very_low'
    
    def filter_single_map(self, wafer_map: np.ndarray, 
                         verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """
        단일 웨이퍼맵 필터링 (밀도 기반 적응형)
        
        Args:
            wafer_map: (H, W) binary map
            verbose: 상세 정보 출력 여부
        
        Returns:
            filtered_map: (H, W) 필터링된 맵
            info: 필터링 정보 딕셔너리
        """
        # 빈 맵은 그대로 반환
        if wafer_map.sum() == 0:
            return wafer_map.copy(), {
                'density': 0.0,
                'strategy': 'empty',
                'original_count': 0,
                'filtered_count': 0,
                'removal_rate': 0.0
            }
        
        # 밀도 계산
        density = self.calculate_density(wafer_map)
        
        # 밀도에 따른 필터 설정 선택
        filter_config, strategy = self.select_filter_config(density)
        
        # 필터 적용
        filter_obj = WaferMapFilter(**filter_config)
        filtered_map = filter_obj.filter_single_map(wafer_map, strategy)
        
        # 통계
        original_count = int(wafer_map.sum())
        filtered_count = int(filtered_map.sum())
        removal_rate = (1 - filtered_count/original_count) * 100 if original_count > 0 else 0
        
        info = {
            'density': density,
            'strategy': strategy,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removal_rate': removal_rate,
            'filter_config': filter_config
        }
        
        if verbose:
            print(f"  Density: {density*100:.2f}% → Strategy: {strategy}")
            print(f"  {original_count} → {filtered_count} ({removal_rate:.1f}% removed)")
        
        return filtered_map, info
    
    def filter_batch_numpy(self, wafer_maps: np.ndarray, 
                          verbose: bool = False) -> Tuple[np.ndarray, list]:
        """
        배치 웨이퍼맵 필터링 (밀도 기반 적응형)
        
        Args:
            wafer_maps: (B, H, W) 또는 (B, 1, H, W) batch
            verbose: 상세 정보 출력 여부
        
        Returns:
            filtered_maps: (B, H, W) 또는 (B, 1, H, W) 필터링된 배치
            infos: 각 맵의 필터링 정보 리스트
        """
        original_shape = wafer_maps.shape
        
        # (B, 1, H, W) -> (B, H, W)
        if len(original_shape) == 4 and original_shape[1] == 1:
            wafer_maps = wafer_maps.squeeze(1)
        
        batch_size = wafer_maps.shape[0]
        filtered_batch = np.zeros_like(wafer_maps)
        infos = []
        
        if verbose:
            print(f"\n🔧 배치 필터링 (밀도 기반 적응형): {batch_size}개")
        
        for i in range(batch_size):
            filtered_batch[i], info = self.filter_single_map(
                wafer_maps[i], 
                verbose=verbose
            )
            infos.append(info)
        
        # 원래 shape 복원
        if len(original_shape) == 4:
            filtered_batch = filtered_batch[:, np.newaxis, :, :]
        
        if verbose:
            # 통계 요약
            strategies = [info['strategy'] for info in infos]
            from collections import Counter
            strategy_counts = Counter(strategies)
            
            print(f"\n📊 필터링 전략 분포:")
            for strategy, count in strategy_counts.items():
                print(f"   {strategy}: {count}개 ({count/batch_size*100:.1f}%)")
            
            avg_removal = np.mean([info['removal_rate'] for info in infos])
            print(f"\n평균 제거율: {avg_removal:.1f}%")
        
        return filtered_batch, infos
    
    def filter_batch_torch(self, wafer_maps: torch.Tensor, 
                          device: Optional[str] = None,
                          verbose: bool = False) -> Tuple[torch.Tensor, list]:
        """
        배치 웨이퍼맵 필터링 (torch 텐서 입력/출력)
        
        Args:
            wafer_maps: (B, 1, H, W) torch tensor
            device: 결과 텐서의 device
            verbose: 상세 정보 출력 여부
        
        Returns:
            filtered_maps: (B, 1, H, W) 필터링된 배치
            infos: 각 맵의 필터링 정보 리스트
        """
        if device is None:
            device = wafer_maps.device
        
        # Torch -> Numpy
        wafer_maps_np = wafer_maps.cpu().numpy()
        
        # 필터링
        filtered_np, infos = self.filter_batch_numpy(wafer_maps_np, verbose=verbose)
        
        # Numpy -> Torch
        filtered_torch = torch.from_numpy(filtered_np).float().to(device)
        
        return filtered_torch, infos


def visualize_density_aware_filtering(wafer_maps, num_samples=10):
    """
    밀도 기반 필터링 효과 시각화
    
    Args:
        wafer_maps: 웨이퍼맵 리스트
        num_samples: 시각화할 샘플 수
    """
    import matplotlib.pyplot as plt
    
    # 밀도 계산하여 다양한 샘플 선택
    densities = []
    for wm in wafer_maps:
        density = wm.sum() / wm.size
        densities.append(density)
    
    densities = np.array(densities)
    sorted_indices = np.argsort(densities)
    
    # 밀도가 다양한 샘플 선택
    step = len(sorted_indices) // num_samples
    selected_indices = sorted_indices[::step][:num_samples]
    
    # 필터 생성
    density_filter = DensityAwareWaferMapFilter()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 10))
    
    for i, idx in enumerate(selected_indices):
        wm = wafer_maps[idx]
        
        # 필터링
        filtered, info = density_filter.filter_single_map(wm, verbose=False)
        
        # 원본
        axes[0, i].imshow(wm, cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDensity: {info["density"]*100:.1f}%\n{info["original_count"]} defects', 
                            fontsize=9)
        axes[0, i].axis('off')
        
        # 필터링 결과
        axes[1, i].imshow(filtered, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'Filtered ({info["strategy"]})\n{info["filtered_count"]} defects\n({100-info["removal_rate"]:.1f}% kept)', 
                            fontsize=9)
        axes[1, i].axis('off')
        
        # 제거된 부분
        diff = wm - filtered
        axes[2, i].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[2, i].set_title(f'Removed\n{int(diff.sum())} ({info["removal_rate"]:.1f}%)', 
                            fontsize=9)
        axes[2, i].axis('off')
    
    plt.suptitle('Density-Aware Adaptive Filtering Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/density_aware_filtering_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 저장: /home/claude/density_aware_filtering_results.png")
    plt.close()


# 사용 예시
if __name__ == "__main__":
    print("="*60)
    print("🧹 Density-Aware Wafer Map Filter Test")
    print("="*60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    test_maps = []
    
    # 1. Very low density (random noise)
    sparse = np.zeros((128, 128), dtype=np.float32)
    for _ in range(20):
        x, y = np.random.randint(0, 128, 2)
        sparse[x:x+2, y:y+2] = 1
    test_maps.append(('Sparse', sparse))
    
    # 2. Low density
    low = np.zeros((128, 128), dtype=np.float32)
    low[60:68, 60:68] = 1
    for _ in range(15):
        x, y = np.random.randint(0, 128, 2)
        low[x, y] = 1
    test_maps.append(('Low', low))
    
    # 3. Medium density
    med = np.zeros((128, 128), dtype=np.float32)
    med[40:80, 40:80] = 1
    test_maps.append(('Medium', med))
    
    # 4. High density
    high = np.zeros((128, 128), dtype=np.float32)
    high[20:110, 20:110] = 1
    for i in range(30, 100, 10):
        high[i:i+2, 30:100] = 0
    test_maps.append(('High', high))
    
    # 5. Very high density
    very_high = np.zeros((128, 128), dtype=np.float32)
    very_high[10:118, 10:118] = 1
    test_maps.append(('Very High', very_high))
    
    # 밀도 기반 필터 적용
    density_filter = DensityAwareWaferMapFilter()
    
    print(f"\n{'='*60}")
    print(f"밀도 기반 적응형 필터링 테스트")
    print(f"{'='*60}\n")
    
    for name, wm in test_maps:
        print(f"{name} Density Pattern:")
        filtered, info = density_filter.filter_single_map(wm, verbose=True)
        print()
    
    print(f"{'='*60}\n")