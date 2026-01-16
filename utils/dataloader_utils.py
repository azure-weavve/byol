import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from utils.wafermap_filter import WaferMapFilter
from utils.density_aware_filter import DensityAwareWaferMapFilter
from utils.region_aware_filter import RegionAwareWaferMapFilter


def convert_to_multichannel(wafer_map, n_categories=10):
    """
    Convert wafer map to multi-channel format
    
    Args:
        wafer_map: numpy array
                   - Binary: (H, W) with values {0, 1}
                   - Category: (n_cat, H, W) with values {0, 1} for each category
        n_categories: number of categories (default: 10)
    
    Returns:
        multi_channel: (n_categories+1, H, W) numpy array
                      channel[0] = binary (spatial pattern)
                      channel[1:n+1] = category-specific
    """
    # Detect input format
    if len(wafer_map.shape) == 2:
        # Binary data: (H, W)
        H, W = wafer_map.shape
        multi_channel = np.zeros((n_categories + 1, H, W), dtype=np.float32)
        
        # Channel 0: binary map
        multi_channel[0] = wafer_map.astype(np.float32)
        
        # Channel 1-n: all zeros (no category info)
        # Already initialized as zeros
        
    elif len(wafer_map.shape) == 3:
        # Category data: (n_cat, H, W)
        n_cat, H, W = wafer_map.shape
        multi_channel = np.zeros((n_categories + 1, H, W), dtype=np.float32)
        
        # Channel 0: binary (any category > 0)
        multi_channel[0] = (wafer_map.sum(axis=0) > 0).astype(np.float32)
        
        # Channel 1-n: category-specific
        # Copy existing categories
        multi_channel[1:n_cat+1] = wafer_map.astype(np.float32)
        
        # If n_cat < n_categories, remaining channels stay 0
        
    else:
        raise ValueError(f"Unexpected wafer_map shape: {wafer_map.shape}")
    
    return multi_channel

def detect_n_categories(data_configs):
    """
    Automatically detect maximum number of categories from data
    """
    max_categories = 0
    
    for config in data_configs:
        file_path = config["path"]
        
        if not os.path.exists(file_path):
            continue
        
        try:
            data = np.load(file_path, allow_pickle=True)
            maps = data['maps']
            
            # Check first sample
            if len(maps) > 0:
                sample = maps[0]
                
                # Convert to array if needed
                if not isinstance(sample, np.ndarray):
                    sample = np.array(sample)
                
                # Category data: 3D (n_cat, H, W)
                if len(sample.shape) == 3:
                    n_cat = sample.shape[0]
                    max_categories = max(max_categories, n_cat)
                # Binary data: 2D (H, W)
                elif len(sample.shape) == 2:
                    pass  # Binary, no categories
                    
        except Exception as e:
            print(f"⚠️  Failed to detect categories from {file_path}: {e}")
            continue
    
    print(f"✅ Detected maximum categories: {max_categories}")
    return max_categories



def prepare_clean_data(data_configs, use_filter=True, filter_params=None, 
                       use_density_aware=False, use_region_aware=False):
    """
    여러 제품 데이터를 로드하고 완전히 정리 + 필터링 + Multi-channel 변환

    Args:
        data_configs: [
            {"path": "path1.npz", "name": "product1"},
            {"path": "path2.npz", "name": "mixed_products"},
        ]
        use_filter: 필터링 적용 여부
        filter_params: 필터 파라미터 딕셔너리
        use_density_aware: True면 밀도 기반 적응형 필터 사용 (권장!)
        use_region_aware: True면 region-aware 필터 사용

    Returns:
        clean_maps: List of (n_categories+1, H, W) arrays
        clean_labels: List of labels
        info: List of filter info dicts
    """

    print("="*60)
    
    # 1. 카테고리 개수 자동 감지
    n_categories = detect_n_categories(data_configs)
    print(f"🔍 Auto-detected categories: {n_categories}")
    
    mode_str = "밀도 기반 적응형" if use_density_aware else "일반"
    print(f"🧹 데이터 완전 정리 시작" + (f" ({mode_str} 필터링 포함)" if use_filter else ""))
    print(f"📊 Multi-channel format: {n_categories + 1} channels")
    print("="*60)
    
    # 필터 초기화
    if use_filter:
        if use_density_aware:
            filter_obj = DensityAwareWaferMapFilter()
        else:
            if filter_params is None:
                filter_params = {
                    'min_component_size': 5,
                    'opening_kernel_size': 1,
                    'closing_kernel_size': 5,
                    'edge_preserve_strength': 0.9
                }
            filter_obj = WaferMapFilter(**filter_params)

    all_clean_maps = []
    all_clean_labels = []
    all_info = []

    for config in data_configs:
        file_path = config["path"]
        name = config.get("name", "unknown")

        print(f"\n📁 {name} 로딩 중: {file_path}")

        if not os.path.exists(file_path):
            print(f"⚠️  파일 없음: {file_path}")
            continue

        try:
            # 데이터 로드
            data = np.load(file_path, allow_pickle=True)
            maps = data['maps']
            labels = data['ids'] if 'ids' in data else data['labels']

            print(f"   원본: {len(maps)}개")

            # 개별 정리
            clean_maps = []
            clean_labels = []
            filtered_count = 0
            info_list = []

            for i, (wm, label) in enumerate(zip(maps, labels)):
                try:
                    # wm을 numpy array로 변환
                    if not isinstance(wm, np.ndarray):
                        wm = np.array(wm.tolist() if hasattr(wm, 'tolist') else wm, dtype=np.float32)
                    elif wm.dtype == object:
                        wm = np.array(wm.tolist(), dtype=np.float32)
                    else:
                        wm = wm.astype(np.float32)
                    
                    # 검증: 2D (binary) 또는 3D (category)
                    if len(wm.shape) == 2:
                        # Binary data (H, W)
                        H, W = wm.shape
                        if H == 0 or W == 0:
                            continue
                        
                    elif len(wm.shape) == 3:
                        # Category data (n_cat, H, W)
                        n_cat, H, W = wm.shape
                        if n_cat == 0 or H == 0 or W == 0:
                            continue
                    else:
                        # Invalid shape
                        continue
                    
                    # NaN, Inf 처리
                    if np.any(np.isnan(wm)) or np.any(np.isinf(wm)):
                        wm = np.nan_to_num(wm, nan=0.0, posinf=1.0, neginf=0.0)

                    # 정규화 (0-1 range)
                    if wm.max() > 1.0:
                        wm = wm / wm.max()
                    
                    # 🔹 Multi-channel 변환
                    multi_channel_wm = convert_to_multichannel(wm, n_categories)
                    
                    # 🔹 필터링 적용 (channel 0에만)
                    if use_filter and multi_channel_wm[0].sum() > 0:
                        original_defects = multi_channel_wm[0].sum()
                        
                        if use_density_aware:
                            filtered_ch0, info = filter_obj.filter_single_map(multi_channel_wm[0])
                            multi_channel_wm[0] = filtered_ch0
                        else:
                            multi_channel_wm[0] = filter_obj.filter_single_map(multi_channel_wm[0])
                            info = None
                        
                        filtered_defects = multi_channel_wm[0].sum()
                        
                        # 너무 많이 제거되면 스킵
                        if filtered_defects < original_defects * 0.2:
                            continue
                        
                        if filtered_defects < original_defects:
                            filtered_count += 1
                    
                    # 최종 검증
                    assert isinstance(multi_channel_wm, np.ndarray)
                    assert multi_channel_wm.dtype == np.float32
                    assert len(multi_channel_wm.shape) == 3  # (C, H, W)
                    assert multi_channel_wm.shape[0] == n_categories + 1
                    assert not np.any(np.isnan(multi_channel_wm))

                    clean_maps.append(multi_channel_wm)
                    clean_labels.append(label)
                    info_list.append(info)

                except Exception as e:
                    # 문제 있는 데이터는 조용히 건너뜀
                    continue

            success_rate = len(clean_maps) / len(maps) * 100
            print(f"   정리됨: {len(clean_maps)}개 ({success_rate:.1f}%)")
            print(f"   Shape: ({n_categories + 1}, H, W)")
            if use_filter and filtered_count > 0:
                print(f"   필터링됨: {filtered_count}개 ({filtered_count/len(clean_maps)*100:.1f}%)")

            all_clean_maps.extend(clean_maps)
            all_clean_labels.extend(clean_labels)
            all_info.extend(info_list)

        except Exception as e:
            print(f"❌ {name} 로딩 실패: {e}")
            continue

    print(f"\n✅ 전체 정리 완료: {len(all_clean_maps)}개")
    print(f"   Final shape per sample: ({n_categories + 1}, H, W)")
    return all_clean_maps, all_clean_labels, all_info


class MultiSizeWaferDataset(Dataset):
    """인덱스를 함께 반환하는 Dataset"""

    def __init__(self, wafer_maps, labels, target_size=(128, 128), 
                 use_filter=False, filter_on_the_fly=False, filter_params=None,
                 use_density_aware=False, is_training=False, use_augmentation=False):
        """
        Args:
            wafer_maps: 웨이퍼맵 리스트
            labels: 라벨 리스트
            target_size: 리사이즈 타겟 크기
            use_filter: 필터링 사용 여부
            filter_on_the_fly: True면 __getitem__ 시마다 필터링 (느림, 메모리 절약)
                              False면 초기화 시 모두 필터링 (빠름, 메모리 사용)
            filter_params: 필터 파라미터
            use_density_aware: True면 밀도 기반 적응형 필터 사용 (권장!)
        """
        self.wafer_maps = []
        self.labels = []
        self.original_indices = []
        self.target_size = target_size
        self.use_filter = use_filter
        self.filter_on_the_fly = filter_on_the_fly
        self.use_density_aware = use_density_aware
        self.is_training = is_training
        self.use_augmentation = use_augmentation

        # 필터 초기화
        if use_filter:
            if use_density_aware:
                self.filter_obj = DensityAwareWaferMapFilter()
            else:
                if filter_params is None:
                    filter_params = {
                        'min_component_size': 5,
                        'opening_kernel_size': 1,
                        'closing_kernel_size': 5,
                        'edge_preserve_strength': 0.9
                    }
                self.filter_obj = WaferMapFilter(**filter_params)
        
        print(f"🛡️  Dataset 생성 중...")
        if use_filter and not filter_on_the_fly:
            print(f"   사전 필터링 적용 중...")

        for idx, (wm, label) in enumerate(zip(wafer_maps, labels)):
            # 🔴 Shape 검증 수정: (C, H, W) 형식
            if (isinstance(wm, np.ndarray) and
                wm.dtype == np.float32 and
                len(wm.shape) == 3 and      # (C, H, W)
                wm.shape[0] > 0 and         # C > 0
                wm.shape[1] > 0 and         # H > 0
                wm.shape[2] > 0):           # W > 0

                # 사전 필터링 (filter_on_the_fly=False인 경우)
                # Channel 0에만 적용
                if use_filter and not filter_on_the_fly:
                    if wm[0].sum() > 0:
                        original_defects = wm[0].sum()
                        
                        if use_density_aware:
                            wm[0], info = self.filter_obj.filter_single_map(wm[0])
                        else:
                            wm[0] = self.filter_obj.filter_single_map(wm[0])
                        
                        # 너무 많이 제거되면 스킵
                        if wm[0].sum() < original_defects * 0.2:
                            continue
                
                self.wafer_maps.append(wm)
                self.labels.append(label)
                self.original_indices.append(idx)

        print(f"   최종 Dataset: {len(self.wafer_maps)}개")

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx]  # (C, H, W) - already multi-channel
        label = self.labels[idx]
        original_idx = self.original_indices[idx]

        # On-the-fly 필터링 (filter_on_the_fly=True인 경우)
        # Note: 이미 prepare_clean_data에서 필터링 했으므로 보통은 skip
        if self.use_filter and self.filter_on_the_fly:
            # Channel 0에만 필터링 적용
            if wafer_map[0].sum() > 0:
                if self.use_density_aware:
                    wafer_map[0], _ = self.filter_obj.filter_single_map(wafer_map[0])
                else:
                    wafer_map[0] = self.filter_obj.filter_single_map(wafer_map[0])

        # 안전한 전처리
        # wafer_map: (C, H, W) numpy array → (C, target_H, target_W) tensor
        tensor = torch.tensor(wafer_map, dtype=torch.float32)  # (C, H, W)
        
        # Resize
        # F.interpolate expects (B, C, H, W), so add batch dim
        tensor_4d = tensor.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(tensor_4d, size=self.target_size, mode='bilinear', align_corners=False)
        resized = resized.squeeze(0)  # (C, target_H, target_W)

        # 🔴 Augmentation 적용 (training 시에만!)
        if self.is_training and self.use_augmentation:
            resized_aug = self._apply_augmentation(resized)
        else:
            resized_aug = None

        return resized, resized_aug, label, original_idx
    
    def _apply_augmentation(self, tensor):
        """
        회전 불변성을 위한 Augmentation
        D4 Dihedral group의 8가지 변환 중 하나를 균등하게 선택
        
        Args:
            tensor: (C, H, W) - multi-channel
        
        Returns:
            tensor: (C, H, W) - augmented
        """
        # 8가지 변환 중 하나를 균등하게 선택
        transform_id = torch.randint(0, 8, (1,)).item()
        
        if transform_id == 0:
            return tensor  # Identity (변환 없음)
        elif transform_id == 1:
            return torch.rot90(tensor, 1, dims=[1, 2])  # 90도 회전
        elif transform_id == 2:
            return torch.rot90(tensor, 2, dims=[1, 2])  # 180도 회전
        elif transform_id == 3:
            return torch.rot90(tensor, 3, dims=[1, 2])  # 270도 회전
        elif transform_id == 4:
            return torch.flip(tensor, dims=[2])  # 좌우 반전
        elif transform_id == 5:
            return torch.flip(tensor, dims=[1])  # 상하 반전
        elif transform_id == 6:
            # 90도 회전 + 좌우 반전 (대각선 대칭)
            return torch.flip(torch.rot90(tensor, 1, dims=[1, 2]), dims=[2])
        elif transform_id == 7:
            # 90도 회전 + 상하 반전 (다른 대각선 대칭)
            return torch.flip(torch.rot90(tensor, 1, dims=[1, 2]), dims=[1])


def collate_fn(batch):
    """인덱스를 포함한 collate 함수
    + 문제 데이터를 자동으로 필터링하는 collate 함수
    + 빈 맵(모두 0)도 제거
    + Multi-channel 지원"""

    safe_data = []
    safe_data_aug = []
    safe_labels = []
    safe_indices = []
    has_aug = True

    for data, data_aug, label, original_idx in batch:
        try:
            # Shape 검증: (C, H, W)
            if (isinstance(data, torch.Tensor) and
                data.dtype == torch.float32 and
                len(data.shape) == 3 and  # (C, H, W)
                data.shape[0] > 0 and     # C > 0
                data.shape[1] > 0 and     # H > 0
                data.shape[2] > 0 and     # W > 0
                data.sum() > 0):          # Not all zeros

                safe_data.append(data)
                safe_data_aug.append(data_aug)
                safe_labels.append(label)
                safe_indices.append(original_idx)

                # 첫 번째 샘플로 augmentation 여부 판단
                if len(safe_data) == 1:
                    has_aug = (data_aug is not None)

        except:
            continue

    if len(safe_data) == 0:
        # 모든 샘플이 문제인 경우 더미 배치 반환
        # Multi-channel dummy
        dummy = torch.zeros((1, 11, 128, 128), dtype=torch.float32)  # 11 channels
        return dummy, None, ["dummy"], [0]

    batch_data = torch.stack(safe_data)
    
    if has_aug:
        batch_data_aug = torch.stack(safe_data_aug)
    else:
        batch_data_aug = None

    return batch_data, batch_data_aug, safe_labels, safe_indices



def create_dataloaders(wafer_maps, labels, batch_size=64, target_size=(128, 128), test_size=0.2, 
                        use_filter=True, filter_on_the_fly=False, filter_params=None, 
                        use_density_aware=False, use_augmentation=False):
    
    print("\n🔧 안전한 DataLoader 생성")
    print("="*40)

    if use_filter:
        if use_density_aware:
            mode = "Density-Aware (밀도 기반 적응형)"
        elif filter_on_the_fly:
            mode = "On-the-fly"
    else:
        mode = "Pre-filtering"
    print(f"   필터링 모드: {mode}")

    # 🔹 train/valid 분할을 먼저 수행
    train_indices, valid_indices = train_test_split(
        range(len(wafer_maps)), test_size=test_size, random_state=42
    )
    
    # 🔹 분할된 데이터로 train/valid 데이터 생성
    train_maps = [wafer_maps[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    valid_maps = [wafer_maps[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    # 🔹 별도의 dataset 객체 생성
    train_dataset = MultiSizeWaferDataset(
        train_maps, train_labels, 
        target_size=target_size,
        use_filter=use_filter,
        filter_on_the_fly=filter_on_the_fly,
        filter_params=filter_params,
        use_density_aware=use_density_aware,
        is_training=True,  # 🔹 train은 True
        use_augmentation=use_augmentation
    )
    
    valid_dataset = MultiSizeWaferDataset(
        valid_maps, valid_labels, 
        target_size=target_size,
        use_filter=use_filter,
        filter_on_the_fly=filter_on_the_fly,
        filter_params=filter_params,
        use_density_aware=use_density_aware,
        is_training=False,  # 🔹 valid은 False
        use_augmentation=False  # 🔹 항상 False
    )

    print(f"   Train: {len(train_dataset)}개 (Augmentation: {use_augmentation})")
    print(f"   Valid: {len(valid_dataset)}개 (Augmentation: False Fixed)")

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"   Train 배치: {len(train_loader)}개 / Valid 배치: {len(valid_loader)}개")

    return train_loader, valid_loader