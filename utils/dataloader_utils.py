import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from utils.wafermap_filter import WaferMapFilter
from utils.density_aware_filter import DensityAwareWaferMapFilter
from utils.region_aware_filter import RegionAwareWaferMapFilter


def convert_to_multichannel(wafer_maps, n_categories=12):
    """
    정수 라벨 배열을 multi-channel one-hot 형식으로 변환 (배치 벡터화)
    
    Args:
        wafer_maps: numpy array
                   - 단일: (H, W) 정수 배열 (값 0~12)
                   - 배치: (n, H, W) 정수 배열 (값 0~12)
        n_categories: 불량 카테고리 개수 (기본값: 12)
                     0은 non-wafer + good chip이므로 제외
    
    Returns:
        multi_channel: numpy array
                      - 단일 입력: (n_categories+1, H, W)
                      - 배치 입력: (n, n_categories+1, H, W)
                      
                      channel[0] = 불량 위치 (값 > 0)
                      channel[1~12] = 각 카테고리별 위치 (값 == k)
    """
    # 단일 웨이퍼 처리 (H, W) → 배치 형태로 변환
    single_input = False
    if len(wafer_maps.shape) == 2:
        single_input = True
        wafer_maps = wafer_maps[np.newaxis, ...]  # (1, H, W)
    
    n, H, W = wafer_maps.shape
    n_channels = n_categories + 1  # 13 channels (0: spatial, 1-12: categories)
    
    # 결과 배열 초기화
    multi_channel = np.zeros((n, n_channels, H, W), dtype=np.float32)
    
    # Channel 0: 불량 위치 (값 > 0인 모든 위치)
    multi_channel[:, 0, :, :] = (wafer_maps > 0).astype(np.float32)
    
    # Channel 1~12: 각 카테고리별 one-hot
    for k in range(1, n_categories + 1):
        multi_channel[:, k, :, :] = (wafer_maps == k).astype(np.float32)
    
    # 단일 입력이었으면 배치 차원 제거
    if single_input:
        multi_channel = multi_channel[0]  # (13, H, W)
    
    return multi_channel

def detect_n_categories(data_configs):
    """
    데이터 파일들을 스캔하여 최대 카테고리 개수 자동 감지 (값 기반)
    
    Args:
        data_configs: [{"path": "...", "name": "..."}, ...]
    
    Returns:
        max_category: 최대 카테고리 번호 (예: 12)
                     0은 non-wafer + good chip이므로 카테고리에서 제외
    """
    max_category = 0
    
    for config in data_configs:
        file_path = config["path"]
        
        if not os.path.exists(file_path):
            print(f"⚠️  파일 없음: {file_path}")
            continue
        
        try:
            data = np.load(file_path, allow_pickle=True)
            maps = data['maps']
            
            if len(maps) == 0:
                continue
            
            # 샘플링하여 최대값 확인 (전체 스캔은 느릴 수 있음)
            n_samples = min(100, len(maps))
            sample_indices = np.linspace(0, len(maps)-1, n_samples, dtype=int)
            
            for idx in sample_indices:
                sample = maps[idx]
                
                if not isinstance(sample, np.ndarray):
                    sample = np.array(sample)
                
                sample_max = int(sample.max())
                max_category = max(max_category, sample_max)
            
            print(f"✅ {config.get('name', 'unknown')}: 최대 카테고리 = {max_category}")
                    
        except Exception as e:
            print(f"⚠️  {file_path} 감지 실패: {e}")
            continue
    
    print(f"📊 감지된 최대 카테고리: {max_category}")
    return max_category



def prepare_clean_data(data_configs, use_filter=True, filter_params=None, 
                       use_density_aware=False, use_region_aware=False):
    """
    여러 제품 데이터를 로드하고 완전히 정리 + 필터링 + Multi-channel 변환

    Args:
        data_configs: [{"path": "...", "name": "..."}, ...]
        use_filter: 필터링 적용 여부
        filter_params: 필터 파라미터 딕셔너리
        use_density_aware: True면 밀도 기반 적응형 필터 사용
        use_region_aware: True면 region-aware 필터 사용

    Returns:
        clean_maps: List of (n_categories+1, H, W) arrays
        clean_labels: List of labels
        info: List of filter info dicts
    """

    print("="*60)
    
    # 1. 카테고리 개수 자동 감지 (값 기반)
    n_categories = detect_n_categories(data_configs)
    n_channels = n_categories + 1  # 0: spatial pattern, 1~n: categories
    
    mode_str = "밀도 기반 적응형" if use_density_aware else "일반"
    print(f"🧹 데이터 완전 정리 시작" + (f" ({mode_str} 필터링 포함)" if use_filter else ""))
    print(f"📊 Multi-channel format: {n_channels} channels (1 spatial + {n_categories} categories)")
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

            # ========== 배치 변환 ==========
            # maps를 numpy array로 변환
            if not isinstance(maps, np.ndarray):
                maps = np.array(maps)
            
            # object dtype 처리
            if maps.dtype == object:
                # 각 웨이퍼의 shape이 다를 수 있으므로 개별 처리 필요
                maps_list = [np.array(m, dtype=np.float32) for m in maps]
            else:
                maps = maps.astype(np.float32)
                maps_list = None  # 배치 처리 가능
            
            # 배치 처리 가능한 경우 (모든 웨이퍼 shape 동일)
            if maps_list is None:
                # (n, H, W) → (n, 13, H, W) 배치 변환
                multi_channel_maps = convert_to_multichannel(maps, n_categories=n_categories)
                
                print(f"   배치 변환 완료: {multi_channel_maps.shape}")
                
                # 개별 필터링 및 검증
                for i in range(len(multi_channel_maps)):
                    wm = multi_channel_maps[i]  # (13, H, W)
                    label = labels[i]
                    
                    # NaN, Inf 처리
                    if np.any(np.isnan(wm)) or np.any(np.isinf(wm)):
                        wm = np.nan_to_num(wm, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # 필터링 적용 (channel 0에만)
                    info = None
                    if use_filter and wm[0].sum() > 0:
                        if use_density_aware:
                            wm[0], info = filter_obj.filter_single_map(wm[0])
                        else:
                            wm[0] = filter_obj.filter_single_map(wm[0])
                    
                    all_clean_maps.append(wm)
                    all_clean_labels.append(label)
                    all_info.append(info)
            
            else:
                # shape이 다른 경우 개별 처리
                print(f"   ⚠️  웨이퍼 크기가 다름 - 개별 처리")
                
                for i, (wm, label) in enumerate(zip(maps_list, labels)):
                    try:
                        # 2D 검증
                        if len(wm.shape) != 2 or wm.shape[0] == 0 or wm.shape[1] == 0:
                            continue
                        
                        # 단일 웨이퍼 변환
                        multi_wm = convert_to_multichannel(wm, n_categories=n_categories)
                        
                        # NaN, Inf 처리
                        if np.any(np.isnan(multi_wm)) or np.any(np.isinf(multi_wm)):
                            multi_wm = np.nan_to_num(multi_wm, nan=0.0, posinf=1.0, neginf=0.0)
                        
                        # 필터링 적용 (channel 0에만)
                        info = None
                        if use_filter and multi_wm[0].sum() > 0:                            
                            if use_density_aware:
                                multi_wm[0], info = filter_obj.filter_single_map(multi_wm[0])
                            else:
                                multi_wm[0] = filter_obj.filter_single_map(multi_wm[0])
                        
                        all_clean_maps.append(multi_wm)
                        all_clean_labels.append(label)
                        all_info.append(info)
                        
                    except Exception as e:
                        continue
            
            print(f"   정리됨: {len(all_clean_maps)}개")

        except Exception as e:
            print(f"❌ {name} 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"✅ 전체 정리 완료: {len(all_clean_maps)}개")
    print(f"   Shape per sample: ({n_channels}, H, W)")
    print(f"{'='*60}")
    
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
                data.shape[2] > 0):       # W > 0

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
        dummy = torch.zeros((1, 13, 128, 128), dtype=torch.float32)  # 11 channels
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