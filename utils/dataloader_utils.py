import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from utils.wafermap_filter import WaferMapFilter
from utils.density_aware_filter import DensityAwareWaferMapFilter
from utils.region_aware_filter import RegionAwareWaferMapFilter


def prepare_clean_data(data_configs, use_filter=True, filter_params=None, use_density_aware=False, use_region_aware=False):
    """
    여러 제품 데이터를 로드하고 완전히 정리 + 필터링

    Args:
        data_configs: [
            {"path": "path1.npz", "name": "product1"},
            {"path": "path2.npz", "name": "mixed_products"},
        ]
        use_filter: 필터링 적용 여부
        filter_params: 필터 파라미터 딕셔너리
        use_density_aware: True면 밀도 기반 적응형 필터 사용 (권장!)

    Returns:
        clean_maps, clean_labels
    """

    print("="*60)
    mode_str = "밀도 기반 적응형" if use_density_aware else "일반"
    print(f"🧹 데이터 완전 정리 시작" + (f" ({mode_str} 필터링 포함)" if use_filter else ""))
    print("="*60)
    
    # 필터 초기화
    if use_filter:
        if use_density_aware:
            # 밀도 기반 적응형 필터
            filter_obj = DensityAwareWaferMapFilter()
        else:
            # 일반 필터
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
                    # 완전한 정리 과정
                    if isinstance(wm, np.ndarray) and wm.dtype == object:
                        # object array → 강제 변환
                        clean_wm = np.array(wm.tolist(), dtype=np.float32)
                    elif isinstance(wm, np.ndarray):
                        clean_wm = wm.astype(np.float32)
                    elif isinstance(wm, (list, tuple)):
                        clean_wm = np.array(wm, dtype=np.float32)
                    else:
                        clean_wm = np.array(wm, dtype=np.float32)

                    # 검증
                    if len(clean_wm.shape) != 2 or clean_wm.shape[0] == 0 or clean_wm.shape[1] == 0:
                        continue

                    # NaN, Inf 처리
                    if np.any(np.isnan(clean_wm)) or np.any(np.isinf(clean_wm)):
                        clean_wm = np.nan_to_num(clean_wm, nan=0.0, posinf=1.0, neginf=0.0)

                    # 정규화
                    if clean_wm.max() > 0:
                        clean_wm = clean_wm / clean_wm.max()

                    # 🔹 필터링 적용
                    if use_filter and clean_wm.sum() > 0:
                        original_defects = clean_wm.sum()
                        
                        if use_density_aware:
                            # 밀도 기반 적응형 필터링
                            clean_wm_org = clean_wm.copy()
                            clean_wm, info = filter_obj.filter_single_map(clean_wm)
                            filtered_defects = clean_wm.sum()
                            if info['strategy'] == "very_low" and use_region_aware:
                                region_obj = RegionAwareWaferMapFilter(n_sectors=12, n_rings=3, sector_density_threshold=0.1, closing_kernel_size=6, min_region_size=50)
                                clean_wm = region_obj.detect_clustering_regions(clean_wm.squeeze())
                                clean_wm = clean_wm.unsqueeze(dim=0)
                        else:
                            # 일반 필터링
                            clean_wm = filter_obj.filter_single_map(clean_wm)
                            filtered_defects = clean_wm.sum()
                            info = None
                        
                        # 너무 많이 제거되면 스킵 (패턴이 거의 사라짐) -> 이게 진짜 필요할까?
                        if filtered_defects < original_defects * 0.2:
                            clean_wm = clean_wm_org.copy()
                            label = label + "_filter"
                            # continue
                        
                        if filtered_defects < original_defects:
                            filtered_count += 1
                    # 최종 검증
                    assert isinstance(clean_wm, np.ndarray)
                    assert clean_wm.dtype == np.float32
                    assert len(clean_wm.shape) == 2
                    assert not np.any(np.isnan(clean_wm))

                    clean_maps.append(clean_wm)
                    clean_labels.append(label)
                    info_list.append(info)

                except Exception as e:
                    # 문제 있는 데이터는 조용히 건너뜀
                    continue

            success_rate = len(clean_maps) / len(maps) * 100
            print(f"   정리됨: {len(clean_maps)}개 ({success_rate:.1f}%)")
            if use_filter and filtered_count > 0:
                print(f"   필터링됨: {filtered_count}개 ({filtered_count/len(clean_maps)*100:.1f}%)")

            all_clean_maps.extend(clean_maps)
            all_clean_labels.extend(clean_labels)
            all_info.extend(info_list)

        except Exception as e:
            print(f"❌ {name} 로딩 실패: {e}")
            continue

    print(f"\n✅ 전체 정리 완료: {len(all_clean_maps)}개")
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
        self.original_indices = []  # 🔴 추가!
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
            if (isinstance(wm, np.ndarray) and
                wm.dtype == np.float32 and
                len(wm.shape) == 2 and
                wm.shape[0] > 0 and wm.shape[1] > 0):

                # 사전 필터링 (filter_on_the_fly=False인 경우)
                if use_filter and not filter_on_the_fly:
                    if wm.sum() > 0:
                        original_defects = wm.sum()
                        
                        if use_density_aware:
                            wm, info = self.filter_obj.filter_single_map(wm)
                        else:
                            wm = self.filter_obj.filter_single_map(wm)
                        
                        # 너무 많이 제거되면 스킵
                        if wm.sum() < original_defects * 0.2:
                            continue
                self.wafer_maps.append(wm)
                self.labels.append(label)
                self.original_indices.append(idx)  # 🔴 원본 인덱스 저장

        print(f"   최종 Dataset: {len(self.wafer_maps)}개")

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx]
        label = self.labels[idx]
        original_idx = self.original_indices[idx]  # 🔴 추가

        # On-the-fly 필터링 (filter_on_the_fly=True인 경우)
        if self.use_filter and self.filter_on_the_fly:
            if wafer_map.sum() > 0:
                if self.use_density_aware:
                    wafer_map, _ = self.filter_obj.filter_single_map(wafer_map)
                else:
                    wafer_map = self.filter_obj.filter_single_map(wafer_map)

        # 안전한 전처리
        tensor = torch.tensor(wafer_map, dtype=torch.float32)
        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor_4d, size=self.target_size, mode='bilinear', align_corners=False)
        resized = resized.squeeze(0)  # (1, H, W)

        # 🔴 Augmentation 적용 (training 시에만!)
        if self.is_training and self.use_augmentation:
            resized_aug = self._apply_augmentation(resized)
        else:
            resized_aug = None

        return resized, resized_aug, label, original_idx  # 🔴 인덱스도 반환
    
    def _apply_augmentation(self, tensor):
        """
        회전 불변성을 위한 Augmentation
        D4 Dihedral group의 8가지 변환 중 하나를 균등하게 선택
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
    + 빈 맵(모두 0)도 제거"""

    safe_data = []
    safe_data_aug = []
    safe_labels = []
    safe_indices = []  # 🔴 추가
    has_aug = True  # 🔹 첫 번째 유효한 샘플로 판단

    for data, data_aug, label, original_idx in batch:  # 🔴 4개 받기
        try:
            if (isinstance(data, torch.Tensor) and
                data.dtype == torch.float32 and
                len(data.shape) == 3 and
                data.sum() > 0):  # 빈 맵 제거:

                safe_data.append(data)
                safe_data_aug.append(data_aug)  # None이거나 tensor
                safe_labels.append(label)
                safe_indices.append(original_idx)  # 🔴 추가

                # 🔹 첫 번째 샘플로 augmentation 여부 판단
                if len(safe_data) == 1:
                    has_aug = (data_aug is not None)

        except:
            continue

    if len(safe_data) == 0:
        # 모든 샘플이 문제인 경우 더미 배치 반환
        dummy = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        return dummy, ["dummy"], [0]  # 🔴 dummy index

    batch_data = torch.stack(safe_data)
    # 🔹 augmentation이 있으면 stack, 없으면 None
    if has_aug:
        batch_data_aug = torch.stack(safe_data_aug)
    else:
        batch_data_aug = None

    return batch_data, batch_data_aug, safe_labels, safe_indices  # 🔴 4개 반환


# def create_dataloaders(wafer_maps, labels, batch_size=64, target_size=(128, 128), test_size=0.2, use_filter=True, filter_on_the_fly=False,
#                         filter_params=None, use_density_aware=False, is_training=False, use_augmentation=False):
#     """
#     필터링 기능이 추가된 안전한 DataLoader들 생성
    
#     Args:
#         wafer_maps: 웨이퍼맵 리스트
#         labels: 라벨 리스트
#         batch_size: 배치 크기
#         target_size: 리사이즈 타겟 크기
#         test_size: validation 비율
#         use_filter: 필터링 사용 여부
#         filter_on_the_fly: True면 런타임 필터링, False면 사전 필터링
#         filter_params: 필터 파라미터
#         use_density_aware: True면 밀도 기반 적응형 필터 사용 (권장!)
    
#     Returns:
#         train_loader, valid_loader
#     """

#     print("\n🔧 안전한 DataLoader 생성")
#     print("="*40)

#     if use_filter:
#         if use_density_aware:
#             mode = "Density-Aware (밀도 기반 적응형)"
#         else:
#             mode = "On-the-fly" if filter_on_the_fly else "Pre-filtering"
#         print(f"   필터링 모드: {mode}")
#     # Dataset 생성
#     dataset = MultiSizeWaferDataset(
#         wafer_maps, labels, 
#         target_size=target_size,
#         use_filter=use_filter,
#         filter_on_the_fly=filter_on_the_fly,
#         filter_params=filter_params,
#         use_density_aware=use_density_aware,
#         is_training=False,  # 기본값
#         use_augmentation=False
#     )

#     # train/valid 분할
#     train_indices, valid_indices = train_test_split(
#         range(len(dataset)), test_size=test_size, random_state=42
#     )

#     # Train dataset with augmentation
#     # 🔴 Train dataset에만 augmentation 활성화
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     train_dataset.dataset.is_training = True
#     train_dataset.dataset.use_augmentation = use_augmentation
    
#     # Valid dataset without augmentation
#     valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
#     valid_dataset.dataset.is_training = False
#     valid_dataset.dataset.use_augmentation = False

#     print(f"   Train: {len(train_dataset)}개 / Valid: {len(valid_dataset)}개")

#     # DataLoader 생성
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=0,
#         pin_memory=False,
#         drop_last=False
#     )

#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=0,
#         pin_memory=False,
#         drop_last=False
#     )

#     print(f"   Train 배치: {len(train_loader)}개 / Valid 배치: {len(valid_loader)}개")

#     return train_loader, valid_loader

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
    print(len(train_indices), len(valid_indices))
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