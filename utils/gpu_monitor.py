import torch


def get_gpu_memory_usage():
    """
    현재 GPU 메모리 사용량을 MB 단위로 반환
    """
    if torch.cuda.is_available():
        # PyTorch가 할당한 메모리
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        # PyTorch가 예약한 메모리
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        # 최대 할당 메모리 (reset 이후 peak)
        peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        # 전체 GPU 메모리
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak,
            'total_mb': total,
            'free_mb': total - allocated
        }
    return None


def print_gpu_memory(stage_name):
    """
    GPU 메모리 사용량을 보기 좋게 출력
    """
    mem = get_gpu_memory_usage()
    if mem:
        print(f"\n{'='*60}")
        print(f"GPU Memory [{stage_name}]")
        print(f"{'='*60}")
        print(f"Allocated: {mem['allocated_mb']:8.1f} MB ({mem['allocated_mb']/1024:.2f} GB)")
        print(f"Peak:      {mem['peak_mb']:8.1f} MB ({mem['peak_mb']/1024:.2f} GB)")
        print(f"Reserved:  {mem['reserved_mb']:8.1f} MB ({mem['reserved_mb']/1024:.2f} GB)")
        print(f"Total:     {mem['total_mb']:8.1f} MB ({mem['total_mb']/1024:.2f} GB)")
        print(f"Usage:     {(mem['allocated_mb']/mem['total_mb']*100):.1f}% (Peak: {(mem['peak_mb']/mem['total_mb']*100):.1f}%)")
        print(f"{'='*60}\n")


def reset_peak_stats():
    """
    Peak memory 카운터 리셋
    측정 구간 시작 전에 호출
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb():
    """
    현재 peak memory를 MB로 반환 (간단 조회용)
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0