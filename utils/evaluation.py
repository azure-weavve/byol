"""
Evaluation functions for BYOL wafer pattern clustering

Metrics:
1. Retrieval quality (Precision@k, Recall@k, MRR)
2. Clustering quality (Silhouette, Calinski-Harabasz, Davies-Bouldin)
3. Rotation invariance (D4 group consistency)

PyTorch 1.4.0 compatible
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN



def evaluate_knn_consistency(model, dataloader, device, all_features=None,
                              n_samples=500, k=20):
    """
    kNN Consistency 평가: C4 augmented 버전이 원본의 k-NN에 포함되는 비율

    원리:
        - N개 샘플에 대해 C4 회전(90°, 180°, 270°) 적용 → 각 embedding 추출
        - 전체 validation set feature pool에서 원본의 k-NN 검색
        - k-NN 안에 C4 변환 버전이 포함된 비율 = consistency

    Args:
        model: BYOL model
        dataloader: validation dataloader
        device: torch device
        all_features: 미리 추출한 전체 feature (None이면 내부에서 추출)
        n_samples: 평가에 사용할 샘플 수 (전체에서 샘플링)
        k: k-NN의 k 값

    Returns:
        metrics: dict with knn_consistency and details
    """
    model.eval()

    # 1. 전체 feature pool 준비 (이미 추출된 것 재활용)
    if all_features is None:
        from utils.train_byol import extract_features
        all_features, _ = extract_features(model, dataloader, device,
                                            use_target=True, verbose=False)

    # numpy로 변환 (이미 numpy면 그대로)
    if isinstance(all_features, torch.Tensor):
        all_features_np = all_features.cpu().numpy()
    else:
        all_features_np = all_features

    N_total = all_features_np.shape[0]

    # 2. 평가용 샘플 추출 (dataloader에서 원본 이미지 필요)
    sample_images = []
    sample_indices = []

    # 랜덤 인덱스 선택
    selected_indices = np.random.permutation(N_total)[:n_samples]
    selected_set = set(selected_indices.tolist())

    # dataloader에서 해당 인덱스의 이미지 수집
    current_idx = 0
    for data in dataloader:
        if isinstance(data, (list, tuple)):
            if len(data) == 4:
                images = data[0]
            else:
                images = data[0] if len(data) > 0 else data
        else:
            images = data

        batch_size = images.size(0)
        for i in range(batch_size):
            global_idx = current_idx + i
            if global_idx in selected_set:
                sample_images.append(images[i])
                sample_indices.append(global_idx)

        current_idx += batch_size

        if len(sample_images) >= n_samples:
            break

    actual_n_samples = len(sample_images)
    if actual_n_samples == 0:
        print("⚠️  No samples collected for kNN consistency evaluation")
        return {'knn_consistency': 0.0, 'n_samples': 0}

    # 3. 전체 feature pool의 L2 norm 미리 계산 (cosine distance용)
    #    euclidean distance 사용
    #    sklearn 없이 직접 계산 (PyTorch 1.4.0 호환)

    # 4. 각 샘플에 대해 C4 변환 후 kNN consistency 계산
    consistencies = []

    with torch.no_grad():
        for idx, (img, global_idx) in enumerate(zip(sample_images, sample_indices)):
            # C4 회전 적용 (90°, 180°, 270° - 0°는 원본이므로 제외)
            rotated_images = []
            for rot_k in [1, 2, 3]:  # 90°, 180°, 270°
                rotated = torch.rot90(img, k=rot_k, dims=(-2, -1))
                rotated_images.append(rotated)

            # 회전된 이미지들의 embedding 추출
            rotated_batch = torch.stack(rotated_images).to(device)  # (3, C, H, W)
            rotated_embeddings = model.get_embeddings(rotated_batch,
                                                       use_target=True)  # (3, D)
            rotated_embeddings_np = rotated_embeddings.cpu().numpy()

            # 원본 feature (이미 추출된 all_features에서 가져옴)
            original_feature = all_features_np[global_idx]  # (D,)

            # 원본의 k-NN 검색 (euclidean distance)
            # all_features_np: (N_total, D), original_feature: (D,)
            diffs = all_features_np - original_feature[np.newaxis, :]  # (N_total, D)
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # (N_total,)

            # 자기 자신 제외하고 k개 선택
            distances[global_idx] = float('inf')
            knn_indices = np.argpartition(distances, k)[:k]  # top-k 인덱스
            knn_features = all_features_np[knn_indices]  # (k, D)

            # 각 회전 embedding이 k-NN 안에 있는지 확인
            # "있다" = k-NN 중 가장 가까운 것과의 거리가 threshold 이하
            # 대신 더 직접적인 방법: 회전 embedding과 k-NN feature들 간 최소 거리
            # → 원본과 k-NN의 최대 거리보다 작으면 "포함"으로 판정
            knn_max_dist = np.max(distances[knn_indices])

            n_found = 0
            for rot_emb in rotated_embeddings_np:
                # 회전 embedding과 전체 feature pool 간 거리
                rot_diffs = all_features_np - rot_emb[np.newaxis, :]
                rot_distances = np.sqrt(np.sum(rot_diffs ** 2, axis=1))

                # 이 회전 embedding의 가장 가까운 이웃이 원본의 k-NN 안에 있는가?
                # 또는 더 직접적으로: 이 회전 embedding이 원본의 k-NN 반경 안에 있는가?
                rot_dist_to_original = np.sqrt(np.sum((rot_emb - original_feature) ** 2))
                if rot_dist_to_original <= knn_max_dist:
                    n_found += 1

            consistency = n_found / 3.0  # C4에서 원본 제외 3개
            consistencies.append(consistency)

            if (idx + 1) % 100 == 0:
                print(f"  kNN Consistency: {idx+1}/{actual_n_samples} samples processed, "
                      f"running avg: {np.mean(consistencies):.4f}")

    avg_consistency = np.mean(consistencies)
    std_consistency = np.std(consistencies)

    metrics = {
        'knn_consistency': avg_consistency,
        'knn_consistency_std': std_consistency,
        'n_samples': actual_n_samples,
        'k': k,
        'perfect_consistency_ratio': np.mean([c == 1.0 for c in consistencies]),
        'zero_consistency_ratio': np.mean([c == 0.0 for c in consistencies]),
    }

    print(f"\n  kNN Consistency: {avg_consistency:.4f} (±{std_consistency:.4f})")
    print(f"  Perfect (3/3): {metrics['perfect_consistency_ratio']:.1%}")
    print(f"  Zero (0/3): {metrics['zero_consistency_ratio']:.1%}")

    return metrics


def compute_pairwise_distances(features, metric='euclidean'):
    """
    Compute pairwise distances

    Args:
        features: (N, D) feature tensor
        metric: 'euclidean' or 'cosine'

    Returns:
        distances: (N, N) distance matrix
    """
    if metric == 'euclidean':
        # Euclidean distance
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        features_squared = (features ** 2).sum(dim=1, keepdim=True)
        distances = features_squared + features_squared.t() - 2 * torch.mm(features, features.t())
        distances = torch.sqrt(torch.clamp(distances, min=0.0))

    elif metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = torch.mm(features_norm, features_norm.t())
        distances = 1 - cosine_sim

    else:
        raise ValueError(f"Invalid metric: {metric}")

    return distances


def evaluate_retrieval(features, k=5, metric='euclidean', ground_truth_func=None):
    """
    Evaluate retrieval quality

    Args:
        features: (N, D) feature tensor
        k: number of nearest neighbors
        metric: distance metric
        ground_truth_func: function to determine if two samples are similar
                          (optional, for synthetic evaluation)

    Returns:
        metrics: dict with Precision@k, Recall@k, MRR
    """
    N = features.size(0)

    # Compute distances
    distances = compute_pairwise_distances(features, metric=metric)

    # Get k nearest neighbors for each sample (excluding self)
    # Sort by distance
    sorted_indices = torch.argsort(distances, dim=1)

    # Exclude self (first element is always self with distance 0)
    nearest_neighbors = sorted_indices[:, 1:k+1]

    # If ground truth function is provided, compute precision/recall
    if ground_truth_func is not None:
        precisions = []
        recalls = []
        reciprocal_ranks = []

        for i in range(N):
            # Get ground truth similar samples
            gt_similar = ground_truth_func(i)

            if len(gt_similar) == 0:
                continue

            # Top-k predictions
            top_k = nearest_neighbors[i].cpu().numpy()

            # True positives
            tp = len(set(top_k) & set(gt_similar))

            # Precision@k
            precision = tp / k
            precisions.append(precision)

            # Recall@k
            recall = tp / len(gt_similar)
            recalls.append(recall)

            # MRR (Mean Reciprocal Rank)
            rank = None
            for r, idx in enumerate(top_k):
                if idx in gt_similar:
                    rank = r + 1
                    break
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        metrics = {
            'precision_at_k': np.mean(precisions),
            'recall_at_k': np.mean(recalls),
            'mrr': np.mean(reciprocal_ranks)
        }

    else:
        # Without ground truth, just return average distances
        avg_distances = []
        for i in range(N):
            top_k = nearest_neighbors[i]
            avg_dist = distances[i, top_k].mean().item()
            avg_distances.append(avg_dist)

        metrics = {
            'avg_distance_top_k': np.mean(avg_distances),
            'std_distance_top_k': np.std(avg_distances)
        }

    return metrics


def estimate_optimal_eps(features, min_samples=10, method='elbow'):
    """
    K-distance graph 기반 최적 eps 추정
    당신의 latent space에 맞춤형
    
    Args:
        features: (N, D) embeddings
        min_samples: DBSCAN min_samples
        method: 'elbow' or 'percentile'
    
    Returns:
        eps: 추정된 eps 값
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # K-distance graph 생성
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    
    # min_samples번째 거리 추출 및 정렬
    k_distances = np.sort(distances[:, min_samples-1], axis=0)
    
    if method == 'elbow':
        # Elbow point 찾기 (2차 미분)
        second_derivative = np.diff(k_distances, n=2)
        elbow_idx = np.argmax(second_derivative)
        eps = k_distances[elbow_idx]
        
    elif method == 'percentile':
        # Percentile 기반 (더 보수적, 권장)
        # 90 percentile = 대부분의 점을 포함하되 sparse 제거
        eps = np.percentile(k_distances, 70)
    
    return eps, k_distances


def visualize_k_distance_graph(features, min_samples=10, save_path=None):
    """
    K-distance graph 시각화 (eps 선택 도움)
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    eps, k_distances = estimate_optimal_eps(features, min_samples, method='elbow')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(k_distances, linewidth=1)
    ax.axhline(y=eps, color='r', linestyle='--', label=f'Estimated eps={eps:.4f}')
    ax.set_xlabel('Data Points (sorted)')
    ax.set_ylabel('K-distance')
    ax.set_title('K-distance Graph for DBSCAN eps Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"K-distance graph saved to {save_path}")
    
    plt.close()
    
    return eps


def evaluate_clustering(features, min_cluster_size=50, min_samples=10):
    """
    DBSCAN with optimal eps estimation
    
    Args:
        features: (N, D) embeddings
        min_cluster_size: minimum cluster size (HDBSCAN 호환)
        min_samples: DBSCAN min_samples
    
    Returns:
        metrics: dict (항상 유효한 값 반환 보장)
        labels: cluster labels
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import torch
    
    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    # ✅ k-distance 계산
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    k_distances = np.sort(distances[:, min_samples-1])
    
    # ✅ eps 후보 생성 (더 넓은 범위)
    percentile_candidates = [30, 40, 50, 60, 70, 80, 90]
    eps_candidates = [np.percentile(k_distances, p) for p in percentile_candidates]
    
    print("\n🔍 Testing multiple eps values:")
    print("─" * 80)
    
    all_results = []  # 모든 결과 저장
    
    for percentile, eps in zip(percentile_candidates, eps_candidates):
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
        
        n_clusters = len(np.unique(labels[labels != -1]))
        noise_ratio = (labels == -1).sum() / len(labels)
        non_noise_count = (labels != -1).sum()
        
        # Silhouette 계산 (가능한 경우)
        sil = None
        if n_clusters >= 2 and non_noise_count >= 10:
            try:
                non_noise = labels[labels != -1]
                clustered_features = features[labels != -1]
                sil = silhouette_score(clustered_features, non_noise)
            except:
                sil = None
        
        # Score 계산
        if sil is not None:
            # 목표: n_clusters 15-40, noise < 15%, silhouette > 0.5
            score = sil - 0.1 * abs(n_clusters - 25) / 25 - 0.3 * noise_ratio
        elif n_clusters >= 2:
            # silhouette 없으면 클러스터 수와 noise만 고려
            score = -abs(n_clusters - 25) / 25 - noise_ratio
        else:
            score = -999  # 클러스터가 1개 이하면 최악
        
        all_results.append({
            'percentile': percentile,
            'eps': eps,
            'labels': labels,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': sil,
            'score': score
        })
        
        # 출력
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        print(f"percentile={percentile:2d}: eps={eps:.4f}, "
              f"clusters={n_clusters:2d}, noise={noise_ratio:5.1%}, "
              f"silhouette={sil_str}, score={score:7.4f}")
    
    print("─" * 80)
    
    # ✅ 최선의 결과 선택 (score 기준)
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best = all_results[0]
    
    # ✅ 결과가 너무 안 좋으면 경고
    if best['n_clusters'] < 2:
        print("⚠️  WARNING: Could not find 2+ clusters!")
        print("    This might indicate:")
        print("    1. Model not trained enough (features too similar)")
        print("    2. Need to adjust min_samples or eps range")
        print("    3. Data might have very few distinct patterns")
    
    print(f"\n✅ Selected: eps={best['eps']:.4f}, clusters={best['n_clusters']}, "
          f"noise={best['noise_ratio']:.1%}, silhouette={best['silhouette']}")
    
    labels = best['labels']
    
    # ✅ Metrics 계산 (항상 반환 보장)
    non_noise_mask = labels != -1
    n_clusters = len(np.unique(labels[non_noise_mask]))
    noise_ratio = (labels == -1).sum() / len(labels)
    
    # Clustering quality metrics
    if n_clusters >= 2 and non_noise_mask.sum() >= 10:
        clustered_features = features[non_noise_mask]
        clustered_labels = labels[non_noise_mask]
        
        try:
            silhouette = silhouette_score(clustered_features, clustered_labels)
            calinski = calinski_harabasz_score(clustered_features, clustered_labels)
            davies = davies_bouldin_score(clustered_features, clustered_labels)
        except:
            silhouette = None
            calinski = None
            davies = None
    else:
        silhouette = None
        calinski = None
        davies = None
    
    metrics = {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies
    }

    return metrics, labels


def evaluate_rotation_invariance(model, test_samples, device, metric='cosine'):
    """
    Evaluate D4 rotation invariance

    Args:
        model: BYOL model
        test_samples: (N, 1, H, W) test samples
        device: torch device
        metric: distance metric

    Returns:
        metrics: dict with invariance metrics
    """
    from utils.augmentation import D4Transform

    model.eval()

    all_cos_sims = []
    all_variances = []

    with torch.no_grad():
        for i in range(test_samples.size(0)):
            sample = test_samples[i]  # (1, H, W)

            # Get all 8 D4 transformations -> C4
            # d4_transforms = D4Transform.get_all_transforms(sample)
            # d4_batch = torch.stack(d4_transforms).to(device)  # (8, 1, H, W)
            c4_transforms = D4Transform.get_c4_transforms(sample)
            c4_batch = torch.stack(c4_transforms).to(device)  # (4, C, H, W)

            # Extract embeddings
            # embeddings = model.get_embeddings(d4_batch, use_target=True)  # (8, D)
            embeddings = model.get_embeddings(c4_batch, use_target=True)  # (4, D)

            # Compute pairwise cosine similarities within D4 group
            embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

            # Average cosine similarity (excluding diagonal)
            # mask = ~torch.eye(8, dtype=torch.bool, device=device)
            mask = ~torch.eye(4, dtype=torch.bool, device=device)
            avg_cos_sim = cos_sim_matrix[mask].mean().item()
            all_cos_sims.append(avg_cos_sim)

            # Variance of embeddings (lower is better for invariance)
            variance = embeddings.var(dim=0).mean().item()
            all_variances.append(variance)

    metrics = {
        'avg_cosine_similarity': np.mean(all_cos_sims),
        'std_cosine_similarity': np.std(all_cos_sims),
        'avg_variance': np.mean(all_variances),
        'std_variance': np.std(all_variances),
        'min_cosine_similarity': np.min(all_cos_sims),
        'max_cosine_similarity': np.max(all_cos_sims)
    }

    return metrics


def evaluate_cluster_consistency_d4(model, test_samples, device, min_cluster_size=5):
    """
    Evaluate if D4 transformations are assigned to the same cluster

    Args:
        model: BYOL model
        test_samples: (N, 1, H, W) test samples
        device: torch device
        min_cluster_size: minimum cluster size for HDBSCAN

    Returns:
        consistency_ratio: ratio of D4 groups in same cluster
    """
    from utils.augmentation import D4Transform

    model.eval()

    all_embeddings = []
    group_ids = []

    with torch.no_grad():
        for i in range(test_samples.size(0)):
            sample = test_samples[i]

            # Get all 8 D4 transformations
            d4_transforms = D4Transform.get_all_transforms(sample)
            d4_batch = torch.stack(d4_transforms).to(device)

            # Extract embeddings
            embeddings = model.get_embeddings(d4_batch, use_target=True)

            all_embeddings.append(embeddings.cpu())
            # group_ids.extend([i] * 8)  # Group ID for each transformation (D4)
            group_ids.extend([i] * 4)  # Group ID for each transformation (C4)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N*8, D)

    # Cluster
    _, cluster_labels = evaluate_clustering(
        all_embeddings,
        min_cluster_size=min_cluster_size
    )

    # Check consistency: how many D4 groups are in the same cluster?
    n_samples = test_samples.size(0)
    consistent_groups = 0

    for i in range(n_samples):
        # Get cluster labels for this D4 group
        # group_labels = cluster_labels[i*8:(i+1)*8] # D4
        group_labels = cluster_labels[i*4:(i+1)*4] # C4

        # Check if all in same cluster (and not noise)
        unique_labels = np.unique(group_labels)
        if len(unique_labels) == 1 and unique_labels[0] != -1:
            consistent_groups += 1

    consistency_ratio = consistent_groups / n_samples

    return consistency_ratio


def evaluate_all(model, dataloader, device, n_samples_invariance=100, n_samples_knn=500, k_knn=20, log_dir='logs'):
    """
    Comprehensive evaluation with DBSCAN optimization
    변경사항:
    - knn_consistency 평가 추가
    - rotation_invariance는 유지 (모니터링용, composite score에서만 제외)
    """
    from utils.train_byol import extract_features
    import os

    print("Extracting features...")
    features, _ = extract_features(model, dataloader, device, use_target=True, verbose=False)

    print("\nEvaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(features, k=5, metric='euclidean')

    # K-distance graph 시각화 (eps 선택 도움)
    print("\nVisualizing k-distance graph...")
    visualize_k_distance_graph(
        features, 
        min_samples=10,
        save_path=os.path.join(log_dir, 'k_distance_graph.png')
    )

    print("\nEvaluating clustering...")
    clustering_metrics, labels = evaluate_clustering(features, min_cluster_size=50)

    # Get test samples for rotation invariance
    # ===== 기존 rotation invariance (모니터링용 유지) =====
    print("\nEvaluating rotation invariance...")
    test_samples = []
    for data in dataloader:
        if isinstance(data, (list, tuple)):
            images = data[0]
        else:
            images = data

        test_samples.append(images)

        if len(test_samples) * images.size(0) >= n_samples_invariance:
            break

    test_samples = torch.cat(test_samples, dim=0)[:n_samples_invariance]

    invariance_metrics = evaluate_rotation_invariance(model, test_samples, device)
    consistency_ratio = evaluate_cluster_consistency_d4(model, test_samples, device)

    # ===== 🆕 kNN Consistency 추가 =====
    print("\nEvaluating kNN consistency...")
    knn_metrics = evaluate_knn_consistency(
        model, dataloader, device,
        all_features=features,
        n_samples=n_samples_knn,
        k=k_knn
    )

    # Combine all metrics
    all_metrics = {
        'retrieval': retrieval_metrics,
        'clustering': clustering_metrics,
        'rotation_invariance': invariance_metrics,
        'cluster_consistency_d4': consistency_ratio,
        'knn_consistency': knn_metrics,  # 🆕
    }

    return all_metrics, labels

# evaluation.py에 추가
def find_optimal_dbscan_params(features, min_samples_list=[5, 10, 15], 
                                percentiles=[75, 80, 85, 90, 95]):
    """
    여러 파라미터 조합 테스트
    """
    results = []
    
    for min_samples in min_samples_list:
        for percentile in percentiles:
            eps, _ = estimate_optimal_eps(features, min_samples, method='percentile')
            # 수동으로 percentile 적용
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(features)
            distances, _ = neighbors_fit.kneighbors(features)
            k_distances = np.sort(distances[:, min_samples-1], axis=0)
            eps = np.percentile(k_distances, percentile)
            
            # 평가
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
            
            n_clusters = len(np.unique(labels[labels != -1]))
            noise_ratio = (labels == -1).sum() / len(labels)
            
            if n_clusters > 1 and (labels != -1).sum() > 10:
                non_noise = labels[labels != -1]
                clustered_features = features[labels != -1]
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(clustered_features, non_noise)
            else:
                silhouette = None
            
            results.append({
                'min_samples': min_samples,
                'percentile': percentile,
                'eps': eps,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'silhouette': silhouette
            })
    
    return pd.DataFrame(results)

# 사용 (epoch 0에 한 번만)
# tuning_results = find_optimal_dbscan_params(features)
# print(tuning_results.to_string())
# 가장 좋은 조합 선택해서 파라미터 고정


def print_evaluation_results(metrics):
    """
    Print evaluation results in a nice format

    Args:
        metrics: dict from evaluate_all
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Retrieval
    print("\nRetrieval Metrics:")
    print("-" * 60)
    for key, value in metrics['retrieval'].items():
        if value is not None:
            print(f"  {key:30s}: {value:.4f}")

    # Clustering
    print("\nClustering Metrics:")
    print("-" * 60)
    for key, value in metrics['clustering'].items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")

    # Rotation Invariance
    print("\nRotation Invariance Metrics:")
    print("-" * 60)
    for key, value in metrics['rotation_invariance'].items():
        print(f"  {key:30s}: {value:.4f}")

    print(f"\n  {'Cluster Consistency (D4)':30s}: {metrics['cluster_consistency_d4']:.4f}")

    print("\n" + "="*60)


def test_evaluation():
    """Test evaluation functions"""
    print("Testing evaluation functions...")

    # Create dummy features
    N = 100
    D = 512
    features = torch.randn(N, D)

    # Test retrieval
    print("\nTesting retrieval evaluation...")
    retrieval_metrics = evaluate_retrieval(features, k=5)
    print(f"Retrieval metrics: {retrieval_metrics}")

    # Test clustering
    print("\nTesting clustering evaluation...")
    clustering_metrics, labels = evaluate_clustering(features, min_cluster_size=10)
    print(f"Clustering metrics: {clustering_metrics}")
    print(f"Unique labels: {np.unique(labels)}")

    print("\nEvaluation test passed!")


if __name__ == "__main__":
    test_evaluation()