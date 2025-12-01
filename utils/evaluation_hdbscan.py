"""
Evaluation functions for BYOL wafer pattern clustering

Metrics:
1. Retrieval quality (Precision@k, Recall@k, MRR)
2. Clustering quality (Silhouette, Calinski-Harabasz, Davies-Bouldin)
3. Rotation invariance (D4 group consistency)

PyTorch 1.4.0 compatible
"""

import torch
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Using DBSCAN instead.")


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


def evaluate_clustering(features, min_cluster_size=50, min_samples=10, use_hdbscan=True):
    """
    Evaluate clustering quality

    Args:
        features: (N, D) feature tensor or numpy array
        min_cluster_size: minimum cluster size for HDBSCAN
        min_samples: minimum samples for HDBSCAN
        use_hdbscan: use HDBSCAN if available, else DBSCAN

    Returns:
        metrics: dict with clustering metrics
        labels: cluster labels
    """
    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    # Perform clustering
    if use_hdbscan and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(features)
    else:
        # Fallback to DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_samples)
        labels = clusterer.fit_predict(features)

    # Filter out noise points (label -1)
    non_noise_mask = labels != -1
    n_noise = (labels == -1).sum()
    n_samples = len(labels)
    noise_ratio = n_noise / n_samples

    if non_noise_mask.sum() < 2:
        # Not enough samples for clustering metrics
        return {
            'n_clusters': 0,
            'noise_ratio': noise_ratio,
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None
        }, labels

    # Get clustered features and labels
    clustered_features = features[non_noise_mask]
    clustered_labels = labels[non_noise_mask]

    # Number of clusters
    n_clusters = len(np.unique(clustered_labels))

    # Compute metrics (only for non-noise points)
    if n_clusters > 1:
        silhouette = silhouette_score(clustered_features, clustered_labels)
        calinski = calinski_harabasz_score(clustered_features, clustered_labels)
        davies = davies_bouldin_score(clustered_features, clustered_labels)
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

            # Get all 8 D4 transformations
            d4_transforms = D4Transform.get_all_transforms(sample)
            d4_batch = torch.stack(d4_transforms).to(device)  # (8, 1, H, W)

            # Extract embeddings
            embeddings = model.get_embeddings(d4_batch, use_target=True)  # (8, D)

            # Compute pairwise cosine similarities within D4 group
            embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

            # Average cosine similarity (excluding diagonal)
            mask = ~torch.eye(8, dtype=torch.bool, device=device)
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
            group_ids.extend([i] * 8)  # Group ID for each transformation

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N*8, D)

    # Cluster
    _, cluster_labels = evaluate_clustering(
        all_embeddings,
        min_cluster_size=min_cluster_size,
        use_hdbscan=True
    )

    # Check consistency: how many D4 groups are in the same cluster?
    n_samples = test_samples.size(0)
    consistent_groups = 0

    for i in range(n_samples):
        # Get cluster labels for this D4 group
        group_labels = cluster_labels[i*8:(i+1)*8]

        # Check if all in same cluster (and not noise)
        unique_labels = np.unique(group_labels)
        if len(unique_labels) == 1 and unique_labels[0] != -1:
            consistent_groups += 1

    consistency_ratio = consistent_groups / n_samples

    return consistency_ratio


def evaluate_all(model, dataloader, device, n_samples_invariance=100):
    """
    Comprehensive evaluation

    Args:
        model: BYOL model
        dataloader: data loader
        device: torch device
        n_samples_invariance: number of samples for rotation invariance test

    Returns:
        all_metrics: dict with all metrics
    """
    from utils.train_byol import extract_features

    print("Extracting features...")
    features, _ = extract_features(model, dataloader, device, use_target=True)

    print("\nEvaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(features, k=5, metric='euclidean')

    print("\nEvaluating clustering...")
    clustering_metrics, labels = evaluate_clustering(features, min_cluster_size=50)

    # Get test samples for rotation invariance
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

    # Combine all metrics
    all_metrics = {
        'retrieval': retrieval_metrics,
        'clustering': clustering_metrics,
        'rotation_invariance': invariance_metrics,
        'cluster_consistency_d4': consistency_ratio
    }

    return all_metrics, labels


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
