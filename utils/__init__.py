"""
Utilities package for BYOL wafer pattern clustering
"""

from .augmentation import (
    D4Transform,
    WaferAugmentation,
    BYOLAugmentation,
    get_byol_augmentation
)

from .train_byol import (
    train_byol_epoch,
    validate_byol_epoch,
    extract_features,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    detect_collapse,
    log_training_info
)

from .evaluation import (
    compute_pairwise_distances,
    evaluate_retrieval,
    evaluate_clustering,
    evaluate_rotation_invariance,
    evaluate_cluster_consistency_d4,
    evaluate_all,
    print_evaluation_results
)

from .byol_monitor import (
    BYOLMonitor,
    visualize_latent_space
)

__all__ = [
    # Augmentation
    'D4Transform',
    'WaferAugmentation',
    'BYOLAugmentation',
    'get_byol_augmentation',

    # Training
    'train_byol_epoch',
    'validate_byol_epoch',
    'extract_features',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    'detect_collapse',
    'log_training_info',

    # Evaluation
    'compute_pairwise_distances',
    'evaluate_retrieval',
    'evaluate_clustering',
    'evaluate_rotation_invariance',
    'evaluate_cluster_consistency_d4',
    'evaluate_all',
    'print_evaluation_results',

    # Monitoring
    'BYOLMonitor',
    'visualize_latent_space'
]
