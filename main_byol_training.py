"""
Main BYOL Training Script

Complete training pipeline for BYOL-based wafer pattern clustering

Usage:
    python main_byol_training.py [--config CONFIG_PATH] [--resume CHECKPOINT_PATH]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import math
from datetime import datetime
import numpy as np
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.byol import BYOL, get_tau_schedule
from utils.batch_augmentation import get_batch_byol_augmentation
from utils.train_byol import (
    train_byol_epoch, validate_byol_epoch, extract_features,
    save_checkpoint, load_checkpoint, EarlyStopping, detect_collapse, log_training_info
)
from utils.evaluation import evaluate_all, print_evaluation_results
from utils.byol_monitor import BYOLMonitor, visualize_latent_space
from utils.dataloader_utils import prepare_clean_data, create_dataloaders
from utils.gpu_monitor import print_gpu_memory, reset_peak_stats, get_peak_memory_mb



class CosineAnnealingWarmUpRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm restarts and warmup
    PyTorch 1.4.0 compatible implementation
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.T_cur = last_epoch
        self.T_i = T_0
        self.cycle = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def load_wafer_data(data_configs, use_filter=True, use_density_aware=False, use_region_aware=False):
    """
    Load and prepare wafer data from multiple sources

    Args:
        data_configs: List of dicts with 'path' and 'name' keys
                     Example: [{"path": "data.npz", "name": "product1"}]
        use_filter: Whether to apply wafer map filtering
        use_density_aware: Use density-aware adaptive filtering (recommended)
        use_region_aware: Use region-aware filtering for very low density maps

    Returns:
        wafer_maps: List of numpy arrays
        labels: List of labels
        info: List of filter info dicts
    """
    if not data_configs or len(data_configs) == 0:
        raise ValueError("data_configs must be provided with at least one dataset")

    # Load and clean data
    wafer_maps, labels, info = prepare_clean_data(
        data_configs,
        use_filter=use_filter,
        filter_params=None,
        use_density_aware=use_density_aware,
        use_region_aware=use_region_aware
    )

    return wafer_maps, labels, info

# 🆕 Composite score 계산 함수
def compute_composite_score(eval_metrics, avg_cos_sim, weights):
    """
    Composite score = silhouette * w1 + rotation_inv * w2 + avg_cos_sim * w3
    (w3는 음수이므로 cos_sim이 높으면 페널티)

    변경된 Composite score:
    knn_consistency * 0.5 + silhouette * 0.3 + avg_cos_sim * (-0.2)

    기존 rotation_invariance 제거, knn_consistency 추가
    """
    clustering = eval_metrics.get('clustering', {})
    knn = eval_metrics.get('knn_consistency', {})

    silhouette = clustering.get('silhouette')
    knn_consistency = knn.get('knn_consistency')

    # 하나라도 없으면 계산 불가
    if silhouette is None or knn_consistency is None:
        return None

    score = (knn_consistency * weights.get('knn_consistency', 0.5)
             + silhouette * weights.get('silhouette', 0.3)
             + avg_cos_sim * weights.get('avg_cos_sim', -0.2))

    return score


def train_byol_wafer(config):
    """
    Main training function

    Args:
        config: dict with training configuration
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Data loaders
    print("\nPreparing data...")

    # Load wafer data
    wafer_maps, labels, _ = load_wafer_data(
        data_configs=config['data_configs'],
        use_filter=config.get('use_filter', True),
        use_density_aware=config.get('use_density_aware', False),
        use_region_aware=config.get('use_region_aware', False)
    )

    if wafer_maps is None or len(wafer_maps) == 0:
        raise ValueError("Failed to load data. Please check your data_configs paths.")
    
    # ✅ Auto-detect input channels
    n_channels = wafer_maps[0].shape[0]  # (C, H, W)
    # Safety check
    for i in range(min(10, len(wafer_maps))):
        assert wafer_maps[i].shape[0] == n_channels, \
            f"Channel mismatch at index {i}: {wafer_maps[i].shape[0]} vs {n_channels}"
    print(f"Detected input channels: {n_channels}")

    # Create dataloaders from real data
    # IMPORTANT: use_augmentation=False because BYOL applies augmentation in training loop
    train_loader, val_loader = create_dataloaders(
        wafer_maps=wafer_maps,
        labels=labels,
        batch_size=config['batch_size'],
        target_size=(config['wafer_size'], config['wafer_size']),
        test_size=config.get('test_size', 0.2),
        use_filter=False,  # Already filtered in prepare_clean_data
        filter_on_the_fly=False,
        filter_params=None,
        use_density_aware=False,
        use_augmentation=False  # BYOL applies augmentation in train_byol_epoch
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating BYOL model...")
    model = BYOL(
        input_channels=n_channels,  # ✅ 추가
        encoder_dim=config['encoder_dim'],
        projector_hidden=config['projector_hidden'],
        projector_out=config['projector_out'],
        predictor_hidden=config['predictor_hidden'],
        use_radial_encoding=config['use_radial_encoding'],
        use_attention=config['use_attention'],
        wafer_size=(config['wafer_size'], config['wafer_size']),
        tau=config['tau_base']
    ).to(device)
    print_gpu_memory("After Model Creation")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    print("\nSetting up optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['base_lr'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_max=config['eta_max'],
        T_up=config['T_up'],
        gamma=config['gamma']
    )

    # Augmentation
    print("Setting up augmentation...")
    aug_weak = get_batch_byol_augmentation('weak', n_spatial_channels=config.get('n_spatial_channels', 13))
    aug_strong = get_batch_byol_augmentation('strong', n_spatial_channels=config.get('n_spatial_channels', 13))

    # ✅ 수정: resume 파라미터 전달
    resume_training = config.get('resume_path') is not None and os.path.exists(config.get('resume_path', ''))

    # Monitor
    monitor = BYOLMonitor(
        log_dir=config['log_dir'],
        eval_frequency=config['eval_frequency'],
        save_plots=True,
        resume=resume_training  # ✅ 자동으로 이전 history 로드
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_delta'],
        mode=config['early_stopping_mode']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_composite = -float('inf')  # 🆕 composite score 기준 best

    if config.get('resume_path') is not None and os.path.exists(config['resume_path']):
        print(f"\nResuming from checkpoint: {config['resume_path']}")
        start_epoch, resumed_loss, best_val_loss, best_composite = load_checkpoint(
            model, optimizer, scheduler, config['resume_path'], device
        )
        start_epoch += 1
        print(f"Resumed: epoch {start_epoch}, last_loss: {resumed_loss:.4f}, best: {best_val_loss:.4f}")

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("="*60)

    # Peak memory 측정 시작
    reset_peak_stats()

    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        if epoch == start_epoch:
            print_gpu_memory("Before Training")

        # 🆕 Variance config 준비
        variance_config = {
            'type': config.get('variance_type', 'target_std_robust'),
            'target_std': config.get('variance_target_std', 1.0),
            'margin': config.get('variance_margin', 0.1),
            'weight': config.get('variance_weight', 0.0),
            'covariance_weight': config.get('covariance_weight', 0.0),  # 🆕
        }

        # Get current tau for EMA update
        tau = get_tau_schedule(epoch, config['epochs'], config['tau_base'], config['tau_max'])

        # Train
        train_loss, byol_loss, var_loss, cov_loss, feat_std, avg_cos_sim = train_byol_epoch(
            model, train_loader, optimizer, device,
            tau=tau, augmentation=aug_weak, augmentation_strong=aug_strong, epoch=epoch, variance_config=variance_config, verbose=False
        )

        # Validate
        val_loss = validate_byol_epoch(
            model, val_loader, device, augmentation=aug_weak, augmentation_strong=aug_strong, verbose=False
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Collapse detection
        with torch.no_grad():
            # Get some features for collapse detection
            sample_batch = next(iter(val_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_batch = sample_batch[0]
            sample_batch = sample_batch[:min(32, len(sample_batch))].to(device)
            sample_features = model.get_embeddings(sample_batch, use_target=True)

            is_collapsed, collapse_info = detect_collapse(sample_features)
        
        if epoch == start_epoch:
            print_gpu_memory("After First Epoch (Peak = Max Training Memory)")

        # Log to monitor
        monitor.log_epoch(epoch, train_loss, val_loss, current_lr, tau)
        monitor.log_variance_metrics(
            epoch=epoch,
            byol_loss=byol_loss,
            variance_loss=var_loss,
            covariance_loss=cov_loss,                                       # 🆕
            variance_weight=variance_config['weight'],
            covariance_weight=variance_config.get('covariance_weight', 0.0),  # 🆕
            feature_std=feat_std,
            avg_cos_sim=avg_cos_sim,
            target_std=variance_config.get('target_std', 1.0)
        )
        monitor.log_collapse_detection(
            epoch, collapse_info['feat_std'],
            collapse_info['avg_cos_sim'], is_collapsed
        )

        # Elapsed time
        elapsed_time = time.time() - epoch_start_time

        # Print info
        log_training_info(
            epoch, train_loss, val_loss, current_lr, tau,
            elapsed_time, collapse_info
        )

        # Evaluate periodically
        if monitor.should_evaluate(epoch):
            print(f"\nPerforming evaluation at epoch {epoch+1}...")
            eval_metrics, cluster_labels = evaluate_all(
                model, val_loader, device, n_samples_invariance=100, k_knn=config['k_knn'], log_dir=config['log_dir']
            )
            print_evaluation_results(eval_metrics)

            monitor.log_evaluation(epoch, eval_metrics)

            # 첫 evaluation 후 peak memory 출력 (evaluation이 train보다 메모리를 더 쓸 수 있음)
            if epoch == start_epoch + config['eval_frequency'] - 1:
                print_gpu_memory("After First Evaluation (Overall Peak)")

            # Visualize latent space
            features, _ = extract_features(model, val_loader, device, use_target=True, verbose=False)
            visualize_latent_space(
                features[:1000],
                labels=cluster_labels[:1000] if cluster_labels is not None else None,
                method='tsne',
                save_path=os.path.join(config['log_dir'], f'latent_space_epoch_{epoch+1}.png'),
                title=f'Latent Space (Epoch {epoch+1})'
            )

            # 🆕 Composite score 계산
            composite_weights = config.get('composite_weights', {
                'knn_consistency': 0.5, 'silhouette': 0.3, 'avg_cos_sim': -0.2
            })
            current_composite = compute_composite_score(
                eval_metrics, avg_cos_sim, composite_weights
            )

            if current_composite is not None:
                print(f"  📊 Composite Score: {current_composite:.4f} (best: {best_composite:.4f})")

                # Best model 저장 (composite 기준)
                if current_composite > best_composite:
                    best_composite = current_composite
                    save_path = os.path.join(config['save_dir'], 'best_model.pth')
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, val_loss, save_path,
                        best_val_loss=best_val_loss,
                        best_composite=best_composite,
                        config=config
                    )
                    # 점수 구성요소도 출력
                    knn = eval_metrics.get('knn_consistency', {}).get('knn_consistency')
                    sil = eval_metrics.get('clustering', {}).get('silhouette', 0)
                    rot = eval_metrics.get('rotation_invariance', {}).get('avg_cosine_similarity', 0)
                    print(f"  🏆 Best model saved! composite={current_composite:.4f} "
                        f"(knn={knn:.3f}, sil={sil:.3f}, rot={rot:.3f}, cos={avg_cos_sim:.3f})")

                # Early stopping 체크
                if early_stopping(current_composite):
                    print(f"\n⏹ Early stopping at epoch {epoch+1} "
                        f"(composite score not improving for {config['early_stopping_patience']} evaluations)")
                    break

        # Val loss best 별도 추적 (참고용, 저장은 안 함)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Regular checkpoint (기존과 동일)
        if epoch == 0 or (epoch + 1) % config['save_frequency'] == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, save_path,
                best_val_loss=best_val_loss,
                best_composite=best_composite,
                config=config
            )

        # # ❌ 기존 val_loss 기반 early stopping 제거
        # if early_stopping(val_loss):
        #     print(f"\nEarly stopping triggered at epoch {epoch+1}")
        #     break

        # Save plots periodically
        if epoch == 0 or (epoch + 1) % config['save_frequency'] == 0:
            monitor.plot_training_curves()
            monitor.plot_evaluation_metrics()
            monitor.save_history()

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    eval_metrics, cluster_labels = evaluate_all(
        model, val_loader, device, n_samples_invariance=200, k_knn=config['k_knn'], log_dir=config['log_dir']
    )
    print_evaluation_results(eval_metrics)

    # ✅ 추가: Final Evaluation 결과를 history에 저장
    monitor.log_evaluation(epoch, eval_metrics)

    # Save final model
    save_path = os.path.join(config['save_dir'], 'final_model.pth')
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_loss, save_path,
        best_val_loss=best_val_loss,  # 🔴 추가
        config=config, eval_metrics=eval_metrics
    )

    # Final plots
    monitor.plot_training_curves()
    monitor.plot_evaluation_metrics()
    monitor.save_history()
    monitor.print_summary()

    # Final visualization
    features, _ = extract_features(model, val_loader, device, use_target=True)
    visualize_latent_space(
        features,
        labels=cluster_labels,
        method='tsne',
        save_path=os.path.join(config['log_dir'], 'final_latent_space.png'),
        title='Final Latent Space'
    )

    # ✅ 추가: 학습 요약 생성
    print("\n" + "="*60)
    print("Generating Training Summary...")
    print("="*60)
    
    from utils.training_summary import generate_training_summary
    generate_training_summary(log_dir=config['log_dir'], interval=config['save_frequency'])
    
    print("\nTraining completed!")


def get_default_config(path):
    """Get default configuration"""
    config = {
        # Data - Multiple wafer data sources
        'data_configs': [
            {"path": f"{path}/dataset/extract_multi_channel/dataset/root/root_map_data_bin_cat_map.npz", "name": "Root"},
            {"path": f"{path}/dataset/extract_multi_channel/dataset/rose/rose_map_data_bin_cat_map.npz", "name": "Rose"},
            {"path": f"{path}/dataset/extract_multi_channel/dataset/santa/santa_map_data_bin_cat_map.npz", "name": "Santa"},
            {"path": f"{path}/dataset/extract_multi_channel/dataset/zuma_pro/zuma_pro_map_data_bin_cat_map.npz", "name": "Zuma_pro"},
            {"path": f"{path}/dataset/extract_multi_channel/dataset/thetis/thetis_map_data_bin_cat_map.npz", "name": "Thetis"}
        ],
        'use_filter': True,
        'use_density_aware': True,
        'use_region_aware': False,
        'test_size': 0.2,

        # Data parameters
        'wafer_size': 128,
        'batch_size': 256,

        # Model
        'encoder_dim': 512,
        'projector_hidden': 1024,
        'projector_out': 256,
        'predictor_hidden': 1024,
        'use_radial_encoding': True,
        'use_attention': True,

        # Training
        'epochs': 100,
        # 'base_lr': 0.0001,
        'base_lr': 0.00005,
        'weight_decay': 0.01,

        # BYOL
        'tau_base': 0.996,
        'tau_max': 0.999,

        # Augmentation
        'augmentation_type': 'strong',

        # Scheduler
        'T_0': 30,
        'T_mult': 1,
        # 'eta_max': 0.001,
        'eta_max': 0.0005,
        'T_up': 5,
        'gamma': 0.9,
        
        # Variance Regularization (Target Std)
        'variance_type': 'target_std_robust',  # 'target_std' or 'target_std_robust'
        'variance_target_std': 1.0,            # 목표 std
        'variance_margin': 0.1,                # robust margin (0.9~1.1 허용)
        'variance_weight': 0.05,                # loss weight (0.2 -> 0.05)
        'covariance_weight': 0.05,              # 🆕 (0.04 -> 0.1 -> 0.05)

        # Composite score weights (early stopping & best model 기준)
        'composite_weights': {
            'knn_consistency': 0.5,
            'silhouette': 0.3,
            'avg_cos_sim': -0.2,  # 음수 = 낮을수록 좋음
        },

        # K in KNN
        'k_knn': 20,

        # Monitoring
        'eval_frequency': 5,
        'save_frequency': 5,

        # Early stopping
        'early_stopping_patience': 6,          # eval_frequence * early_stopping_patience로 실제 early_stopping 적용
        'early_stopping_delta': 0.01,
        'early_stopping_mode': 'max',          # min, max 중 하나

        # bin category channel
        'n_spatial_channels' : 13,             # BIN Category Channel 수

        # Paths
        'save_dir': 'checkpoints',
        'log_dir': 'logs',
        'resume_path': None
        #'resume_path': f"{path}/clustering/byol_multi_channel/checkpoints/best_model.pth"
        #'resume_path': f"{path}/clustering/byol_multi_channel/checkpoints/checkpoint_epoch_10.pth"
    }

    return config


def main():
    """Main entry point"""
    path = '/mnt/kh0213.jang/Documents/wm811k'
    base_path = path + "/clustering/pth_file"
    today = datetime.today()
    result = today.strftime("%y%m%d")

    # Get default config
    config = get_default_config(path=path)

    # Start training
    train_byol_wafer(config)


if __name__ == "__main__":
    main()