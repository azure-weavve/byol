"""
BYOL Training Monitor

Tracks and logs:
- Training/validation loss
- Learning rate and tau schedule
- Feature statistics (for collapse detection)
- Periodic evaluation metrics
- Visualization (t-SNE/UMAP)
- Support for resuming training (previous history + current)

PyTorch 1.4.0 compatible
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os


# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    """numpy, torch 타입을 자동으로 처리하는 JSON encoder"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


class BYOLMonitor:
    """
    Monitor BYOL training progress
    
    Features:
    - 학습 중 metrics 자동 저장
    - 재개 학습 시 이전 history 자동 로드
    - 통합된 history로 plot 생성 (첫 epoch부터 현재까지)
    """
    def __init__(self, log_dir='logs', eval_frequency=10, save_plots=True, resume=False):
        """
        Args:
            log_dir: directory to save logs and plots
            eval_frequency: evaluate every N epochs
            save_plots: save plots to disk
            resume: True면 이전 history 자동 로드
        """
        self.log_dir = log_dir
        self.eval_frequency = eval_frequency
        self.save_plots = save_plots

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # History (이전 데이터 포함)
        self.history = defaultdict(list)
        
        # Collapse detection
        self.collapse_warnings = []
        
        # ✅ 재개 학습 시 이전 history 자동 로드
        if resume:
            self._load_previous_history()

    def _load_previous_history(self):
        """이전 학습의 history 로드"""
        history_path = os.path.join(self.log_dir, 'history.json')
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    previous_history = json.load(f)
                
                # 이전 데이터를 현재 history에 로드
                for key, values in previous_history.items():
                    if isinstance(values, list):
                        self.history[key] = values.copy()
                    else:
                        self.history[key] = values
                
                n_epochs = len(self.history.get('epoch', []))
                print(f"✅ Previous history loaded: {n_epochs} epochs")
                
            except Exception as e:
                print(f"⚠️  Failed to load previous history: {e}")
                self.history = defaultdict(list)
        else:
            print(f"ℹ️  No previous history found (new training)")

    def log_epoch(self, epoch, train_loss, val_loss, learning_rate, tau, **kwargs):
        """
        Log epoch metrics
        
        ✅ 자동으로 기존 데이터와 합쳐짐

        Args:
            epoch: epoch number
            train_loss: training loss
            val_loss: validation loss
            learning_rate: current learning rate
            tau: current tau value
            **kwargs: additional metrics to log
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(learning_rate)
        self.history['tau'].append(tau)

        # Log additional metrics
        for key, value in kwargs.items():
            self.history[key].append(value)

    def log_collapse_detection(self, epoch, feat_std, avg_cos_sim, is_collapsed):
        """
        Log collapse detection info

        Args:
            epoch: epoch number
            feat_std: feature standard deviation
            avg_cos_sim: average cosine similarity
            is_collapsed: whether collapse detected
        """
        self.history['feat_std'].append(feat_std)
        self.history['avg_cos_sim'].append(avg_cos_sim)

        if is_collapsed:
            warning = f"Epoch {epoch}: Collapse detected! feat_std={feat_std:.6f}, cos_sim={avg_cos_sim:.6f}"
            self.collapse_warnings.append(warning)
            print(f"\n{'!'*60}")
            print(f"WARNING: {warning}")
            print(f"{'!'*60}\n")

    def log_evaluation(self, epoch, eval_metrics):
        """
        Log evaluation metrics

        Args:
            epoch: epoch number
            eval_metrics: dict of evaluation metrics
        """
        # Flatten nested dict
        for category, metrics in eval_metrics.items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    full_key = f"{category}_{key}"
                    if value is not None:
                        self.history[full_key].append((epoch, value))
            else:
                self.history[category].append((epoch, metrics))

    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation loss curves
        
        ✅ 재개 학습 시 이전 데이터(첫 epoch) + 현재 데이터를 함께 표시

        Args:
            save_path: path to save plot (if None, use log_dir)
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'training_curves.png')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        if len(self.history['train_loss']) > 0:
            epochs = self.history['epoch']
            axes[0, 0].plot(epochs, self.history['train_loss'], 
                           label='Train Loss', marker='o', linewidth=2, color='#1f77b4')
            axes[0, 0].plot(epochs, self.history['val_loss'], 
                           label='Val Loss', marker='s', linewidth=2, color='#ff7f0e')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss (전체 학습 기록)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        if len(self.history['learning_rate']) > 0:
            epochs = self.history['epoch']
            axes[0, 1].plot(epochs, self.history['learning_rate'], 
                           marker='o', color='green', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule (전체 학습 기록)')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # Tau schedule
        if len(self.history['tau']) > 0:
            epochs = self.history['epoch']
            axes[1, 0].plot(epochs, self.history['tau'], 
                           marker='o', color='orange', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Tau (EMA Momentum)')
            axes[1, 0].set_title('Tau Schedule (전체 학습 기록)')
            axes[1, 0].grid(True, alpha=0.3)

        # Collapse detection
        if len(self.history['feat_std']) > 0:
            epochs = self.history['epoch']
            ax1 = axes[1, 1]
            ax1.plot(epochs, self.history['feat_std'], 
                    marker='o', color='blue', linewidth=2, label='Feature Std')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Feature Std', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(epochs, self.history['avg_cos_sim'], 
                    marker='s', color='red', linewidth=2, label='Avg Cos Sim')
            ax2.set_ylabel('Avg Cosine Similarity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
            
            axes[1, 1].set_title('Collapse Detection (전체 학습 기록)')

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training curves saved to {save_path}")

        plt.close()

    def plot_evaluation_metrics(self, save_path=None):
        """
        Plot evaluation metrics over time
        
        ✅ 재개 학습 시 이전 데이터 + 현재 데이터 함께 표시

        Args:
            save_path: path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'evaluation_metrics.png')

        # Find evaluation metrics
        eval_keys = [k for k in self.history.keys() 
                     if k.startswith(('retrieval_', 'clustering_', 'rotation_'))]

        if len(eval_keys) == 0:
            print("ℹ️  No evaluation metrics to plot")
            return

        # Create subplots
        n_metrics = len(eval_keys)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, key in enumerate(eval_keys):
            if i >= len(axes):
                break

            data = self.history[key]
            if len(data) > 0:
                epochs, values = zip(*data)
                axes[i].plot(epochs, values, marker='o', linewidth=2, color='#1f77b4')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].set_title(key.replace('_', ' ').title() + ' (전체 학습 기록)')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(eval_keys), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Evaluation metrics saved to {save_path}")

        plt.close()

    def save_history(self, save_path=None):
        """
        Save history to JSON (이전 + 현재 데이터 통합)

        Args:
            save_path: path to save history
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'history.json')

        # ✅ NumpyEncoder로 자동 변환
        with open(save_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2, cls=NumpyEncoder)

        print(f"✅ History saved to {save_path} ({len(self.history['epoch'])} epochs)")

    def load_history(self, load_path=None):
        """
        Load history from JSON

        Args:
            load_path: path to load history from
        """
        if load_path is None:
            load_path = os.path.join(self.log_dir, 'history.json')

        with open(load_path, 'r') as f:
            history_dict = json.load(f)

        self.history = defaultdict(list, history_dict)

        print(f"✅ History loaded from {load_path} ({len(self.history['epoch'])} epochs)")

    def print_summary(self):
        """
        Print training summary (전체 학습 기록)
        """
        print("\n" + "="*60)
        print("TRAINING SUMMARY (전체 학습 기록)")
        print("="*60)

        if len(self.history['epoch']) > 0:
            last_epoch = self.history['epoch'][-1]
            best_train_loss = min(self.history['train_loss'])
            best_val_loss = min(self.history['val_loss'])

            print(f"\nTotal Epochs:        {int(last_epoch) + 1}")
            print(f"First Epoch:         {int(self.history['epoch'][0])}")
            print(f"Last Epoch:          {int(last_epoch)}")
            print(f"Best Train Loss:     {best_train_loss:.6f}")
            print(f"Best Val Loss:       {best_val_loss:.6f}")

            if len(self.collapse_warnings) > 0:
                print(f"\nCollapse Warnings:   {len(self.collapse_warnings)}")
                for warning in self.collapse_warnings[-3:]:  # Show last 3
                    print(f"  - {warning}")

        print("\n" + "="*60)

    def should_evaluate(self, epoch):
        """
        Check if should perform evaluation this epoch

        Args:
            epoch: current epoch

        Returns:
            True if should evaluate
        """
        return (epoch + 1) % self.eval_frequency == 0

    def get_stats(self):
        """
        Get training statistics
        
        Returns:
            dict with summary stats
        """
        if len(self.history['epoch']) == 0:
            return {}
        
        return {
            'total_epochs': len(self.history['epoch']),
            'first_epoch': int(self.history['epoch'][0]),
            'last_epoch': int(self.history['epoch'][-1]),
            'best_train_loss': float(min(self.history['train_loss'])),
            'best_val_loss': float(min(self.history['val_loss'])),
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
        }


def visualize_latent_space(features, labels=None, method='tsne', save_path=None, title='Latent Space'):
    """
    Visualize latent space using dimensionality reduction

    Args:
        features: (N, D) features
        labels: (N,) cluster labels (optional)
        method: 'tsne' or 'umap'
        save_path: path to save plot
        title: plot title
    """
    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)

    else:
        raise ValueError(f"Invalid method: {method}")

    # Plot
    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Color by cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                # Noise points
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c='gray', s=10, alpha=0.3, label='Noise')
            else:
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           s=20, alpha=0.6, label=f'Cluster {label}')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        title = f"{title} ({n_clusters} clusters)"

    else:
        # No labels, just scatter
        plt.scatter(features_2d[:, 0], features_2d[:, 1], s=10, alpha=0.5)

    plt.title(title)
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Latent space visualization saved to {save_path}")

    plt.close()


def test_monitor():
    """Test monitoring system with resume"""
    print("Testing BYOL monitor with resume support...")

    # First training (10 epochs)
    print("\n" + "="*60)
    print("Phase 1: Initial training (10 epochs)")
    print("="*60)
    
    monitor = BYOLMonitor(log_dir='test_logs', eval_frequency=5, resume=False)

    for epoch in range(10):
        train_loss = 2.0 - epoch * 0.05 + np.random.randn() * 0.1
        val_loss = 2.1 - epoch * 0.04 + np.random.randn() * 0.1
        lr = 0.001 * (0.9 ** (epoch // 5))
        tau = 0.996 + epoch * 0.0001

        monitor.log_epoch(epoch, train_loss, val_loss, lr, tau)

        feat_std = 1.0 - epoch * 0.03
        avg_cos_sim = 0.1 + epoch * 0.02
        is_collapsed = feat_std < 0.3
        monitor.log_collapse_detection(epoch, feat_std, avg_cos_sim, is_collapsed)

    monitor.plot_training_curves()
    monitor.save_history()

    # Resume training (10 more epochs)
    print("\n" + "="*60)
    print("Phase 2: Resume training (10 more epochs)")
    print("="*60)
    
    monitor_resumed = BYOLMonitor(log_dir='test_logs', eval_frequency=5, resume=True)
    # ✅ 자동으로 이전 10 epochs 로드됨!

    for epoch in range(10, 20):
        train_loss = 1.0 - (epoch-10) * 0.03 + np.random.randn() * 0.05
        val_loss = 1.1 - (epoch-10) * 0.025 + np.random.randn() * 0.05
        lr = 0.0005 * (0.9 ** ((epoch-10) // 5))
        tau = 0.997 + (epoch-10) * 0.0001

        monitor_resumed.log_epoch(epoch, train_loss, val_loss, lr, tau)

        feat_std = 0.7 - (epoch-10) * 0.02
        avg_cos_sim = 0.3 + (epoch-10) * 0.01
        is_collapsed = feat_std < 0.3
        monitor_resumed.log_collapse_detection(epoch, feat_std, avg_cos_sim, is_collapsed)

    monitor_resumed.plot_training_curves()
    monitor_resumed.save_history()
    monitor_resumed.print_summary()

    print("\n✅ Monitor test passed!")
    print(f"📊 Total epochs in history: {len(monitor_resumed.history['epoch'])}")
    print(f"   First epoch: {int(monitor_resumed.history['epoch'][0])}")
    print(f"   Last epoch: {int(monitor_resumed.history['epoch'][-1])}")


if __name__ == "__main__":
    test_monitor()