"""
BYOL Training Monitor

Tracks and logs:
- Training/validation loss
- Learning rate and tau schedule
- Feature statistics (for collapse detection)
- Periodic evaluation metrics
- Visualization (t-SNE/UMAP)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os


class BYOLMonitor:
    """
    Monitor BYOL training progress
    """
    def __init__(self, log_dir='logs', eval_frequency=10, save_plots=True):
        """
        Args:
            log_dir: directory to save logs and plots
            eval_frequency: evaluate every N epochs
            save_plots: save plots to disk
        """
        self.log_dir = log_dir
        self.eval_frequency = eval_frequency
        self.save_plots = save_plots

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # History
        self.history = defaultdict(list)

        # Collapse detection
        self.collapse_warnings = []

    def log_epoch(self, epoch, train_loss, val_loss, learning_rate, tau, **kwargs):
        """
        Log epoch metrics

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

        Args:
            save_path: path to save plot (if None, use log_dir)
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'training_curves.png')

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        if len(self.history['train_loss']) > 0:
            epochs = self.history['epoch']
            axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
            axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # Learning rate
        if len(self.history['learning_rate']) > 0:
            epochs = self.history['epoch']
            axes[0, 1].plot(epochs, self.history['learning_rate'], marker='o', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)

        # Tau schedule
        if len(self.history['tau']) > 0:
            epochs = self.history['epoch']
            axes[1, 0].plot(epochs, self.history['tau'], marker='o', color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Tau (EMA Momentum)')
            axes[1, 0].set_title('Tau Schedule')
            axes[1, 0].grid(True)

        # Collapse detection
        if len(self.history['feat_std']) > 0:
            epochs = self.history['epoch']
            ax1 = axes[1, 1]
            ax1.plot(epochs, self.history['feat_std'], marker='o', color='blue', label='Feature Std')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Feature Std', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(epochs, self.history['avg_cos_sim'], marker='s', color='red', label='Avg Cos Sim')
            ax2.set_ylabel('Avg Cosine Similarity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            axes[1, 1].set_title('Collapse Detection Metrics')

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.close()

    def plot_evaluation_metrics(self, save_path=None):
        """
        Plot evaluation metrics over time

        Args:
            save_path: path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'evaluation_metrics.png')

        # Find evaluation metrics
        eval_keys = [k for k in self.history.keys() if k.startswith(('retrieval_', 'clustering_', 'rotation_'))]

        if len(eval_keys) == 0:
            print("No evaluation metrics to plot")
            return

        # Create subplots
        n_metrics = len(eval_keys)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, key in enumerate(eval_keys):
            if i >= len(axes):
                break

            data = self.history[key]
            if len(data) > 0:
                epochs, values = zip(*data)
                axes[i].plot(epochs, values, marker='o')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].set_title(key.replace('_', ' ').title())
                axes[i].grid(True)

        # Hide unused subplots
        for i in range(len(eval_keys), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Evaluation metrics saved to {save_path}")

        plt.close()

    def save_history(self, save_path=None):
        """
        Save history to JSON

        Args:
            save_path: path to save history
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'history.json')

        # Convert to serializable format
        history_dict = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                # Convert numpy/torch values to native Python
                history_dict[key] = [
                    float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                    for v in value
                ]
            else:
                history_dict[key] = value

        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"History saved to {save_path}")

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

        print(f"History loaded from {load_path}")

    def print_summary(self):
        """
        Print training summary
        """
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        if len(self.history['epoch']) > 0:
            last_epoch = self.history['epoch'][-1]
            best_train_loss = min(self.history['train_loss'])
            best_val_loss = min(self.history['val_loss'])

            print(f"\nTotal Epochs:        {last_epoch + 1}")
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
        print(f"Latent space visualization saved to {save_path}")

    plt.close()


def test_monitor():
    """Test monitoring system"""
    print("Testing BYOL monitor...")

    monitor = BYOLMonitor(log_dir='test_logs', eval_frequency=5)

    # Simulate training
    for epoch in range(20):
        train_loss = 2.0 - epoch * 0.05 + np.random.randn() * 0.1
        val_loss = 2.1 - epoch * 0.04 + np.random.randn() * 0.1
        lr = 0.001 * (0.9 ** (epoch // 5))
        tau = 0.996 + epoch * 0.0001

        monitor.log_epoch(epoch, train_loss, val_loss, lr, tau)

        # Collapse detection
        feat_std = 1.0 - epoch * 0.03
        avg_cos_sim = 0.1 + epoch * 0.02
        is_collapsed = feat_std < 0.3
        monitor.log_collapse_detection(epoch, feat_std, avg_cos_sim, is_collapsed)

    # Plot
    monitor.plot_training_curves()
    monitor.save_history()
    monitor.print_summary()

    print("\nMonitor test passed!")


if __name__ == "__main__":
    test_monitor()
