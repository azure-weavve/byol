"""
BYOL Training Monitor with Variance Tracking

Extended to track variance regularization metrics
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


class BYOLMonitor:
    """
    Monitor BYOL training with variance regularization
    
    Tracks:
    - Training/validation loss
    - BYOL loss, Variance loss (분리)
    - Feature Std, Avg Cosine Similarity
    - Variance weight (curriculum)
    - Clustering metrics
    - Rotation invariance
    """
    
    def __init__(self, log_dir='logs', eval_frequency=5, save_plots=True, resume=False):
        self.log_dir = log_dir
        self.eval_frequency = eval_frequency
        self.save_plots = save_plots
        
        os.makedirs(log_dir, exist_ok=True)
        
        # History
        if resume and os.path.exists(os.path.join(log_dir, 'history.json')):
            self.load_history()
            print(f"✅ Resumed from existing history ({len(self.history['epoch'])} epochs)")
        else:
            self.history = {
                'epoch': [],
                'train_loss': [],
                'val_loss': [],
                'learning_rate': [],
                'tau': [],
                # 🆕 Variance tracking
                'byol_loss': [],
                'variance_loss': [],
                'variance_weight': [],
                'feature_std': [],
                'avg_cos_sim': [],
                'target_std': [],
                'covariance_loss': [],      # 🆕
                'covariance_weight': [],    # 🆕
                # Collapse detection
                'feat_std_collapse': [],
                'avg_cos_sim_collapse': [],
                'is_collapsed': [],
                # Evaluation
                'silhouette': [],
                'n_clusters': [],
                'noise_ratio': [],
                'rotation_invariance': []
            }
    
    def log_epoch(self, epoch, train_loss, val_loss, lr, tau):
        """Log basic epoch metrics"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['tau'].append(tau)
    
    def log_variance_metrics(self, epoch, byol_loss, variance_loss, variance_weight,
                            feature_std, avg_cos_sim, target_std=1.0, covariance_loss=0.0, covariance_weight=0.0):
        """
        🆕 Log variance regularization metrics
        
        Args:
            epoch: current epoch
            byol_loss: BYOL loss component
            variance_loss: Variance regularization loss
            variance_weight: weight for variance loss
            feature_std: current feature std
            avg_cos_sim: average cosine similarity
            target_std: target std value
        """
        self.history['byol_loss'].append(byol_loss)
        self.history['variance_loss'].append(variance_loss)
        self.history['covariance_loss'].append(covariance_loss)      # 🆕
        self.history['variance_weight'].append(variance_weight)
        self.history['covariance_weight'].append(covariance_weight)  # 🆕
        self.history['feature_std'].append(feature_std)
        self.history['avg_cos_sim'].append(avg_cos_sim)
        self.history['target_std'].append(target_std)
    
    def log_collapse_detection(self, epoch, feat_std, avg_cos_sim, is_collapsed):
        """Log collapse detection metrics"""
        self.history['feat_std_collapse'].append(feat_std)
        self.history['avg_cos_sim_collapse'].append(avg_cos_sim)
        self.history['is_collapsed'].append(is_collapsed)
    
    def log_evaluation(self, epoch, metrics):
        """Log evaluation metrics"""
        clustering = metrics.get('clustering', {})
        rotation = metrics.get('rotation_invariance', {})
        
        self.history['silhouette'].append(clustering.get('silhouette'))
        self.history['n_clusters'].append(clustering.get('n_clusters'))
        self.history['noise_ratio'].append(clustering.get('noise_ratio'))
        self.history['rotation_invariance'].append(
            rotation.get('avg_cosine_similarity')
        )
    
    def should_evaluate(self, epoch):
        """Check if should perform evaluation"""
        return (epoch + 1) % self.eval_frequency == 0
    
    def plot_training_curves(self):
        """Plot training curves with variance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = self.history['epoch']
        
        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 🆕 2. BYOL vs Variance Loss
        ax = axes[0, 1]
        if self.history['byol_loss']:
            ax.plot(epochs, self.history['byol_loss'], label='BYOL Loss', linewidth=2, color='blue')
            ax.plot(epochs, self.history['variance_loss'], label='Variance Loss', linewidth=2, color='orange')
            if self.history.get('covariance_loss') and any(v > 0 for v in self.history['covariance_loss']):
                ax.plot(epochs, self.history['covariance_loss'], label='Covariance Loss',
                       linewidth=2, color='green', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 🆕 3. Feature Std with Target
        ax = axes[0, 2]
        if self.history['feature_std']:
            ax.plot(epochs, self.history['feature_std'], label='Feature Std', linewidth=2, color='green')
            if self.history['target_std']:
                ax.axhline(y=self.history['target_std'][0], color='red', linestyle='--', 
                          label=f'Target ({self.history["target_std"][0]:.2f})', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Feature Std')
            ax.set_title('Feature Standard Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 🆕 4. Variance Weight (Curriculum)
        ax = axes[1, 0]
        if self.history['variance_weight']:
            ax.plot(epochs, self.history['variance_weight'], label='Variance Weight', 
                   linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight')
            ax.set_title('Variance Regularization Weight')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 🆕 5. Avg Cosine Similarity
        ax = axes[1, 1]
        if self.history['avg_cos_sim']:
            ax.plot(epochs, self.history['avg_cos_sim'], label='Avg Cos Sim', 
                   linewidth=2, color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title('Average Cosine Similarity')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Learning Rate & Tau
        ax = axes[1, 2]
        ax.plot(epochs, self.history['learning_rate'], label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(epochs, self.history['tau'], label='Tau', linewidth=2, color='orange')
        ax2.set_ylabel('Tau', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
        ax.set_title('Learning Rate & EMA Tau')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.log_dir, 'training_curves.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training curves saved to {save_path}")
        
        plt.close()
    
    def plot_variance_analysis(self):
        """
        🆕 Detailed variance analysis plot
        """
        if not self.history['feature_std']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = self.history['epoch']
        
        # 1. Feature Std over time with zones
        ax = axes[0, 0]
        ax.plot(epochs, self.history['feature_std'], linewidth=2, color='green', label='Feature Std')
        
        # Target zone
        if self.history['target_std']:
            target = self.history['target_std'][0]
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2, label=f'Target ({target:.2f})')
            ax.axhspan(target * 0.9, target * 1.1, alpha=0.2, color='green', label='Good Zone (±10%)')
            ax.axhspan(0, target * 0.7, alpha=0.2, color='red', label='Collapse Zone')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Std')
        ax.set_title('Feature Standard Deviation Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Variance Loss trend
        ax = axes[0, 1]
        if self.history['variance_loss']:
            ax.plot(epochs, self.history['variance_loss'], linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Variance Loss')
            ax.set_title('Variance Regularization Loss')
            ax.grid(True, alpha=0.3)
        
        # 3. Scatter: Feature Std vs Silhouette
        ax = axes[1, 0]
        if self.history['silhouette']:
            valid_epochs = [i for i, s in enumerate(self.history['silhouette']) if s is not None]
            if valid_epochs:
                feat_stds = [self.history['feature_std'][epochs.index(self.history['epoch'][i])] 
                            for i in valid_epochs]
                silhouettes = [self.history['silhouette'][i] for i in valid_epochs]
                
                scatter = ax.scatter(feat_stds, silhouettes, c=valid_epochs, cmap='viridis', s=100)
                ax.set_xlabel('Feature Std')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Feature Std vs Clustering Quality')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Epoch')
        
        # 4. Dual axis: BYOL vs Variance Loss
        ax = axes[1, 1]
        if self.history['byol_loss'] and self.history['variance_loss']:
            ax.plot(epochs, self.history['byol_loss'], linewidth=2, color='blue', label='BYOL Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('BYOL Loss', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.legend(loc='upper left')
            
            ax2 = ax.twinx()
            ax2.plot(epochs, self.history['variance_loss'], linewidth=2, color='orange', label='Var Loss')
            ax2.set_ylabel('Variance Loss', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.legend(loc='upper right')
            ax.set_title('Loss Components Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.log_dir, 'variance_analysis.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Variance analysis saved to {save_path}")
        
        plt.close()
    
    def plot_evaluation_metrics(self):
        """Plot evaluation metrics"""
        # ... 기존 코드 유지 ...
        pass  # 기존 구현 그대로
    
    def save_history(self):
        """Save history to JSON"""
        history_path = os.path.join(self.log_dir, 'history.json')
        
        # Convert to serializable format
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"✅ History saved to {history_path}")
    
    def load_history(self):
        """Load history from JSON"""
        history_path = os.path.join(self.log_dir, 'history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            return True
        return False
    
    def print_summary(self):
        """Print training summary with variance metrics"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        if not self.history['epoch']:
            print("No training history available")
            return
        
        # Best epoch by validation loss
        best_idx = np.argmin(self.history['val_loss'])
        best_epoch = self.history['epoch'][best_idx]
        
        print(f"\n🏆 Best Epoch: {best_epoch + 1}")
        print(f"   Val Loss: {self.history['val_loss'][best_idx]:.6f}")
        
        if self.history['feature_std']:
            print(f"   Feature Std: {self.history['feature_std'][best_idx]:.4f}")
        if self.history['avg_cos_sim']:
            print(f"   Avg Cos Sim: {self.history['avg_cos_sim'][best_idx]:.4f}")
        if self.history['silhouette'] and best_idx < len(self.history['silhouette']):
            sil = self.history['silhouette'][best_idx]
            if sil is not None:
                print(f"   Silhouette: {sil:.4f}")
        
        # Final epoch
        print(f"\n📊 Final Epoch: {self.history['epoch'][-1] + 1}")
        print(f"   Train Loss: {self.history['train_loss'][-1]:.6f}")
        print(f"   Val Loss: {self.history['val_loss'][-1]:.6f}")
        
        if self.history['feature_std']:
            feat_std = self.history['feature_std'][-1]
            target_std = self.history['target_std'][-1] if self.history['target_std'] else 1.0
            deviation = abs(feat_std - target_std) / target_std * 100
            print(f"   Feature Std: {feat_std:.4f} (target: {target_std:.2f}, deviation: {deviation:.1f}%)")
        
        if self.history['avg_cos_sim']:
            print(f"   Avg Cos Sim: {self.history['avg_cos_sim'][-1]:.4f}")
        
        print("="*80)

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
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        features_2d = reducer.fit_transform(features)

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
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