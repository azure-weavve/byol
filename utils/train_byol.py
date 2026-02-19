"""
Training loop for BYOL

Functions:
- train_byol_epoch: single epoch training
- validate_byol_epoch: validation
- train_byol: full training pipeline

PyTorch 1.4.0 compatible
"""

import torch
import torch.nn as nn
import time


def compute_variance_loss(features):
    """
    Feature variance를 유지하도록 regularization
    
    Args:
        features: [batch_size, dim]
    
    Returns:
        loss: variance가 낮으면 패널티
    """
    # 각 dimension의 std 계산
    std_per_dim = features.std(dim=0)  # (dim,)
    
    # Std가 낮으면 loss 증가
    # -log(std)를 사용하면 std가 0에 가까울수록 loss 무한대
    variance_loss = -torch.log(std_per_dim.mean() + 1e-6)
    
    return variance_loss

def compute_variance_loss_target_std(features, target_std=1.0, adaptive=False):
    """
    Feature std를 target 값에 맞추는 loss
    
    Args:
        features: [batch_size, dim]
        target_std: 목표 std (default: 1.0)
        adaptive: True면 dimension별로 다른 target 허용
    
    Returns:
        loss: scalar
        current_std: 현재 평균 std
    """
    # Dimension별 std 계산
    std_per_dim = features.std(dim=0)  # (dim,)
    
    if adaptive:
        # Dimension별로 조정 (일부 dim은 중요도 높음)
        # 현재는 단순 평균
        avg_std = std_per_dim.mean()
    else:
        avg_std = std_per_dim.mean()
    
    # MSE loss to target
    loss = (avg_std - target_std) ** 2
    
    return loss, avg_std.item()


def compute_variance_loss_robust(features, target_std=1.0, margin=0.1):
    """
    더 robust한 버전: margin 내에서는 패널티 없음
    
    target_std = 1.0, margin = 0.1이면
    → 0.9 ~ 1.1 범위는 패널티 0
    → 범위 밖이면 패널티
    """
    std_per_dim = features.std(dim=0)
    avg_std = std_per_dim.mean()
    
    # Margin 밖이면 패널티
    lower_bound = target_std - margin
    upper_bound = target_std + margin
    
    if avg_std < lower_bound:
        loss = (avg_std - lower_bound) ** 2
    elif avg_std > upper_bound:
        loss = (avg_std - upper_bound) ** 2
    else:
        loss = torch.tensor(0.0, device=features.device)
    
    return loss, avg_std.item()

def compute_covariance_loss(features):
    """
    VICReg style covariance regularization
    차원 간 상관관계를 제거하여 directional collapse 방지
    
    Args:
        features: [batch_size, dim]
    Returns:
        loss: scalar
    """
    N, D = features.shape
    
    # 평균 제거
    features_centered = features - features.mean(dim=0)
    
    # Covariance matrix (D, D)
    cov = (features_centered.T @ features_centered) / max(N - 1, 1)
    
    # Off-diagonal 원소의 제곱합
    diag = torch.diag(torch.diag(cov))
    off_diag = cov - diag
    
    loss = (off_diag ** 2).sum() / D
    
    return loss


def train_byol_epoch(model, dataloader, optimizer, device, tau, augmentation, epoch=0, total_epochs=100, variance_config=None, verbose=True):
    """
    Train BYOL for one epoch

    Args:
        model: BYOL model
        dataloader: training data loader
        optimizer: optimizer (e.g., AdamW)
        device: torch device
        tau: EMA momentum for this epoch
        augmentation: augmentation function (BYOLAugmentation)
        epoch: current epoch number
        verbose: print progress
        variance_config: {
            'type': 'target_std',  # or 'target_std_robust'
            'target_std': 1.0,
            'margin': 0.1,         # for robust version
            'weight': 0.2
        }

    Returns:
        avg_loss: average loss for the epoch
        avg_cos_sim: average cosine similarity
    """
    model.train()

    variance_type = variance_config.get('type', 'original')
    variance_weight = variance_config.get('weight', 0.0)
    covariance_weight = variance_config.get('covariance_weight', 0.0)

    total_loss = 0.0
    total_byol_loss = 0.0
    total_var_loss = 0.0
    total_cov_loss = 0.0  # 기존 total_var_loss 아래에 추가
    total_feat_std = 0.0
    total_cos_sim = 0.0
    total_batches = 0

    total_batches_count = len(dataloader)

    for batch_idx, data in enumerate(dataloader):
        # Get data (handle multiple return formats)
        if isinstance(data, (list, tuple)):
            if len(data) == 4:
                # From dataloader_utils: (images, images_aug, labels, indices)
                images = data[0]
                # Ignore data[1] (augmented), data[2] (labels), data[3] (indices)
                # BYOL will apply its own augmentation
            else:
                # Standard format: (images, labels) or just images
                images = data[0] if len(data) > 0 else data
        else:
            images = data

        images = images.to(device)
        batch_size = images.size(0)

        # Generate two augmented views
        view1_list = []
        view2_list = []

        for i in range(batch_size):
            v1, v2 = augmentation(images[i])
            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list).to(device)
        view2 = torch.stack(view2_list).to(device)

        # Forward pass
        optimizer.zero_grad()

        # 1. BYOL loss (+ features for regularization)
        if variance_weight >0 or covariance_weight > 0:
            byol_loss, encoder_features, projector_features = model(
                view1, view2, return_projections=True
            )

            # Variance loss — encoder output (512d)
            if variance_weight > 0:
                if variance_type == 'target_std':
                    var_loss, current_std = compute_variance_loss_target_std(
                        encoder_features, 
                        target_std=variance_config.get('target_std', 1.0)
                    )
                elif variance_type == 'target_std_robust':
                    var_loss, current_std = compute_variance_loss_robust(
                        encoder_features, 
                        target_std=variance_config.get('target_std', 1.0), 
                        margin=variance_config.get('margin', 0.1)
                    )
                else:
                    var_loss = compute_variance_loss(encoder_features)
                    current_std = encoder_features.std(dim=0).mean().item()
            else:
                var_loss = torch.tensor(0.0, device=device)
                current_std = encoder_features.std(dim=0).mean().item()
            
            total_feat_std += current_std

            # Covariance loss — projector output (256d)
            if covariance_weight > 0:
                cov_loss = compute_covariance_loss(projector_features)
            else: 
                cov_loss = torch.tensor(0.0, device=device)

            # Cosine similarity (monitoring용)
            with torch.no_grad(): 
                normalized = encoder_features / (encoder_features.norm(dim=1, keepdim=True) + 1e-8) 
                n_samples = min(100, encoder_features.size(0))
                
                if n_samples >= 2: 
                    indices = torch.randperm(encoder_features.size(0))[:n_samples] 
                    sample_features = normalized[indices] 
                    cos_sim_matrix = torch.mm(sample_features, sample_features.T) 
                    mask = ~torch.eye(n_samples, dtype=torch.bool, device=device) 
                    avg_cos_sim_batch = cos_sim_matrix[mask].mean().item() 
                else: 
                    avg_cos_sim_batch = 0.0 
                total_cos_sim += avg_cos_sim_batch

            # Total loss
            total_loss_batch = byol_loss + variance_weight * var_loss + covariance_weight * cov_loss
        else:
            var_loss = torch.tensor(0.0, device=device)
            cov_loss = torch.tensor(0.0, device=device)  # 🆕
            current_std = 0.0
            avg_cos_sim_batch = 0.0
            total_loss_batch = byol_loss

        # Backward pass
        total_loss_batch.backward()
        optimizer.step()

        # Update target network with EMA
        model.update_target_network(tau=tau)

        # Track metrics
        total_loss += total_loss_batch.item()
        total_byol_loss += byol_loss.item()
        total_var_loss += var_loss.item()
        total_cov_loss += cov_loss.item()  # total_var_loss 아래에 추가
        total_batches += 1

        # Print progress
        if verbose and (batch_idx % 5 == 0 or batch_idx == total_batches_count - 1):
            if variance_weight > 0:
                print(f"Epoch {epoch+1} [Train] [{batch_idx+1}/{total_batches_count}] "
                    f"Total: {total_loss_batch.item():.4f}, "
                      f"BYOL: {byol_loss.item():.4f}, "
                      f"Var: {var_loss.item():.4f}, "
                      f"Cov: {cov_loss.item():.4f}, "
                      f"FeatStd: {current_std:.4f}, "
                      f"Tau: {tau:.4f}")
            else:
                print(f"Epoch {epoch+1} [Train] [{batch_idx+1}/{total_batches_count}] "
                    f"Loss: {total_loss_batch.item():.4f}")

    # Calculate averages
    avg_total_loss = total_loss / total_batches
    avg_byol_loss = total_byol_loss / total_batches
    avg_var_loss = total_var_loss / total_batches
    avg_cov_loss = total_cov_loss / total_batches
    has_reg = (variance_weight > 0 or covariance_weight > 0)
    avg_feat_std = total_feat_std / total_batches if has_reg else 0.0
    avg_cos_sim = total_cos_sim / total_batches if has_reg else 0.0
    
    return avg_total_loss, avg_byol_loss, avg_var_loss, avg_cov_loss, avg_feat_std, avg_cos_sim


def validate_byol_epoch(model, dataloader, device, augmentation, verbose=True):
    """
    Validate BYOL for one epoch

    Args:
        model: BYOL model
        dataloader: validation data loader
        device: torch device
        augmentation: augmentation function
        verbose: print progress

    Returns:
        avg_loss: average validation loss
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_batches_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get data (handle multiple return formats)
            if isinstance(data, (list, tuple)):
                if len(data) == 4:
                    # From dataloader_utils: (images, images_aug, labels, indices)
                    images = data[0]
                else:
                    # Standard format: (images, labels) or just images
                    images = data[0] if len(data) > 0 else data
            else:
                images = data

            images = images.to(device)
            batch_size = images.size(0)

            # Generate two augmented views
            view1_list = []
            view2_list = []

            for i in range(batch_size):
                v1, v2 = augmentation(images[i])
                view1_list.append(v1)
                view2_list.append(v2)

            view1 = torch.stack(view1_list).to(device)
            view2 = torch.stack(view2_list).to(device)

            # Forward pass
            loss = model(view1, view2)

            # Track metrics
            total_loss += loss.item()
            total_batches += 1

            # Print progress
            if verbose and (batch_idx % 10 == 0 or batch_idx == total_batches_count - 1):
                print(f"Validation [{batch_idx+1}/{total_batches_count}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_batches

    return avg_loss


def extract_features(model, dataloader, device, use_target=True, verbose=True):
    """
    Extract features from all data

    Args:
        model: BYOL model
        dataloader: data loader
        device: torch device
        use_target: use target encoder (recommended)
        verbose: print progress

    Returns:
        features: (N, D) tensor of all features
        labels: (N,) tensor of labels (if available)
    """
    model.eval()

    all_features = []
    all_labels = []

    total_batches_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get data (handle multiple return formats)
            if isinstance(data, (list, tuple)):
                if len(data) == 4:
                    # From dataloader_utils: (images, images_aug, labels, indices)
                    images = data[0]
                    labels = data[2]  # labels는 list!
                elif len(data) > 1:
                    images = data[0]
                    labels = data[1]
                else:
                    images = data[0] if len(data) > 0 else data
                    labels = None
            else:
                images = data
                labels = None

            images = images.to(device)

            # Extract features
            features = model.get_embeddings(images, use_target=use_target)

            all_features.append(features.cpu())
            
            # ✅ 수정: labels를 올바르게 처리
            if labels is not None:
                # labels가 list라면 tensor로 변환 필요
                if isinstance(labels, list):
                    # 숫자로 변환 가능하면 tensor로
                    try:
                        labels_tensor = torch.tensor(labels, dtype=torch.long)
                        all_labels.append(labels_tensor)
                    except (ValueError, TypeError):
                        # 문자열 라벨이면 그냥 list로 유지
                        all_labels.extend(labels)
                elif isinstance(labels, torch.Tensor):
                    all_labels.append(labels.cpu())
                else:
                    # 다른 형식이면 list로 추가
                    if isinstance(labels, (list, tuple)):
                        all_labels.extend(labels)
                    else:
                        all_labels.append(labels)

            # Print progress
            if verbose and (batch_idx % 10 == 0 or batch_idx == total_batches_count - 1):
                print(f"Extracting features [{batch_idx+1}/{total_batches_count}]")

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)

    # ✅ 수정: labels 처리
    if len(all_labels) > 0:
        # 첫 번째 원소가 tensor인지 list인지 확인
        if isinstance(all_labels[0], torch.Tensor):
            all_labels = torch.cat(all_labels, dim=0)
            return all_features, all_labels
        else:
            # list of strings/mixed types
            return all_features, all_labels
    else:
        return all_features, None


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, best_val_loss=None, **kwargs):
    """
    Save checkpoint

    Args:
        model: BYOL model
        optimizer: optimizer
        scheduler: learning rate scheduler (optional)
        epoch: current epoch
        loss: current loss
        filepath: path to save checkpoint
        best_val_loss: best validation loss so far (optional)  # 🔴 추가
        **kwargs: additional info to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    # 🔴 best_val_loss 저장
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss

    # Add any additional info
    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device):
    """
    Load checkpoint

    Args:
        model: BYOL model
        optimizer: optimizer
        scheduler: learning rate scheduler (optional)
        filepath: path to checkpoint
        device: torch device

    Returns:
        epoch: epoch to resume from
        loss: loss at checkpoint
        best_val_loss: best validation loss (if available)
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # 🔴 추가

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch+1}, loss: {loss:.4f}, best_val_loss: {best_val_loss:.4f}")

    return epoch, loss, best_val_loss  # 🔴 3개 반환


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: current metric value

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def detect_collapse(features, threshold_std=0.01, threshold_cosine=0.99):
    """
    Detect representation collapse

    Args:
        features: (B, D) feature tensor
        threshold_std: minimum std for each dimension
        threshold_cosine: maximum average cosine similarity

    Returns:
        is_collapsed: bool
        info: dict with diagnostic info
    """
    # Compute statistics
    feat_std = features.std(dim=0).mean().item()
    feat_mean = features.mean(dim=0).mean().item()

    # Normalize features
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    # Compute pairwise cosine similarity
    cos_sim_matrix = torch.mm(features_norm, features_norm.t())

    # Average cosine similarity (excluding diagonal)
    mask = ~torch.eye(cos_sim_matrix.size(0), dtype=torch.bool, device=features.device)
    avg_cos_sim = cos_sim_matrix[mask].mean().item()

    # Check collapse
    is_collapsed = (feat_std < threshold_std) or (avg_cos_sim > threshold_cosine)

    info = {
        'feat_std': feat_std,
        'feat_mean': feat_mean,
        'avg_cos_sim': avg_cos_sim,
        'is_collapsed': is_collapsed
    }

    return is_collapsed, info


def log_training_info(epoch, train_loss, val_loss, learning_rate, tau, elapsed_time, collapse_info=None):
    """
    Log training information

    Args:
        epoch: current epoch
        train_loss: training loss
        val_loss: validation loss
        learning_rate: current learning rate
        tau: current tau value
        elapsed_time: time elapsed for epoch
        collapse_info: collapse detection info (optional)
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Summary")
    print(f"{'='*60}")
    print(f"Train Loss: {train_loss:.6f} / Val Loss: {val_loss:.6f}")
    print(f"Learning Rate: {learning_rate:.6e} / Tau (EMA): {tau:.6f}")
    print(f"Time: {elapsed_time:.2f}s")

    if collapse_info is not None:
        print(f"\nCollapse Detection:")
        print(f"  Feature Std: {collapse_info['feat_std']:.6f} / Avg Cos Sim: {collapse_info['avg_cos_sim']:.6f} / Collapsed: {collapse_info['is_collapsed']}")

    print(f"{'='*60}\n")


def test_training_functions():
    """Test training functions"""
    print("Testing training functions...")

    # Create dummy model and data
    from models.byol import BYOL
    from utils.augmentation import get_byol_augmentation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = BYOL(
        encoder_dim=512,
        projector_hidden=1024,
        projector_out=256,
        predictor_hidden=1024,
        use_radial_encoding=True,
        use_attention=True,
        wafer_size=(128, 128),
        tau=0.996
    ).to(device)

    # Create dummy dataloader
    dummy_data = torch.randn(32, 1, 128, 128)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=4,
        shuffle=True
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create augmentation
    augmentation = get_byol_augmentation('strong')

    # Test training for one epoch
    print("\nTesting training epoch...")
    avg_loss = train_byol_epoch(
        model, dataloader, optimizer, device,
        tau=0.996, augmentation=augmentation, epoch=0
    )
    print(f"Average training loss: {avg_loss:.4f}")

    # Test validation
    print("\nTesting validation...")
    val_loss = validate_byol_epoch(model, dataloader, device, augmentation)
    print(f"Validation loss: {val_loss:.4f}")

    # Test feature extraction
    print("\nTesting feature extraction...")
    features, _ = extract_features(model, dataloader, device, use_target=True)
    print(f"Extracted features shape: {features.shape}")

    # Test collapse detection
    print("\nTesting collapse detection...")
    is_collapsed, info = detect_collapse(features)
    print(f"Collapsed: {is_collapsed}")
    print(f"Info: {info}")

    print("\nTraining functions test passed!")


if __name__ == "__main__":
    # Note: This will only work if models and utils are properly set up
    try:
        test_training_functions()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from project root with proper PYTHONPATH")