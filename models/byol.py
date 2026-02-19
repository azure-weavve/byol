"""
BYOL (Bootstrap Your Own Latent) Model

Complete implementation integrating:
- Online network (encoder + projector + predictor)
- Target network (encoder + projector, EMA updated)
- Loss computation
- EMA update mechanism

PyTorch 1.4.0 compatible
"""

import torch
import torch.nn as nn
import copy
from .encoder import WaferEncoder
from .projector import Projector, Predictor, symmetric_byol_loss


class BYOL(nn.Module):
    """
    BYOL model for wafer map pattern learning

    Architecture:
        Online network:
            encoder_online → projector_online → predictor
        Target network (EMA):
            encoder_target → projector_target

    Training:
        1. Two augmented views: v1, v2
        2. Online forward: p1, p2
        3. Target forward: z1, z2 (no grad)
        4. Loss: symmetric_loss(p1, z2, p2, z1)
        5. Backprop on online network only
        6. EMA update target network
    """
    def __init__(self,
                 input_channels=13,          # 🔴 추가
                 encoder_dim=512,
                 projector_hidden=1024,
                 projector_out=256,
                 predictor_hidden=1024,
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128),
                 tau=0.996):
        super(BYOL, self).__init__()

        self.encoder_dim = encoder_dim
        self.projector_out = projector_out
        self.tau = tau

        # === Online Network (trainable) ===
        self.encoder_online = WaferEncoder(
            input_channels=input_channels,  # 🔴 전달
            output_dim=encoder_dim,
            use_radial_encoding=use_radial_encoding,
            use_attention=use_attention,
            wafer_size=wafer_size
        )

        self.projector_online = Projector(
            input_dim=encoder_dim,
            hidden_dim=projector_hidden,
            output_dim=projector_out
        )

        self.predictor = Predictor(
            input_dim=projector_out,
            hidden_dim=predictor_hidden,
            output_dim=projector_out
        )

        # === Target Network (EMA, no grad) ===
        self.encoder_target = WaferEncoder(
            input_channels=input_channels,  # 🔴 전달
            output_dim=encoder_dim,
            use_radial_encoding=use_radial_encoding,
            use_attention=use_attention,
            wafer_size=wafer_size
        )

        self.projector_target = Projector(
            input_dim=encoder_dim,
            hidden_dim=projector_hidden,
            output_dim=projector_out
        )

        # Initialize target as copy of online
        self._initialize_target_network()

        # Disable gradients for target network
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        for param in self.projector_target.parameters():
            param.requires_grad = False

    def _initialize_target_network(self):
        """Initialize target network as exact copy of online network"""
        # Copy encoder
        self.encoder_target.load_state_dict(self.encoder_online.state_dict())
        # Copy projector
        self.projector_target.load_state_dict(self.projector_online.state_dict())

    @torch.no_grad()
    def update_target_network(self, tau=None):
        """
        Exponential Moving Average update of target network
        Args:
            tau: momentum coefficient (0.996 ~ 0.999)
                 If None, use self.tau
            Update rule:
                ξ ← τ·ξ + (1-τ)·θ
        """
        if tau is None:
            tau = self.tau

        # Update encoder
        for online_params, target_params in zip(
            self.encoder_online.parameters(),
            self.encoder_target.parameters()
        ):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

        # Update projector
        for online_params, target_params in zip(
            self.projector_online.parameters(),
            self.projector_target.parameters()
        ):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    def forward(self, view1, view2, return_projections=False):
        """
        Forward pass for training
        Args:
            view1: (B, 1, H, W) first augmented view
            view2: (B, 1, H, W) second augmented view
        Returns:
            loss: BYOL loss (symmetric)
            Optional: predictions and projections for monitoring
        """
        # === Online network forward ===
        # View 1
        feat1_online = self.encoder_online(view1)
        proj1_online = self.projector_online(feat1_online)
        pred1 = self.predictor(proj1_online)

        # View 2
        feat2_online = self.encoder_online(view2)
        proj2_online = self.projector_online(feat2_online)
        pred2 = self.predictor(proj2_online)

        # === Target network forward (no grad) ===
        with torch.no_grad():
            # View 1
            feat1_target = self.encoder_target(view1)
            proj1_target = self.projector_target(feat1_target)

            # View 2
            feat2_target = self.encoder_target(view2)
            proj2_target = self.projector_target(feat2_target)

        # === Compute symmetric loss ===
        loss = symmetric_byol_loss(pred1, proj2_target, pred2, proj1_target)

        if return_projections:
            encoder_features = torch.cat([feat1_online, feat2_online])
            projector_features = torch.cat([proj1_online, proj2_online])
            return loss, encoder_features, projector_features
        return loss
    
    def encode(self, x, use_target=False):
        """
        Extract features for inference/evaluation

        Args:
            x: (B, 1, H, W) wafer map
            use_target: if True, use target encoder (recommended after training)
            Returns:
            (B, encoder_dim) features
        """
        if use_target:
            with torch.no_grad():
                return self.encoder_target(x)
        else:
            return self.encoder_online(x)
        
    def get_embeddings(self, x, use_target=True):
        """
        Get embeddings for clustering/retrieval
        Args:
            x: (B, 1, H, W) wafer map
            use_target: use target encoder (more stable)
        Returns:
            (B, encoder_dim) embeddings
        """
        with torch.no_grad():
            if use_target:
                features = self.encoder_target(x)
            else:
                features = self.encoder_online(x)
        return features
    
    def forward_with_monitoring(self, view1, view2):
        """
        Forward pass with additional monitoring information
        Returns:
            loss, monitoring_dict
        """
        # === Online network forward ===
        feat1_online = self.encoder_online(view1)
        proj1_online = self.projector_online(feat1_online)
        pred1 = self.predictor(proj1_online)

        feat2_online = self.encoder_online(view2)
        proj2_online = self.projector_online(feat2_online)
        pred2 = self.predictor(proj2_online)

        # === Target network forward ===
        with torch.no_grad():
            feat1_target = self.encoder_target(view1)
            proj1_target = self.projector_target(feat1_target)

            feat2_target = self.encoder_target(view2)
            proj2_target = self.projector_target(feat2_target)

        # === Compute loss ===
        loss = symmetric_byol_loss(pred1, proj2_target, pred2, proj1_target)

        # === Monitoring info ===
        with torch.no_grad():
            # Feature statistics
            feat_mean = feat1_online.mean().item()
            feat_std = feat1_online.std().item()

            # Projection statistics
            proj_mean = proj1_online.mean().item()
            proj_std = proj1_online.std().item()

            # Cosine similarity between predictions and targets
            pred1_norm = pred1 / (pred1.norm(dim=1, keepdim=True) + 1e-8)
            proj2_norm = proj2_target / (proj2_target.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim = (pred1_norm * proj2_norm).sum(dim=1).mean().item()
        
        monitoring_dict = {
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'proj_mean': proj_mean,
            'proj_std': proj_std,
            'cos_similarity': cos_sim
        }

        return loss, monitoring_dict
    
def get_tau_schedule(epoch, total_epochs, tau_base=0.996, tau_max=0.999):
    """
    Cosine schedule for tau (EMA momentum)
    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        tau_base: initial tau value
        tau_max: final tau value
    Returns:
        tau value for current epoch
    Strategy:
    - Early training: tau low → target updates faster
    - Late training: tau high → target more stable
    """
    import math

    progress = epoch / total_epochs
    tau = tau_max - (tau_max - tau_base) * (1 + math.cos(math.pi * progress)) / 2

    return tau
    

def test_byol():
    """Test BYOL model"""
    print("Testing BYOL model...")
    model = BYOL(
        encoder_dim=512,
        projector_hidden=1024,
        projector_out=256,
        predictor_hidden=1024,
        use_radial_encoding=True,
        use_attention=True,
        wafer_size=(128, 128),
        tau=0.996
    )

    # Count parameters
    online_params = (
        sum(p.numel() for p in model.encoder_online.parameters()) +
        sum(p.numel() for p in model.projector_online.parameters()) +
        sum(p.numel() for p in model.predictor.parameters())
    )

    target_params = (
        sum(p.numel() for p in model.encoder_target.parameters()) +
        sum(p.numel() for p in model.projector_target.parameters())
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Online network parameters: {online_params:,}")
    print(f"Target network parameters: {target_params:,}")
    print(f"Total parameters: {online_params + target_params:,}")

    # Test forward pass
    batch_size = 4
    view1 = torch.randn(batch_size, 13, 128, 128)
    view2 = torch.randn(batch_size, 13, 128, 128)

    print(f"\nInput shapes: {view1.shape}, {view2.shape}")

    # Training mode
    model.train()
    loss = model(view1, view2)
    print(f"Training loss: {loss.item():.4f}")

    # Test with monitoring
    loss, monitoring = model.forward_with_monitoring(view1, view2)
    print(f"\nMonitoring info:")
    for key, value in monitoring.items():
        print(f"  {key}: {value:.4f}")

    # Test EMA update
    tau_before = model.tau
    model.update_target_network(tau=0.999)
    print(f"\nEMA update with tau={0.999:.3f} completed")

    # Test inference
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(view1, use_target=True)
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Test tau schedule
    print("\nTau schedule (100 epochs):")
    for epoch in [0, 10, 30, 50, 70, 99]:
        tau = get_tau_schedule(epoch, 100, tau_base=0.996, tau_max=0.999)
        print(f"  Epoch {epoch:3d}: tau = {tau:.6f}")

    print("\nBYOL model test passed!")

if __name__ == "__main__":
    test_byol()