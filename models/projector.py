"""
Projector and Predictor for BYOL

PyTorch 1.4.0 compatible implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron base class
    Used for both Projector and Predictor
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bn=True):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bn = use_bn

        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Second layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, output_dim)
        """
        # First layer
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Second layer
        x = self.fc2(x)

        return x


class Projector(nn.Module):
    """
    Projector network: projects encoder features to lower-dimensional space

    Architecture:
        Linear(512, 1024)
        BatchNorm1d(1024)
        ReLU
        Linear(1024, 256)

    Used in both online and target networks
    """
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256):
        super(Projector, self).__init__()

        self.mlp = MLP(input_dim, hidden_dim, output_dim, use_bn=True)

    def forward(self, x):
        """
        Args:
            x: (B, 512) encoder features
        Returns:
            (B, 256) projected features
        """
        return self.mlp(x)


class Predictor(nn.Module):
    """
    Predictor network: predicts target projection from online projection

    Architecture:
        Linear(256, 1024)
        BatchNorm1d(1024)
        ReLU
        Linear(1024, 256)

    IMPORTANT: Only used in online network!
    This asymmetry is crucial for preventing collapse
    """
    def __init__(self, input_dim=256, hidden_dim=1024, output_dim=256):
        super(Predictor, self).__init__()

        self.mlp = MLP(input_dim, hidden_dim, output_dim, use_bn=True)

    def forward(self, x):
        """
        Args:
            x: (B, 256) projected features from online network
        Returns:
            (B, 256) predicted features
        """
        return self.mlp(x)


def normalize_features(x, dim=1, eps=1e-8):
    """
    L2 normalization for PyTorch 1.4.0 compatibility

    Args:
        x: input tensor
        dim: dimension to normalize
        eps: epsilon for numerical stability

    Returns:
        L2-normalized tensor
    """
    # Manual L2 normalization (F.normalize may have issues in PyTorch 1.4.0)
    norm = torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=True))
    return x / (norm + eps)


def byol_loss_function(p, z):
    """
    BYOL loss function: MSE after L2 normalization

    Args:
        p: predictor output from online network (B, D)
        z: projector output from target network (B, D)

    Returns:
        loss: scalar loss value

    Formula:
        L = 2 - 2 * <p, z> / (||p|| * ||z||)
          = 2 * (1 - cosine_similarity(p, z))
    """
    # L2 normalize both vectors
    p = normalize_features(p, dim=1)
    z = normalize_features(z, dim=1)

    # Compute loss: 2 - 2 * cosine_similarity
    # cosine_similarity = (p * z).sum(dim=1)
    loss = 2 - 2 * (p * z).sum(dim=1).mean()

    return loss


def symmetric_byol_loss(p1, z2, p2, z1):
    """
    Symmetric BYOL loss from both views

    Args:
        p1: prediction from online network on view 1
        z2: projection from target network on view 2
        p2: prediction from online network on view 2
        z1: projection from target network on view 1

    Returns:
        total loss (sum of both directions)
    """
    # Make sure target projections are detached (no gradient)
    z1 = z1.detach()
    z2 = z2.detach()

    # Compute loss in both directions
    loss1 = byol_loss_function(p1, z2)
    loss2 = byol_loss_function(p2, z1)

    return loss1 + loss2


class ProjectorPredictor(nn.Module):
    """
    Combined Projector + Predictor module for convenience
    Used in online network
    """
    def __init__(self,
                 encoder_dim=512,
                 projector_hidden=1024,
                 projector_out=256,
                 predictor_hidden=1024):
        super(ProjectorPredictor, self).__init__()

        self.projector = Projector(
            input_dim=encoder_dim,
            hidden_dim=projector_hidden,
            output_dim=projector_out
        )

        self.predictor = Predictor(
            input_dim=projector_out,
            hidden_dim=predictor_hidden,
            output_dim=projector_out
        )

    def forward(self, x, return_projection=False):
        """
        Args:
            x: (B, encoder_dim) encoder features
            return_projection: if True, return both projection and prediction

        Returns:
            if return_projection:
                projection, prediction
            else:
                prediction only
        """
        projection = self.projector(x)
        prediction = self.predictor(projection)

        if return_projection:
            return projection, prediction
        else:
            return prediction


def test_projector_predictor():
    """Test projector and predictor"""
    print("Testing Projector and Predictor...")

    # Create modules
    projector = Projector(input_dim=512, hidden_dim=1024, output_dim=256)
    predictor = Predictor(input_dim=256, hidden_dim=1024, output_dim=256)

    # Count parameters
    proj_params = sum(p.numel() for p in projector.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())

    print(f"Projector parameters: {proj_params:,}")
    print(f"Predictor parameters: {pred_params:,}")

    # Test forward pass
    batch_size = 4
    encoder_output = torch.randn(batch_size, 512)

    with torch.no_grad():
        # Online network path
        projection = projector(encoder_output)
        prediction = predictor(projection)

        print(f"\nEncoder output shape: {encoder_output.shape}")
        print(f"Projection shape: {projection.shape}")
        print(f"Prediction shape: {prediction.shape}")

        # Test loss computation
        # Simulate two views
        p1 = torch.randn(batch_size, 256)
        z2 = torch.randn(batch_size, 256)
        p2 = torch.randn(batch_size, 256)
        z1 = torch.randn(batch_size, 256)

        loss = symmetric_byol_loss(p1, z2, p2, z1)
        print(f"\nSymmetric BYOL loss: {loss.item():.4f}")

    # Test combined module
    print("\nTesting ProjectorPredictor...")
    proj_pred = ProjectorPredictor(
        encoder_dim=512,
        projector_hidden=1024,
        projector_out=256,
        predictor_hidden=1024
    )

    total_params = sum(p.numel() for p in proj_pred.parameters())
    print(f"Total parameters: {total_params:,}")

    with torch.no_grad():
        projection, prediction = proj_pred(encoder_output, return_projection=True)
        print(f"Projection shape: {projection.shape}")
        print(f"Prediction shape: {prediction.shape}")

    print("\nProjector and Predictor test passed!")


if __name__ == "__main__":
    test_projector_predictor()
