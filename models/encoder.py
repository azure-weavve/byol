"""
WaferEncoder: ResNet-18 based encoder for wafer map pattern learning

PyTorch 1.4.0 compatible implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RadialPositionalEncoder(nn.Module):
    """
    Encodes radial distance from wafer center
    Helps model understand center vs edge patterns
    """
    def __init__(self, wafer_size=(128, 128), embedding_dim=16):
        super(RadialPositionalEncoder, self).__init__()
        self.wafer_size = wafer_size
        self.embedding_dim = embedding_dim

        # Create radial distance map
        h, w = wafer_size
        center_h, center_w = h / 2.0, w / 2.0

        y_coords = torch.arange(h).float() - center_h
        x_coords = torch.arange(w).float() - center_w

        yy, xx = torch.meshgrid(y_coords, x_coords)
        radial_dist = torch.sqrt(xx ** 2 + yy ** 2)

        # Normalize to [0, 1]
        radial_dist = radial_dist / radial_dist.max()

        # Register as buffer (not trainable parameter)
        self.register_buffer('radial_dist', radial_dist.unsqueeze(0).unsqueeze(0))

        # Learnable embedding
        self.conv = nn.Conv2d(1, embedding_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            (B, C + embedding_dim, H, W) with radial encoding concatenated
        """
        B = x.size(0)
        radial_features = self.conv(self.radial_dist.expand(B, -1, -1, -1))
        return torch.cat([x, radial_features], dim=1)


class SelfAttention2D(nn.Module):
    """
    Self-attention for capturing global pattern relationships
    PyTorch 1.4.0 compatible version
    """
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with self-attention applied
        """
        B, C, H, W = x.size()

        # Project to Q, K, V
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')
        key = self.key_conv(x).view(B, -1, H * W)  # (B, C', H*W)
        value = self.value_conv(x).view(B, -1, H * W)  # (B, C, H*W)

        # Attention weights
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        # Residual connection with learnable gate
        out = self.gamma * out + x

        return out


class BasicBlock(nn.Module):
    """
    Basic ResNet block with 2 conv layers
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class WaferEncoder(nn.Module):
    """
    ResNet-18 based encoder for wafer maps

    Architecture:
        Input: (B, 1, 128, 128) - grayscale wafer map
        Conv stem: 1 → 64 channels
        ResNet blocks: 64 → 128 → 256 → 512
        Global Average Pooling
        Output: (B, 512) - feature vector

    Memory estimation (batch_size=256):
        Parameters: ~11M
        Forward pass: ~2GB VRAM
    """
    def __init__(self,
                 input_channels=1,
                 output_dim=512,
                 use_radial_encoding=True,
                 use_attention=True,
                 wafer_size=(128, 128),
                 layers=[2, 2, 2, 2]):  # ResNet-18 configuration
        super(WaferEncoder, self).__init__()

        self.use_radial_encoding = use_radial_encoding
        self.use_attention = use_attention
        self.output_dim = output_dim

        # Optional radial positional encoding
        if use_radial_encoding:
            self.radial_encoder = RadialPositionalEncoder(wafer_size, embedding_dim=16)
            input_channels += 16
        else:
            self.radial_encoder = None

        # Initial convolution stem
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        # Optional self-attention
        if use_attention:
            self.attention = SelfAttention2D(512, reduction=8)
        else:
            self.attention = None

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a ResNet layer with multiple blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: (B, 1, 128, 128) wafer map
        Returns:
            (B, 512) feature vector
        """
        # Optional radial encoding
        if self.radial_encoder is not None:
            x = self.radial_encoder(x)

        # Conv stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Optional self-attention
        if self.attention is not None:
            x = self.attention(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        Returns feature maps from each layer
        """
        features = {}

        if self.radial_encoder is not None:
            x = self.radial_encoder(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        features['stem'] = x

        x = self.maxpool(x)

        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        x = self.layer4(x)
        features['layer4'] = x

        if self.attention is not None:
            x = self.attention(x)
            features['attention'] = x

        return features


def test_encoder():
    """Test encoder with sample input"""
    print("Testing WaferEncoder...")

    # Create model
    encoder = WaferEncoder(
        input_channels=1,
        use_radial_encoding=True,
        use_attention=True,
        wafer_size=(128, 128)
    )

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)

    with torch.no_grad():
        output = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test feature maps
    features = encoder.get_feature_maps(x)
    print("\nFeature map shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print("\nWaferEncoder test passed!")


if __name__ == "__main__":
    test_encoder()
