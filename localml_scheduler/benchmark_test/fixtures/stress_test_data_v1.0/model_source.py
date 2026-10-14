"""Stress Test Data v1.0 models restricted to the student operation vocabulary."""

from __future__ import annotations

import torch
from torch import nn


CONVOLUTIONAL_ARCHITECTURES = {
    "vgg": "plain",
    "resnet": "residual",
    "densenet": "dense",
    "convnext_compatible": "depthwise",
    "mobilenet_v2": "depthwise",
    "mobilenet_v3": "depthwise",
    "mbconv": "depthwise",
    "efficient_residual": "residual",
}
MLP_ARCHITECTURES = {
    "patch_mlp",
    "mixer_mlp",
    "gmlp_compatible",
    "resmlp_compatible",
}
RECURRENT_ARCHITECTURES = {
    "row_lstm",
    "bilstm",
    "patch_gru",
    "conv_lstm",
}
HYBRID_ARCHITECTURES = {
    "conv_mlp": "plain",
    "depthwise_mlp": "depthwise",
    "residual_mlp": "residual",
    "dense_mlp": "dense",
}


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Stress Test Data v1.0 activation: {name}") from exc


class ConvAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            _activation(activation),
        )


class PlainBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str) -> None:
        super().__init__()
        self.body = ConvAct(channels, channels, activation=activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, activation=activation),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = _activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class DenseBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str) -> None:
        super().__init__()
        growth = max(4, channels // 2)
        self.features = ConvAct(channels, growth, activation=activation)
        self.merge = ConvAct(channels + growth, channels, activation=activation, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat((inputs, self.features(inputs)), dim=1))


class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str) -> None:
        super().__init__()
        hidden = channels * 2
        self.body = nn.Sequential(
            ConvAct(channels, hidden, activation=activation, kernel_size=1),
            ConvAct(hidden, hidden, activation=activation, groups=hidden),
            nn.Conv2d(hidden, channels, 1),
        )
        self.activation = _activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


def _block(kind: str, channels: int, activation: str) -> nn.Module:
    block_types = {
        "plain": PlainBlock,
        "residual": ResidualBlock,
        "dense": DenseBlock,
        "depthwise": DepthwiseBlock,
    }
    try:
        return block_types[kind](channels, activation=activation)
    except KeyError as exc:
        raise ValueError(f"Unsupported Stress Test Data v1.0 block: {kind}") from exc


class ConvClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvAct(3, width, activation=activation)]
        layers.extend(_block(kind, width, activation) for _ in range(depth))
        layers.extend(
            [
                nn.MaxPool2d(2),
                ConvAct(width, width * 2, activation=activation),
            ]
        )
        layers.extend(_block(kind, width * 2, activation) for _ in range(depth))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, *, activation: str, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(
            nn.Linear(width, width * expansion),
            _activation(activation),
            nn.Linear(width * expansion, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.body(self.norm(inputs))


class ImageMLPClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str) -> None:
        super().__init__()
        expansion = {
            "patch_mlp": 2,
            "mixer_mlp": 3,
            "gmlp_compatible": 4,
            "resmlp_compatible": 2,
        }[kind]
        self.patch_features = nn.Sequential(
            ConvAct(3, width, activation=activation, kernel_size=4, stride=4),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(width, activation=activation, expansion=expansion) for _ in range(depth))
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.patch_features(inputs)))


class RecurrentClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvAct(3, width, activation=activation, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        if kind == "patch_gru":
            self.recurrent: nn.Module = nn.GRU(
                width,
                width,
                num_layers=max(1, depth),
                batch_first=True,
            )
            output_width = width
        else:
            bidirectional = kind == "bilstm"
            self.recurrent = nn.LSTM(
                width,
                width,
                num_layers=max(1, depth),
                batch_first=True,
                bidirectional=bidirectional,
            )
            output_width = width * (2 if bidirectional else 1)
        self.head = nn.Linear(output_width, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.features(inputs).unsqueeze(1)
        output, _state = self.recurrent(sequence)
        return self.head(output[:, -1])


class HybridClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvAct(3, width, activation=activation),
            *(_block(kind, width, activation) for _ in range(depth)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.LayerNorm(width * 2),
            _activation(activation),
            nn.Linear(width * 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


def build_model(
    architecture: str,
    width: int,
    depth: int,
    activation: str,
) -> nn.Module:
    """Build one Stress Test Data v1.0 model using current student operations."""

    if architecture in CONVOLUTIONAL_ARCHITECTURES:
        return ConvClassifier(
            kind=CONVOLUTIONAL_ARCHITECTURES[architecture],
            width=int(width),
            depth=int(depth),
            activation=activation,
        )
    if architecture in MLP_ARCHITECTURES:
        return ImageMLPClassifier(
            kind=architecture,
            width=int(width),
            depth=int(depth),
            activation=activation,
        )
    if architecture in RECURRENT_ARCHITECTURES:
        return RecurrentClassifier(
            kind=architecture,
            width=int(width),
            depth=int(depth),
            activation=activation,
        )
    if architecture in HYBRID_ARCHITECTURES:
        return HybridClassifier(
            kind=HYBRID_ARCHITECTURES[architecture],
            width=int(width),
            depth=int(depth),
            activation=activation,
        )
    raise ValueError(f"Unsupported Stress Test Data v1.0 architecture: {architecture}")
