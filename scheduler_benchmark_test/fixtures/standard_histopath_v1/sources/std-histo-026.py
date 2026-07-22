"""Generated standard scheduler benchmark model: std-histo-026."""
from __future__ import annotations

import os
import torch
from torch import nn

from localml_scheduler.elastic import ElasticTrainingSession
from scheduler_benchmark_test.standard.training_runtime import run_generated_job

epochs = int(os.environ.get("STANDARD_BENCH_EPOCHS", "50"))
batch_size = int(os.environ.get("STANDARD_BENCH_BATCH_SIZE", "32"))

def _activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
    }[name]


def _norm2d(name: str, channels: int) -> nn.Module:
    if name == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.BatchNorm2d(channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, groups: int = 1, activation: str, norm: str):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False),
            _norm2d(norm, out_ch),
            _activation(activation),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str, norm: str):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels, activation=activation, norm=norm),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _norm2d(norm, channels),
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class DenseLikeBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str, norm: str):
        super().__init__()
        growth = max(8, channels // 2)
        self.features = ConvNormAct(channels, growth, activation=activation, norm=norm)
        self.merge = nn.Conv2d(channels + growth, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat((x, self.features(x)), dim=1))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, *, activation: str, norm: str):
        super().__init__()
        del norm
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            _activation(activation),
            nn.Conv2d(channels * 4, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.norm(self.depthwise(x)))


class ConvClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str, norm: str, dropout: float):
        super().__init__()
        block_type: Callable[..., nn.Module]
        if kind == "vgg":
            block_type = lambda channels, **kwargs: ConvNormAct(channels, channels, **kwargs)
        elif kind == "resnet":
            block_type = ResidualBlock
        elif kind == "densenet":
            block_type = DenseLikeBlock
        elif kind == "convnext":
            block_type = ConvNeXtBlock
        else:
            raise ValueError(f"Unsupported conventional CNN architecture: {kind}")
        layers: list[nn.Module] = [ConvNormAct(3, width, stride=2, activation=activation, norm=norm)]
        channels = width
        for stage in range(3):
            for _ in range(depth):
                layers.append(block_type(channels, activation=activation, norm=norm))
            if stage < 2:
                layers.append(ConvNormAct(channels, channels * 2, stride=2, activation=activation, norm=norm))
                channels *= 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class InvertedBlock(nn.Module):
    def __init__(self, channels: int, *, expansion: int, activation: str, norm: str, shuffle: bool = False):
        super().__init__()
        hidden = channels * expansion
        self.shuffle = shuffle
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            _norm2d(norm, hidden),
            _activation(activation),
            ConvNormAct(hidden, hidden, groups=hidden, activation=activation, norm=norm),
            nn.Conv2d(hidden, channels, 1, bias=False),
            _norm2d(norm, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.body(x)
        if self.shuffle and out.shape[1] % 2 == 0:
            b, c, h, w = out.shape
            out = out.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)
        return out


class EfficientClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, activation: str, norm: str, dropout: float):
        super().__init__()
        expansion = {"mobilenet_v2": 4, "mobilenet_v3": 3, "mbconv": 6, "shufflenet": 2}[kind]
        layers: list[nn.Module] = [ConvNormAct(3, width, stride=2, activation=activation, norm=norm)]
        channels = width
        for stage in range(3):
            for _ in range(depth + (1 if kind == "mbconv" else 0)):
                layers.append(
                    InvertedBlock(
                        channels,
                        expansion=expansion,
                        activation=activation,
                        norm=norm,
                        shuffle=kind == "shufflenet",
                    )
                )
            if stage < 2:
                layers.append(ConvNormAct(channels, channels * 2, stride=2, activation=activation, norm=norm))
                channels *= 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, dim: int, *, kind: str, dropout: float):
        super().__init__()
        token_hidden = max(32, tokens // 2)
        channel_hidden = dim * (3 if kind == "gmlp" else 2)
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, token_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(token_hidden, tokens)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(channel_hidden, dim)
        )
        self.residual_scale = nn.Parameter(torch.ones(dim)) if kind == "resmlp" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        update = self.channel_mlp(self.norm2(x))
        return x + (update * self.residual_scale if self.residual_scale is not None else update)


class MixerClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, dropout: float, **_: object):
        super().__init__()
        patch = 8
        tokens = (96 // patch) ** 2
        self.patch_embed = nn.Conv2d(3, width, patch, stride=patch)
        self.blocks = nn.Sequential(*(MixerBlock(tokens, width, kind=kind, dropout=dropout) for _ in range(depth + 1)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.head(self.norm(self.blocks(x)).mean(dim=1))


class RecurrentClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, dropout: float, **_: object):
        super().__init__()
        self.kind = kind
        self.patch = nn.Conv2d(3, width, 6, stride=6) if kind in {"patch_gru", "convlstm"} else None
        input_dim = width if self.patch is not None else 96 * 3
        recurrent_cls = nn.GRU if kind == "patch_gru" else nn.LSTM
        bidirectional = kind == "bilstm"
        self.recurrent = recurrent_cls(
            input_dim,
            width,
            num_layers=max(1, depth),
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * (2 if bidirectional else 1), 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch is None:
            sequence = x.permute(0, 2, 1, 3).flatten(2)
        else:
            embedded = self.patch(x)
            sequence = embedded.flatten(2).transpose(1, 2)
        output, _state = self.recurrent(sequence)
        return self.head(output[:, -1])


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TransformerClassifier(nn.Module):
    def __init__(self, *, kind: str, width: int, depth: int, dropout: float, **_: object):
        super().__init__()
        heads = 4 if width % 4 == 0 else 2
        patch = 8 if kind != "window_transformer" else 6
        self.stem: nn.Module
        if kind == "conv_vit":
            self.stem = nn.Sequential(nn.Conv2d(3, width // 2, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(width // 2, width, 4, stride=4))
        else:
            self.stem = nn.Conv2d(3, width, patch, stride=patch)
        tokens_per_side = 12 if patch == 8 or kind == "conv_vit" else 16
        token_count = tokens_per_side**2
        self.position = nn.Parameter(torch.zeros(1, token_count, width))
        self.blocks = nn.Sequential(*(AttentionBlock(width, heads, dropout=dropout) for _ in range(depth + 1)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x).flatten(2).transpose(1, 2)
        x = x + self.position[:, : x.shape[1]]
        return self.head(self.norm(self.blocks(x)).mean(dim=1))


def build_architecture(
    *,
    family: str,
    architecture: str,
    width: int,
    depth: int,
    activation: str,
    norm: str,
    dropout: float,
) -> nn.Module:
    if family == "cnn":
        return ConvClassifier(
            kind=architecture,
            width=int(width),
            depth=int(depth),
            activation=activation,
            norm=norm,
            dropout=float(dropout),
        )
    if family == "efficient_cnn":
        return EfficientClassifier(
            kind=architecture,
            width=int(width),
            depth=int(depth),
            activation=activation,
            norm=norm,
            dropout=float(dropout),
        )
    if family == "mlp_mixer":
        return MixerClassifier(kind=architecture, width=int(width), depth=int(depth), dropout=float(dropout))
    if family == "recurrent":
        return RecurrentClassifier(kind=architecture, width=int(width), depth=int(depth), dropout=float(dropout))
    if family == "vision_transformer":
        return TransformerClassifier(kind=architecture, width=int(width), depth=int(depth), dropout=float(dropout))
    raise ValueError(f"Unsupported benchmark model family: {family}")

SPEC = {"activation": "relu", "architecture": "mobilenet_v3", "dataset_size": 174464, "depth": 2, "dropout": 0.0, "epochs": 50, "family": "efficient_cnn", "input_shape": [3, 96, 96], "job_id": "std-histo-026", "learning_rate": 0.001, "norm": "batch", "precision": "tf32", "profile_bucket": "standard-v1:efficient_cnn:mobilenet_v3:tf32", "seed": 42026, "submitted_batch_size": 32, "variant": "identity_relu_batch", "variant_index": 0, "width": 32}


class Model(nn.Module):
    """Job-local architecture variant selected by the generated specification."""

    def __init__(self):
        super().__init__()
        self.network = build_architecture(
            family=SPEC["family"],
            architecture=SPEC["architecture"],
            width=SPEC["width"],
            depth=SPEC["depth"],
            activation=SPEC["activation"],
            norm=SPEC["norm"],
            dropout=SPEC["dropout"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> nn.Module:
    return Model()


def build_loader(session, dataset):
    return session.make_dataloader(
        dataset,
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


def register_training_state(session, model, optimizer, scaler):
    session.register_training_state(model, optimizer, scaler=scaler)


def restore_training_state(session):
    return session.restore_if_present()


def optimizer_step_completed(session, samples, epoch, batch_index, global_step, metrics):
    session.optimizer_step_completed(samples, epoch, batch_index, global_step, metrics=metrics)


if __name__ == "__main__":
    session = ElasticTrainingSession.from_env()
    run_generated_job(
        spec=SPEC,
        build_model=build_model,
        build_loader=build_loader,
        register_training_state=register_training_state,
        restore_training_state=restore_training_state,
        optimizer_step_completed=optimizer_step_completed,
        session=session,
        epochs=epochs,
        batch_size=batch_size,
    )
