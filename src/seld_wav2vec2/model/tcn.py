from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from fairseq.dataclass import FairseqDataclass
from fairseq.models.wav2vec.wav2vec2 import EXTRACTOR_MODE_CHOICES
from fairseq.modules import Fp32LayerNorm, TransposeLast
from torch.nn.utils import weight_norm

from seld_wav2vec2.model import conv
from seld_wav2vec2.model.utils import get_activation_fn


@dataclass
class TcnConfig(FairseqDataclass):
    num_inputs: int = field(
        default=768, metadata={"help": "input size of TCN"}
    )
    num_channels: List[int] = field(
        default=(768, 768), metadata={"help": "inner dimensions of TCN"}
    )
    kernel_size: int = field(
        default=5, metadata={"help": "kernel-size used in TCN"}
    )
    stride_size: int = field(default=0, metadata={"help": "stride used in TCN"})
    causal: bool = field(default=False, metadata={"help": "causal or non-causal TCN"})
    dropout: float = field(
        default=0.0, metadata={"help": "dropout of TCN module"}
    )
    mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={"help": "norm type used in TCN block"},
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of TCN"},
    )
    btc: bool = field(default=False, metadata={"help": "BTC order"})


@dataclass
class MultiScaleTcnConfig(FairseqDataclass):
    num_inputs: int = field(default=768, metadata={"help": "input size of MSTCN"})
    num_channels: List[int] = field(
        default=(768, 768), metadata={"help": "inner dimensions of TCN"}
    )
    kernels: List[int] = field(
        default=(10, 20, 40), metadata={"help": "kernels of MSTCN"}
    )
    causal: bool = field(default=False, metadata={"help": "causal or non-causal MSTCN"})
    skip: bool = field(default=True, metadata={"help": "skip connection MSTCN"})
    dropout: float = field(default=0.0, metadata={"help": "dropout of MSTCN module"})
    mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={"help": "norm type used in TCN block"},
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of TCN"},
    )


class Wav2vec2AudioFrameClassMultiScaleTcnHead(nn.Module):
    """
    Head for audioframe classification tasks with MultiScaleTCN.

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        dropout_input,
        num_outs,
        out_activation_fn,
        cfg: MultiScaleTcnConfig,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)

        self.ms_tcn = MultiScaleTemporalConv(
            num_inputs=input_dim,
            num_channels=cfg.num_channels,
            kernels=cfg.kernels,
            dropout=cfg.dropout,
            mode=cfg.mode,
            activation_fn=cfg.activation_fn,
            causal=cfg.causal,
            skip=cfg.skip,
        )
        self.out_proj = nn.Linear(len(cfg.kernels) * cfg.num_channels[-1], num_outs)
        self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        x = self.dropout_input(features)
        x = self.ms_tcn(x)
        x = self.out_proj(x)
        x = self.out_activation_fn(x)
        return x


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernels: List[int],
        dropout: float = 0.0,
        mode: str = "default",
        activation_fn: str = "gelu",
        causal: bool = False,
        skip: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        self.conv_layers = nn.ModuleList()
        for k in kernels:
            cfg = {
                "num_inputs": num_inputs,
                "num_channels": num_channels,
                "kernel_size": k,
                "stride_size": 0,
                "dropout": dropout,
                "mode": mode,
                "causal": causal,
                "btc": True,
                "activation_fn": activation_fn,
            }

            self.conv_layers.append(TemporalConvNet(cfg))

        self.skip = skip
        if self.skip:
            if num_inputs != len(kernels) * num_channels[-1]:
                self.skip_proj = nn.Conv1d(
                    in_channels=num_inputs,
                    out_channels=len(kernels) * num_channels[-1],
                    kernel_size=1,
                    stride=1,
                    padding="same",
                )
            else:
                self.skip_proj = None

        self.act = get_activation_fn(activation_fn)

    def forward(self, x):
        r = []
        for cnn in self.conv_layers:
            r.append(cnn(x))
        r = torch.cat(r, dim=-1)

        if self.skip:
            if self.skip_proj:
                x = self.skip_proj(x.transpose(1, 2)).transpose(1, 2)
            r = r + x
        return self.act(r)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        dropout=0.2,
        mode="default",
        activation_fn="gelu",
        causal=False,
    ):
        super().__init__()

        conv_type = "CausalConv1d" if causal else "TemporalConv1d"

        if mode == "layer_norm":
            conv1 = nn.Sequential(
                getattr(conv, conv_type)(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                TransposeLast(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            conv1 = weight_norm(
                getattr(conv, conv_type)(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
        act1 = get_activation_fn(activation_fn)
        dropout1 = nn.Dropout(dropout)

        if mode == "layer_norm":
            conv2 = nn.Sequential(
                getattr(conv, conv_type)(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                TransposeLast(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            conv2 = weight_norm(
                getattr(conv, conv_type)(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
        act2 = get_activation_fn(activation_fn)
        dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(conv1, act1, dropout1, conv2, act2, dropout2)
        self.skip_proj = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.act = get_activation_fn(activation_fn)

    def forward(self, x):
        # x -> (B, C, T)
        out = self.net(x)
        res = x if self.skip_proj is None else self.skip_proj(x)
        return self.act(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, cfg: TcnConfig):
        super().__init__()

        self.btc = cfg["btc"]
        num_channels = cfg["num_channels"]
        kernel_size = cfg["kernel_size"]
        stride_size = cfg["stride_size"]
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = cfg["num_inputs"] if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=cfg["dropout"],
                    causal=cfg["causal"],
                    mode=cfg["mode"],
                    activation_fn=cfg["activation_fn"],
                )
            ]

        self.encoder = nn.Sequential(*layers)
        if stride_size > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    out_channels, out_channels, kernel_size=kernel_size, stride=stride_size
                ),
                TransposeLast(),
                Fp32LayerNorm(out_channels, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            self.downsample = None

    def forward(self, x, padding_mask=None):
        if self.btc:
            x = x.transpose(1, 2)
        r = self.encoder(x)
        if self.downsample is not None:
            r = self.downsample(r)
        if self.btc:
            r = r.transpose(1, 2)
        return r


class TemporalResnetBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        dropout=0.2,
        mode="default",
        activation_fn="gelu",
        bias=True
    ):
        super().__init__()

        if mode == "layer_norm":
            conv1 = nn.Sequential(
                conv.TemporalConv1d(
                    n_inputs, n_outputs, kernel_size, stride=1, dilation=1, bias=bias
                ),
                TransposeLast(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            conv1 = weight_norm(
                conv.TemporalConv1d(
                    n_inputs, n_outputs, kernel_size, stride=1, dilation=1, bias=bias
                )
            )
        act1 = get_activation_fn(activation_fn)
        dropout1 = nn.Dropout(dropout)

        if mode == "layer_norm":
            conv2 = nn.Sequential(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                TransposeLast(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            conv2 = weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
        act2 = get_activation_fn(activation_fn)
        dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(conv1, act1, dropout1, conv2, act2, dropout2)

        if mode == "layer_norm":
            self.skip_proj = nn.Sequential(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                TransposeLast(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                TransposeLast(),
            )
        else:
            self.skip_proj = weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
        self.act = get_activation_fn(activation_fn)

    def forward(self, x):
        # x -> (B, C, T)
        out = self.net(x)
        res = self.skip_proj(x)
        return self.act(out + res)
