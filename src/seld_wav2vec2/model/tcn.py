from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import Fp32LayerNorm, TransposeLast
from pytorch_tcn import TCN
from torch.nn.utils import weight_norm

from seld_wav2vec2.model import conv
from seld_wav2vec2.model.utils import Transpose2DLast, get_activation_fn


@dataclass
class TcnConfig(FairseqDataclass):
    num_inputs: int = field(default=768, metadata={"help": "input size of TCN"})
    num_channels: List[int] = field(
        default=(768, 768), metadata={"help": "inner dimensions of TCN"}
    )
    kernel_size: int = field(default=5, metadata={"help": "kernel-size used in TCN"})
    causal: bool = field(default=False, metadata={"help": "causal or non-causal TCN"})
    dropout: float = field(default=0.0, metadata={"help": "dropout of TCN module"})
    use_norm: str = field(
        default="weight_norm",
        metadata={"help": "norm type used in TCN block"},
    )
    activation: str = field(
        default="gelu",
        metadata={"help": " activation function of TCN"},
    )
    input_shape: str = field(
        default="NLC",
        metadata={"help": "shape of CNN"},
    )
    use_skip_connections: bool = field(
        default=True, metadata={"help": "skip connection"}
    )


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
    use_norm: str = field(
        default="weight_norm",
        metadata={"help": "norm type used in TCN block"},
    )
    activation: str = field(
        default="gelu",
        metadata={"help": " activation function of TCN"},
    )
    input_shape: str = field(
        default="NLC",
        metadata={"help": "shape of CNN"},
    )
    use_skip_connections: bool = field(
        default=True, metadata={"help": "skip connection"}
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
        bidirectional,
        merge_cat,
        num_outs,
        out_activation_fn,
        cfg: MultiScaleTcnConfig,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)
        self.bidirectional = bidirectional
        self.merge_cat = merge_cat

        embed_dim = len(cfg.kernels) * cfg.num_channels[-1]

        if self.bidirectional and self.merge_cat:
            embed_dim = 2 * embed_dim

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
        if self.bidirectional:
            self.ms_tcn_r = MultiScaleTemporalConv(
                num_inputs=input_dim,
                num_channels=cfg.num_channels,
                kernels=cfg.kernels,
                dropout=cfg.dropout,
                mode=cfg.mode,
                activation_fn=cfg.activation_fn,
                causal=cfg.causal,
                skip=cfg.skip,
            )

        self.out_proj = nn.Linear(embed_dim, num_outs)
        self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        x = self.dropout_input(features)
        r = self.ms_tcn(x)

        if self.bidirectional:
            x_rl = torch.flip(x, [1])  # reverse T dimension
            r_rl = self.ms_tcn_r(x_rl)
            r_rl = torch.flip(r_rl, [1])  # return T again
            if self.merge_cat:
                r = torch.cat((r, r_rl), dim=-1)
            else:
                r = r + r_rl

        r = self.out_proj(r)
        r = self.out_activation_fn(r)
        return r


class Wav2vec2AudioFrameClassMultiScaleTcnCatHead(nn.Module):
    """
    Head for audioframe classification tasks with MultiScaleTCN.

    It produces outputs of size (B, T, N)

    """

    def __init__(self, input_dim, cfg, num_outs):
        super().__init__()

        self.cat_heads = nn.ModuleList()

        for d in range(cfg.doa_size):
            self.cat_heads.append(
                Wav2vec2AudioFrameClassMultiScaleTcnHead(
                    input_dim=input_dim,
                    dropout_input=cfg.regression_input_dropout,
                    bidirectional=cfg.bidirectional,
                    merge_cat=cfg.merge_cat,
                    num_outs=num_outs * cfg.n_bins,
                    out_activation_fn=cfg.regression_out_activation_fn,
                    cfg=cfg.regression,
                )
            )

    def forward(self, features, padding_mask=None):
        preds = []
        for head in self.cat_heads:
            preds.append(head(features))
        return torch.cat(preds, dim=-1)


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernels: List[int],
        dropout: float = 0.0,
        use_norm: str = "weight_norm",
        activation: str = "gelu",
        causal: bool = False,
        skip: bool = False,
        input_shape: str = "NLC",
        use_skip_connections: bool = False,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        for k in kernels:
            cfg = {
                "num_inputs": num_inputs,
                "num_channels": num_channels,
                "kernel_size": k,
                "dropout": dropout,
                "use_norm": use_norm,
                "causal": causal,
                "input_shape": input_shape,
                "activation": activation,
                "use_skip_connections": use_skip_connections,
            }

            self.conv_layers.append(TCN(cfg))

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

        self.act = get_activation_fn(activation)

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
        bias=True,
        conv1d=True,
    ):
        super().__init__()

        if conv1d is True:
            conv_layer = "Conv1d"
            transpose_layer = TransposeLast
        else:
            conv_layer = "Conv2d"
            transpose_layer = Transpose2DLast

        if mode == "layer_norm":
            conv1 = nn.Sequential(
                getattr(conv, f"Temporal{conv_layer}")(
                    n_inputs, n_outputs, kernel_size, stride=1, dilation=1, bias=bias
                ),
                transpose_layer(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                transpose_layer(),
            )
        else:
            conv1 = weight_norm(
                getattr(conv, f"Temporal{conv_layer}")(
                    n_inputs, n_outputs, kernel_size, stride=1, dilation=1, bias=bias
                )
            )
        act1 = get_activation_fn(activation_fn)
        dropout1 = nn.Dropout(dropout)

        if mode == "layer_norm":
            conv2 = nn.Sequential(
                getattr(nn, f"{conv_layer}")(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                transpose_layer(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                transpose_layer(),
            )
        else:
            conv2 = weight_norm(
                getattr(nn, f"{conv_layer}")(
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
                getattr(nn, f"{conv_layer}")(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
                transpose_layer(),
                Fp32LayerNorm(n_outputs, elementwise_affine=True),
                transpose_layer(),
            )
        else:
            self.skip_proj = weight_norm(
                getattr(nn, f"{conv_layer}")(
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
