import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import Fp32LayerNorm

from seld_wav2vec2.model.utils import Fp32BatchNorm1d, get_activation_fn


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class TemporalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(TemporalConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        return

    def forward(self, x, inference=None):
        # Implementation of 'same'-type padding (non-causal padding)

        # Check if pad_len is an odd value
        # If so, pad the input one more on the right side
        if self.__padding % 2 != 0:
            x = F.pad(x, [0, 1])

        x = super(TemporalConv1d, self).forward(x)

        return x


class TemporalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super(TemporalConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        return


class Inception(nn.Module):
    def __init__(self, input_size, filters, activation_fn, dropout, batch_norm=True):
        super(Inception, self).__init__()

        self.bottleneck1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv10 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv20 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv40 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding="same",
            bias=False,
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bottleneck2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)

        self.act = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.bottleneck2(self.max_pool(x))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = self.act(y)
        y = self.dropout(y)
        if self.batch_norm:
            y = self.batch_norm(y)
        return y


class ResidualInception(nn.Module):
    def __init__(self, input_size, filters, activation_fn, dropout, batch_norm=True):
        super(ResidualInception, self).__init__()

        self.bottleneck = nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv_layers = nn.ModuleList()
        for d in range(3):
            self.conv_layers.append(
                Inception(
                    input_size=input_size if d == 0 else 4 * filters,
                    filters=filters,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)
        self.act = get_activation_fn(activation_fn)

    def forward(self, x):
        y = x.clone()
        for conv in self.conv_layers:
            y = conv(y)

        if self.batch_norm:
            y = y + self.batch_norm(self.bottleneck(x))
        else:
            y = y + self.bottleneck(x)
        y = self.act(y)
        return y


class InceptionTimeModel(nn.Module):
    def __init__(
        self, input_size, filters, depth, activation_fn, dropout, batch_norm=True
    ):
        super(InceptionTimeModel, self).__init__()

        assert depth % 3 == 0, f"depth: {depth} must be divisible by 3"

        self.inception_residuals = nn.ModuleList()
        for d in range(int(depth / 3)):
            self.inception_residuals.append(
                ResidualInception(
                    input_size if d == 0 else 4 * filters,
                    filters,
                    activation_fn,
                    dropout,
                    batch_norm=batch_norm,
                )
            )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T

        for block in self.inception_residuals:
            x = block(x)
        x = x.transpose(1, 2)  # B, T, C
        return x


class Conv1dCeil(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        bias=True,
        norm_type=None,
        activation_fn=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )  # padding handled manually
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.norm_type = norm_type

        if self.norm_type == "batch_norm":
            self.norm = Fp32BatchNorm1d(in_channels)
        elif self.norm_type == "layer_norm":
            self.norm = Fp32LayerNorm(in_channels)
        else:
            self.norm = None

        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        L_in = x.shape[-1]
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1

        # Calculate output length with ceil
        L_out = (
            math.ceil((L_in + 2 * self.padding - effective_kernel) / self.stride) + 1
        )

        # Calculate total padding needed
        needed = max(
            0, (L_out - 1) * self.stride + effective_kernel - L_in - 2 * self.padding
        )

        # Apply extra padding to the right
        pad_left = self.padding
        pad_right = self.padding + needed

        x = F.pad(x, (pad_left, pad_right))

        x = self.conv(x)
        if self.norm_type == "layer_norm":
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation_fn(x)
        return x
