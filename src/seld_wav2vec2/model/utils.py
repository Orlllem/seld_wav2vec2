
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return nn.SiLU()
    elif activation == "selu":
        return nn.SELU()
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))


def SiLU_inplace_to_False(module):
    for _, layer in module.named_modules():
        if isinstance(layer, nn.SiLU):
            layer.inplace = False


class Transpose2DLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-3, -1)


class Fp32BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        output = F.batch_norm(
            input.float(),
            self.running_mean.float()
            if self.running_mean is not None
            else None
            if not self.training or self.track_running_stats
            else None,
            self.running_var.float()
            if not self.training or self.track_running_stats
            else None,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return output.type_as(input)


class Fp32BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        output = F.batch_norm(
            input.float(),
            self.running_mean.float()
            if self.running_mean is not None
            else None
            if not self.training or self.track_running_stats
            else None,
            self.running_var.float()
            if not self.training or self.track_running_stats
            else None,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return output.type_as(input)


class Fp32InstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, *args, **kwargs):
        self.transpose_last = "transpose_last" in kwargs and kwargs["transpose_last"]
        if "transpose_last" in kwargs:
            del kwargs["transpose_last"]
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.transpose_last:
            input = input.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        output = F.instance_norm(
            input.float(),
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            use_input_stats=self.training or not self.track_running_stats,
            momentum=self.momentum,
            eps=self.eps,
        )
        if self.transpose_last:
            output = output.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return output.type_as(input)
