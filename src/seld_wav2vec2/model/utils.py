
import torch.nn as nn


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
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))
