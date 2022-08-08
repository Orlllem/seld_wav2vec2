import logging
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import register_model
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from fairseq.modules import Fp32GroupNorm, Fp32LayerNorm, TransposeLast
from omegaconf import II, MISSING, open_dict

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2ChConfig(Wav2Vec2Config):
    in_channels: int = field(
        default=4, metadata={"help": "number of input channels - CNN"}
    )
    in_conv_groups: int = field(
        default=1, metadata={"help": "number of conv_group channels - CNN"}
    )


@register_model("wav2vec2_ch", dataclass=Wav2Vec2ChConfig)
class Wav2Vec2ChModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2ChConfig):
        super().__init__(cfg)

        feature_enc_layers = eval(cfg.conv_feature_layers)

        self.feature_extractor = ConvFeatureExtractionChModel(
            in_channels=cfg.in_channels,
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            conv_groups=cfg.in_conv_groups,
        )


@dataclass
class Wav2Vec2ChPretConfig(Wav2Vec2ChConfig):
    pre_w2v_path: str = field(
        default=MISSING, metadata={"help": "path to pretrained data2vec model"}
    )
    normalize: bool = II("task.normalize")
    ignore_mismatched_sizes: bool = field(
        default=False, metadata={"help": "whether to ignore mismatched sizes"}
    )
    data: str = II("task.data")
    # this holds the loaded the pretrained data2vec args
    pre_w2v_args: Any = None
    ddp_backend: str = II("distributed_training.ddp_backend")


@register_model("wav2vec2_ch_pretr", dataclass=Wav2Vec2ChPretConfig)
class Wav2Vec2ChModelPtrained(Wav2Vec2ChModel):
    def __init__(self, cfg: Wav2Vec2ChPretConfig):
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2ChPretConfig, task=None):

        arg_overrides = {
            "conv_feature_layers": cfg.conv_feature_layers,
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.encoder_layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
        }

        if cfg.pre_w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.pre_w2v_path,
                                                            arg_overrides)
            pre_w2v_args = state.get("cfg", None)
            if pre_w2v_args is None:
                pre_w2v_args = convert_namespace_to_omegaconf(state["args"])
            pre_w2v_args.criterion = None
            pre_w2v_args.lr_scheduler = None
            cfg.pre_w2v_args = pre_w2v_args

            logger.info(pre_w2v_args)

        else:
            state = None
            pre_w2v_args = cfg.pre_w2v_args
            if isinstance(pre_w2v_args, Namespace):
                cfg.pre_w2v_args = pre_w2v_args = convert_namespace_to_omegaconf(
                    pre_w2v_args)

        model_normalized = pre_w2v_args.task.get(
            "normalize", pre_w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both"
            "pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(pre_w2v_args):
                pre_w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        pre_w2v_args.task.data = cfg.data

        model = super().build_model(cfg, task)

        if state is not None:
            model = cls.load_model_weights(state, model, cfg)

        return model

    @staticmethod
    def load_model_weights(state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the
                    # weights one by one
                    # We dont load all weights together as that wont be memory
                    # efficient and may cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile(r"encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]

            if cfg.ignore_mismatched_sizes:
                state_dict = model.state_dict()
                state_model = state["model"].copy()
                for key in state["model"]:
                    if key in state_dict.keys():
                        if state["model"][key].shape != state_dict[key].shape:
                            state_model.pop(key)
                            logger.info("key {} is not matching".format(key))
            else:
                state_model = state["model"]

            model.load_state_dict(state_model, strict=False)

        return model


class ConvFeatureExtractionChModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_groups: int = 1,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            conv_groups=1,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias,
                                 groups=conv_groups)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout),
                                     nn.GELU())

        in_d = in_channels
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    conv_groups=conv_groups if i == 0 else 1,
                )
            )
            in_d = dim

    def forward(self, x):

        for conv in self.conv_layers:
            x = conv(x)

        return x


class ConvPool1DFeatureExtractionChModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_groups: int = 1,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            t_pool,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            conv_groups=1,
            padding=1
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias,
                                 groups=conv_groups, padding=padding)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.MaxPool1d(t_pool),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.MaxPool1d(t_pool),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(),
                                     nn.Dropout(p=dropout),
                                     nn.MaxPool1d(t_pool),
                                     nn.GELU())

        in_d = in_channels
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, t, f) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k=3,
                    stride=1,
                    padding=1,
                    t_pool=t,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    conv_groups=conv_groups if i == 0 else 1,
                )
            )
            in_d = dim

    def forward(self, x):

        for conv in self.conv_layers:
            x = conv(x)

        return x


class ConvPool2DFeatureExtractionChModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_groups: int = 1,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            t_pool,
            f_pool,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            conv_groups=1,
            padding=1,
        ):
            def make_conv():
                conv = nn.Conv2d(n_in, n_out, k, stride=stride, bias=conv_bias,
                                 groups=conv_groups, padding=padding)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.MaxPool2d((t_pool, f_pool)),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.MaxPool2d((t_pool, f_pool)),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(),
                                     nn.Dropout(p=dropout),
                                     nn.MaxPool2d((t_pool, f_pool)),
                                     nn.GELU())

        in_d = in_channels
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, t, f) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k=3,
                    stride=1,
                    t_pool=t,
                    f_pool=f,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    conv_groups=conv_groups if i == 0 else 1,
                    padding=1,
                )
            )
            in_d = dim

    def forward(self, x):

        for conv in self.conv_layers:
            x = conv(x)

        # (B, C, T, D) -> (B, C, D, T)
        x = x.transpose(2, 3)

        # reshape (B, C, D, T) -> (B, C*D, T)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

        return x
