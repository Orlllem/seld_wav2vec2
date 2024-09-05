import contextlib
import logging
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2AsrConfig,
    Wav2VecCtc,
    Wav2VecEncoder,
)
from fairseq.modules import Fp32LayerNorm
from fairseq.tasks import FairseqTask
from omegaconf import II, OmegaConf, open_dict
from pytorch_tcn import TCN
from torch import Tensor

from seld_wav2vec2.model.conv import Conv1dCeil, InceptionTimeModel
from seld_wav2vec2.model.tcn import (
    MultiScaleTcnConfig,
    TcnConfig,
    Wav2vec2AudioFrameClassMultiScaleTcnCatHead,
    Wav2vec2AudioFrameClassMultiScaleTcnHead,
)
from seld_wav2vec2.model.utils import (
    Fp32BatchNorm1d,
    SiLU_inplace_to_False,
    get_activation_fn,
)
from seld_wav2vec2.model.wav2vec2_multi_ch import (
    ConformerEncoderHeader,
    TransformerEncoderHeader,
    Wav2Vec2ChConfig,
    Wav2Vec2HeaderConfig,
)

logger = logging.getLogger(__name__)


RNN_CHOICES = ChoiceEnum(["GRU", "LSTM"])
NORM_CHOICES = ChoiceEnum(["layer_norm", "batch_norm", "default"])
ATT_CHOICES = ChoiceEnum(["add-attention", "local-aware-attention", "self-attention"])


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


@dataclass
class RNNConfig(FairseqDataclass):
    layer_type: RNN_CHOICES = field(default="GRU", metadata={"help": "rnn hidden type"})

    hidden_size: int = field(default=768, metadata={"help": "rnn hidden_size in RNN"})
    inner_dim: int = field(default=512, metadata={"help": "rnn hidden_size in RNN"})
    activation_fn: str = field(
        default="selu",
        metadata={"help": " activation function"},
    )
    num_layers: int = field(default=2, metadata={"help": "rnn number layers in RNN"})
    dropout_rnn: float = field(default=0.0, metadata={"help": "dropout of RNN"})
    dropout: float = field(default=0.0, metadata={"help": "dropout of FC"})
    bidirectional: bool = field(
        default=False, metadata={"help": "whether to use bidirectional"}
    )


class RNNEncoder3D(nn.Module):
    def __init__(
        self,
        input_dim=768,
        layer_type="GRU",
        hidden_size=768,
        num_layers=2,
        bidirectional=False,
        dropout_rnn=0.0,
        output_size=512,
        activation_fn="gelu",
    ):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Define Bidirectional GRU layer
        self.rnn = getattr(nn, str(layer_type))(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rnn,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Define a fully connected layer
        # If bidirectional, hidden_size is multiplied by 2
        if output_size == 0:
            self.fc = None
        else:
            self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
            self.activation_fn = get_activation_fn(activation_fn)

    def init_hidden(self, x):
        """
        Initialize hidden state with zeros.

        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
        - h0 or (h0, c0) (torch.Tensor): Initialized hidden state.
        """
        batch_size = x.size(0)

        # Initialize hidden state with zeros
        # Dimensions: (num_layers * num_directions, seq_length, hidden_size)
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(x)
        if self.layer_type == "LSTM":
            c0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            ).to(x)
            hidden_states = (h0, c0)
        else:
            hidden_states = h0
        return hidden_states

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, seq_length, output_size).
        """

        # Initialize hidden state
        hidden = self.init_hidden(x)

        # Forward propagate rnn
        with torch.backends.cudnn.flags(enabled=False):
            out, hidden = self.rnn(
                x, hidden
            )  # out: (batch_size, seq_length, hidden_size * num_directions)

        # Pass through the fully connected layer
        if self.fc is not None:
            out = self.fc(out)  # (batch_size, seq_length, output_size)
            out = self.activation_fn(out)
        return out


@dataclass
class NormLinearConfig(FairseqDataclass):
    activation_fn: str = field(
        default="swish",
        metadata={"help": "activation function"},
    )
    norm_type: NORM_CHOICES = field(
        default="default",
        metadata={"help": "norm type used in linear projection"},
    )


@dataclass
class Wav2Vec2SeldConfig(Wav2Vec2AsrConfig, Wav2Vec2ChConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": "number of layers to freeze in the pretrained transformer model"
        },
    )
    features_norm_linear: bool = field(
        default=False,
        metadata={"help": "apply normalization in the between w2v to headers"},
    )
    norm_linear: NormLinearConfig = NormLinearConfig()
    w2v_groups: bool = field(
        default=False,
        metadata={"help": "separate the w2v-model in two groups"},
    )
    remove_pretrained_modules: bool = field(
        default=True, metadata={"help": "whether to remove pretrained modules"}
    )
    ignore_mismatched_sizes: bool = field(
        default=False, metadata={"help": "whether to ignore mismatched sizes"}
    )
    freeze_norm_input: bool = field(
        default=True,
        metadata={"help": ("unfreeze the norm input")},
    )
    classifier_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in ClassifierHead"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets ClassifierHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )
    regression_cat: bool = field(
        default=False, metadata={"help": "use multiple regression heads model"}
    )
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})
    n_bins: int = field(
        default=1, metadata={"help": "number bins used in discretize if enabled"}
    )
    align_outputs_frames: bool = field(
        default=False, metadata={"help": "apply align frames at outputs"}
    )
    align_only_inference: bool = field(
        default=False,
        metadata={
            "help": "align only in inference, training with frames at ~20ms, but inference is at 100ms"
        },
    )
    align_before_head: bool = field(
        default=True, metadata={"help": "apply align conv pool before head"}
    )
    align_conv_pool: bool = field(
        default=False, metadata={"help": "apply align conv pool at outputs"}
    )
    align_conv_pool_norm: NormLinearConfig = NormLinearConfig()
    label_hop_len_s: float = II("task.label_hop_len_s")
    sample_rate: int = II("task.sample_rate")


@dataclass
class Wav2Vec2SeldClassConfig(Wav2Vec2SeldConfig):
    classifier_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of pooler"},
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    classifier_proj_size: int = field(
        default=768, metadata={"help": "inner_dim in ClassifierHead"}
    )
    proj_before_pooler: bool = field(
        default=False,
        metadata={
            "help": "whether to project before of aftermean-pooling in ClassifierHead"
        },
    )


@dataclass
class Wav2Vec2SeldSeqClassConfig(Wav2Vec2SeldClassConfig):
    pass


@dataclass
class Wav2Vec2SeldSequeceClassConfig(Wav2Vec2SeldClassConfig):
    regression_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of pooler"},
    )
    regression_proj_size: int = field(
        default=768, metadata={"help": "inner_dim in ClassifierHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of pooler in ClassifierHead"}
    )


@dataclass
class Wav2Vec2SeqSeldSequeceClassConfig(Wav2Vec2SeldSequeceClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConfig(Wav2Vec2SeldConfig):
    classifier_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of head"},
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    classifier_proj_size: Any = field(
        default=768, metadata={"help": "inner dimensions of classifier"}
    )
    regression_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of regression inner"},
    )
    regression_proj_size: Any = field(
        default=768, metadata={"help": "inner dimensions of regression"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassConfig(Wav2Vec2SeldAudioFrameClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassMLDecoderConfig(Wav2Vec2SeldAudioFrameClassConfig):
    num_of_groups: Optional[int] = field(
        default=-1, metadata={"help": "number of groups in the decoder"}
    )
    decoder_embedding: Optional[int] = field(
        default=768, metadata={"help": "decoder embedding size"}
    )
    zsl: Optional[int] = field(
        default=0, metadata={"help": "zero shot learning in the decoder"}
    )


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassMLDecoderConfig(
    Wav2Vec2SeldAudioFrameClassMLDecoderConfig
):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassTCNConfig(Wav2Vec2SeldAudioFrameClassConfig):
    fc_regression: Optional[bool] = field(
        default=False, metadata={"help": "use regression as FC block"}
    )
    bidirectional: Optional[bool] = field(
        default=False, metadata={"help": "use bidirectional block"}
    )
    merge_cat: Optional[bool] = field(
        default=False, metadata={"help": "use merge cat in bidirectional block"}
    )
    attention: Optional[bool] = field(
        default=False, metadata={"help": "use attention block"}
    )
    att_type: ATT_CHOICES = field(
        default="add-attention", metadata={"help": "attention type"}
    )

    classifier: TcnConfig = TcnConfig()
    regression: TcnConfig = TcnConfig()


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassTCNConfig(Wav2Vec2SeldAudioFrameClassTCNConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassInceptionTimeConfig(Wav2Vec2SeldConfig):
    classifier_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of head"},
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    classifier_bn: bool = field(
        default=False, metadata={"help": "batch norm in ClassifierHead"}
    )
    classifier_filters: int = field(
        default=32, metadata={"help": "number of filters of classifier"}
    )
    classifier_depth: int = field(default=6, metadata={"help": "depth of classifier"})
    regression_activation_fn: str = field(
        default="gelu",
        metadata={"help": " activation function of regression inner"},
    )
    regression_filters: int = field(
        default=32, metadata={"help": "number of filters of regression"}
    )
    regression_depth: int = field(default=6, metadata={"help": "depth of regression"})
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )
    regression_bn: bool = field(
        default=False, metadata={"help": "batch norm in RegressionHead"}
    )


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassInceptionTimeConfig(
    Wav2Vec2SeldAudioFrameClassInceptionTimeConfig
):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassMultiScaleTCNConfig(Wav2Vec2SeldConfig):
    bidirectional: Optional[bool] = field(
        default=False, metadata={"help": "use bidirectional block"}
    )
    merge_cat: Optional[bool] = field(
        default=False, metadata={"help": "use merge cat in bidirectional block"}
    )
    classifier: MultiScaleTcnConfig = MultiScaleTcnConfig()
    regression: MultiScaleTcnConfig = MultiScaleTcnConfig()


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassMultiScaleTCNConfig(
    Wav2Vec2SeldAudioFrameClassMultiScaleTCNConfig
):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassRNNConfig(Wav2Vec2SeldAudioFrameClassConfig):
    classifier: RNNConfig = RNNConfig()
    regression: RNNConfig = RNNConfig()


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassRNNConfig(Wav2Vec2SeldAudioFrameClassRNNConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConformerConfig(Wav2Vec2SeldConfig):
    classifier_encoder: Wav2Vec2HeaderConfig = Wav2Vec2HeaderConfig()
    regression_encoder: Wav2Vec2HeaderConfig = Wav2Vec2HeaderConfig()


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassConformerConfig(
    Wav2Vec2SeldAudioFrameClassConformerConfig
):
    pass


@register_model("wav2vec2_class", dataclass=Wav2Vec2SeldSeqClassConfig)
class Wav2vecSeqClass(Wav2VecCtc):
    def __init__(self, cfg: Wav2Vec2SeldSeqClassConfig, w2v_encoder: BaseFairseqModel):
        BaseFairseqModel.__init__(self)
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeldSeqClassConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldSequenceClassEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        return logits

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"].long()


@register_model(
    "wav2vec2_seld_sequence_class", dataclass=Wav2Vec2SeqSeldSequeceClassConfig
)
class Wav2vec2SeqSeldSequenceClassEncoder(Wav2vecSeqClass):
    def __init__(
        self, cfg: Wav2Vec2SeqSeldSequeceClassConfig, w2v_encoder: BaseFairseqModel
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldSequeceClassConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldSequenceClassEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class", dataclass=Wav2Vec2SeqSeldAudioFrameClassConfig
)
class Wav2vec2SeqSeldAudioFrameClassEncoder(Wav2vecSeqClass):
    def __init__(
        self, cfg: Wav2Vec2SeqSeldAudioFrameClassConfig, w2v_encoder: BaseFairseqModel
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAudioFrameClassConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class_tcn",
    dataclass=Wav2Vec2SeqSeldAudioFrameClassTCNConfig,
)
class Wav2vec2SeqSeldAudioFrameClassTCNEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAudioFrameClassTCNConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAudioFrameClassTCNConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassTCNEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class_inception_time",
    dataclass=Wav2Vec2SeqSeldAudioFrameClassInceptionTimeConfig,
)
class Wav2vec2SeqSeldAudioFrameClassInceptionTimeEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAudioFrameClassInceptionTimeConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAudioFrameClassInceptionTimeConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassInceptionTimeEncoder(
            cfg, cfg.target_length
        )
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class_mstcn",
    dataclass=Wav2Vec2SeqSeldAudioFrameClassMultiScaleTCNConfig,
)
class Wav2vec2SeqSeldAudioFrameClassMultiScaleTCNEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAudioFrameClassMultiScaleTCNConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAudioFrameClassMultiScaleTCNConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassMultiScaleTcnEncoder(
            cfg, cfg.target_length
        )
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class_rnn",
    dataclass=Wav2Vec2SeqSeldAudioFrameClassRNNConfig,
)
class Wav2vec2SeqSeldAudioFrameClassRNNEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAudioFrameClassRNNConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAudioFrameClassRNNConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassRNNEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model(
    "wav2vec2_seld_audio_frame_class_conformer",
    dataclass=Wav2Vec2SeqSeldAudioFrameClassConformerConfig,
)
class Wav2vec2SeqSeldAudioFrameClassConformerEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAudioFrameClassConformerConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAudioFrameClassConformerConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassConformerEncoder(
            cfg, cfg.target_length
        )
        return cls(cfg, w2v_encoder)


class Wav2vec2SeqClassHead(nn.Module):
    """
    Head for sequence classification tasks following hugging-face wav2vec2
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification

    """

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_outs,
        activation_fn,
        pooler_dropout_input,
        pooler_dropout,
    ):
        super().__init__()
        self.pooler = _mean_pooling

        self.dropout_input = nn.Dropout(p=pooler_dropout_input)

        if inner_dim == 0:
            self.dense = None
        else:
            self.dense = nn.Linear(input_dim, inner_dim)
            self.activation_fn = get_activation_fn(activation_fn)

        self.dropout = nn.Dropout(p=pooler_dropout)

        if inner_dim == 0:
            self.out_proj = torch.nn.Linear(input_dim, num_outs)
        else:
            self.out_proj = torch.nn.Linear(inner_dim, num_outs)

    def forward(self, features, padding_mask, **kwargs):
        x = self.dropout(features)

        if self.dense:
            x = self.dense(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.pooler(features, padding_mask)

        x = self.out_proj(x)
        return x


class Wav2vec2BertClassHead(nn.Module):
    """
    Head for sequence classification tasks. Based on BERT with hidden
    layer after mean-pooling followed by a tanh (or other one). We also modify
    the to apply normalization to the dense layer.

    The header can also be applied to regression tasks using num_outs=1.
    """

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_outs,
        activation_fn,
        pooler_dropout_input,
        pooler_dropout,
    ):
        super().__init__()

        self.pooler = _mean_pooling

        self.dropout_input = nn.Dropout(p=pooler_dropout_input)

        if inner_dim == 0:
            self.dense = None
        else:
            self.dense = nn.Linear(input_dim, inner_dim)
            self.activation_fn = get_activation_fn(activation_fn)
            self.dropout = nn.Dropout(p=pooler_dropout)

        if inner_dim == 0:
            self.out_proj = torch.nn.Linear(input_dim, num_outs)
        else:
            self.out_proj = torch.nn.Linear(inner_dim, num_outs)

    def forward(self, features, padding_mask, **kwargs):
        x = self.dropout_input(features)

        x = self.pooler(x, padding_mask)

        if self.dense:
            x = self.dense(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.out_proj(x)
        return x


class Wav2vec2AudioFrameClassHead(nn.Module):
    """
    Head for audioframe classification tasks. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        inner_dims,
        num_outs,
        activation_fn,
        out_activation_fn,
        dropout_input,
        dropout,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)

        if isinstance(inner_dims, int):
            if inner_dims == 0:
                self.dense = None
            else:
                self.dense = nn.Linear(input_dim, inner_dims)
                self.activation_fn = get_activation_fn(activation_fn)
                self.dropout = nn.Dropout(p=dropout)
        else:
            layers = []
            in_dim = input_dim
            for dim in inner_dims:
                layers.append(
                    [
                        nn.Linear(in_dim, dim),
                        get_activation_fn(activation_fn),
                        nn.Dropout(p=dropout),
                    ]
                )
                in_dim = dim
            layers = sum(layers, [])
            self.dense = nn.Sequential(*layers)
            self.activation_fn = None

        if isinstance(inner_dims, int):
            if inner_dims == 0:
                self.out_proj = torch.nn.Linear(input_dim, num_outs)
                self.out_activation_fn = get_activation_fn(out_activation_fn)
            else:
                self.out_proj = torch.nn.Linear(inner_dims, num_outs)
                self.out_activation_fn = get_activation_fn(out_activation_fn)
        else:
            self.out_proj = torch.nn.Linear(inner_dims[-1], num_outs)
            self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        x = self.dropout_input(features)

        if self.dense:
            x = self.dense(x)

            if self.activation_fn:
                x = self.activation_fn(x)
                x = self.dropout(x)

        x = self.out_proj(x)
        x = self.out_activation_fn(x)
        return x


class Wav2vec2AudioFrameClassCatHead(nn.Module):
    """
    Head for audioframe classification tasks using TCN. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(self, input_dim, cfg, num_outs):
        super().__init__()

        self.cfg = cfg
        self.cat_heads = nn.ModuleList()
        for d in range(cfg.doa_size):
            self.cat_heads.append(
                Wav2vec2AudioFrameClassHead(
                    input_dim=input_dim,
                    inner_dims=cfg.regression_proj_size,
                    num_outs=num_outs * cfg.n_bins,
                    activation_fn=cfg.regression_activation_fn,
                    out_activation_fn=cfg.regression_out_activation_fn,
                    dropout_input=cfg.regression_input_dropout,
                    dropout=cfg.regression_dropout,
                )
            )

    def forward(self, features, padding_mask=None):
        preds = []
        for head in self.cat_heads:
            pred = head(features)
            if self.cfg.n_bins > 1:
                B, T, N = pred.shape
                pred = pred.reshape(B, T, self.cfg.target_length, self.cfg.n_bins)
            preds.append(pred)
        if self.cfg.n_bins > 1:
            preds = torch.stack(preds, dim=3)  # (B, T, N, 3, BINS)
        else:
            preds = torch.cat(preds, dim=-1)  # (B, T, 3N)
        return preds


class Wav2vec2AudioFrameClassTCNHead(nn.Module):
    """
    Head for audioframe classification tasks using TCN. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        dropout_input,
        bidirectional,
        merge_cat,
        attention,
        att_type,
        cfg,
        num_outs,
        out_activation_fn,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)
        self.bidirectional = bidirectional
        self.merge_cat = merge_cat

        cfg.num_inputs = input_dim

        if self.bidirectional and self.merge_cat:
            embed_dim = 2 * cfg["num_channels"][-1]
        else:
            embed_dim = cfg["num_channels"][-1]

        cfg = OmegaConf.to_container(cfg)
        cfg.pop("_name")

        self.tcn = TCN(**cfg)

        if self.bidirectional:
            self.tcn_r = TCN(**cfg)

        self.att_type = att_type
        if attention:
            if self.att_type == "self-attention":
                self.attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
            elif self.att_type == "local-aware-attention":
                self.attn = LocationAwareAttention(embed_dim)
            else:
                self.attn = BahdanauAttention(embed_dim)
        else:
            self.attn = None

        self.out_proj = torch.nn.Linear(embed_dim, num_outs)
        self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        x = self.dropout_input(features)
        r = self.tcn(x)

        if self.bidirectional:
            x_rl = torch.flip(x, [1])  # reverse T dimension
            r_rl = self.tcn_r(x_rl)
            r_rl = torch.flip(r_rl, [1])  # return T again
            if self.merge_cat:
                r = torch.cat((r, r_rl), dim=-1)
            else:
                r = r + r_rl

        if self.attn is not None:
            if self.att_type == "self-attention":
                if padding_mask is not None:
                    B, T = padding_mask.shape
                    mask = padding_mask.unsqueeze(1).expand(B, T, T)
                    mask = mask.repeat(8, 1, 1)
                else:
                    mask = padding_mask
                r, _ = self.attn(r, r, r, attn_mask=mask)
            else:
                _, weights = self.attn(r, r, mask=padding_mask)
                weights = weights.squeeze(1).unsqueeze(-1)
                r = r * weights

        r = self.out_proj(r)
        r = self.out_activation_fn(r)
        return r


class Wav2vec2AudioFrameClassTCNCatHead(nn.Module):
    """
    Head for audioframe classification tasks using TCN. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(self, input_dim, cfg, num_outs):
        super().__init__()

        self.cfg = cfg
        self.cat_heads = nn.ModuleList()
        for d in range(cfg.doa_size):
            self.cat_heads.append(
                Wav2vec2AudioFrameClassTCNHead(
                    input_dim=input_dim,
                    dropout_input=cfg.regression_input_dropout,
                    cfg=cfg.regression,
                    attention=cfg.attention,
                    att_type=cfg.att_type,
                    bidirectional=cfg.bidirectional,
                    merge_cat=cfg.merge_cat,
                    num_outs=num_outs * cfg.n_bins,
                    out_activation_fn=cfg.regression_out_activation_fn,
                )
            )

    def forward(self, features, padding_mask=None):
        preds = []
        for head in self.cat_heads:
            pred = head(features)
            if self.cfg.n_bins > 1:
                B, T, N = pred.shape
                pred = pred.reshape(B, T, self.cfg.target_length, self.cfg.n_bins)
            preds.append(pred)
        if self.cfg.n_bins > 1:
            preds = torch.stack(preds, dim=3)  # (B, T, N, 3, BINS)
        else:
            preds = torch.cat(preds, dim=-1)  # (B, T, 3N)
        return preds


class Wav2vec2AudioFrameClassInceptionTimeHead(nn.Module):
    """
    Head for audioframe classification tasks with InceptionTime.

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        filters,
        depth,
        num_outs,
        activation_fn,
        out_activation_fn,
        dropout_input,
        dropout,
        bn,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)

        self.inception_time = InceptionTimeModel(
            input_size=input_dim,
            filters=filters,
            depth=depth,
            activation_fn=activation_fn,
            batch_norm=bn,
            dropout=dropout,
        )
        self.out_proj = torch.nn.Linear(4 * filters, num_outs)
        self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        x = self.dropout_input(features)
        x = self.inception_time(x)
        x = self.out_proj(x)
        x = self.out_activation_fn(x)
        return x


class Wav2vec2AudioFrameClassRNNHead(nn.Module):
    """
    Head for audioframe classification tasks. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        dropout_input,
        cfg: RNNConfig,
        num_outs,
        out_activation_fn,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)
        self.rnn_encoder = RNNEncoder3D(
            input_dim=input_dim,
            layer_type=cfg.layer_type,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
            dropout_rnn=cfg.dropout_rnn,
            output_size=cfg.inner_dim,
            activation_fn=cfg.activation_fn,
        )

        if cfg.inner_dim == 0:
            if cfg.bidirectional:
                hidden_size = 2 * cfg.hidden_size
            self.out_proj = torch.nn.Linear(hidden_size, num_outs)
        else:
            self.out_proj = torch.nn.Linear(cfg.inner_dim, num_outs)

        self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask):
        if padding_mask is None:
            padding_mask = torch.zeros(
                (features.size(0), features.size(1)), device=features.device
            ).type(torch.bool)

        features = self.dropout_input(features)
        x = self.rnn_encoder(features)
        x = self.out_proj(x)
        x = self.out_activation_fn(x)
        return x


class ConformerFrameHead(nn.Module):
    """Head for sentence-level classification tasks using ConformerEncoder."""

    def __init__(
        self,
        input_dim,
        dropout_input,
        cfg,
        num_outs,
        out_activation_fn="linear",
    ):
        super().__init__()
        if input_dim != cfg.encoder_embed_dim:
            self.input_proj = torch.nn.Linear(input_dim, cfg.encoder_embed_dim)
        else:
            self.input_proj = None
        self.dropout_input = nn.Dropout(p=dropout_input)
        encoder_cls = TransformerEncoderHeader
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoderHeader

        self.encoder = encoder_cls(cfg)

        # assert self.encoder.layer_norm_first is True

        # fix SiLU activations to false
        SiLU_inplace_to_False(self.encoder)

        self.num_outs = num_outs
        if self.num_outs is not None:
            self.out_proj = torch.nn.Linear(cfg.encoder_embed_dim, num_outs)
            self.out_activation_fn = get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask):
        if self.input_proj:
            features = self.input_proj(features)
        x = self.dropout_input(features)

        x, _ = self.encoder(x, padding_mask=padding_mask)

        if self.num_outs is not None:
            x = self.out_proj(x)
            x = self.out_activation_fn(x)
        return x


class Wav2vec2SequenceClassEncoder(Wav2VecEncoder):
    """
    Similar to Wav2Vec2ForSequenceClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification

    Wav2Vec2 Model with a sequence classification head on top (a linear layer
    over the pooled output) for tasks like SUPERB Keyword Spotting.
    """

    def __init__(self, cfg: Wav2Vec2SeldClassConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        if cfg.proj_before_pooler:
            self.classifier_head = Wav2vec2SeqClassHead(
                input_dim=d,
                inner_dim=cfg.classifier_proj_size,
                num_outs=tgt_len,
                activation_fn=cfg.classifier_activation_fn,
                pooler_dropout_input=cfg.classifier_input_dropout,
                pooler_dropout=cfg.classifier_dropout,
            )
        else:
            self.classifier_head = Wav2vec2BertClassHead(
                input_dim=d,
                inner_dim=cfg.classifier_proj_size,
                num_outs=tgt_len,
                activation_fn=cfg.classifier_activation_fn,
                pooler_dropout_input=cfg.classifier_input_dropout,
                pooler_dropout=cfg.classifier_dropout,
            )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

    def overrides_cfg_model(self, cfg):
        self.apply_mask = cfg.apply_mask
        self.cfg = cfg

        arg_overrides = {
            "conv_feature_layers": cfg.conv_feature_layers,
            "in_channels": cfg.in_channels,
            "in_conv_groups": cfg.in_conv_groups,
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both"
            "pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data

        if "precompute_mask_indices" in w2v_args.task:
            with open_dict(w2v_args):
                w2v_args.task.pop("precompute_mask_indices")
                w2v_args.task.pop("inferred_w2v_config")

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model
        if not cfg.grad_norm:
            self.w2v_model.compile()

        if hasattr(self.w2v_model.feature_extractor, "norm_input"):
            if self.w2v_model.feature_extractor.norm_input is not None:
                assert w2v_args.model.get("learn_norm_input", False)
                if cfg.get("freeze_norm_input", True):
                    freeze_module_params(self.w2v_model.feature_extractor.norm_input)
                    logger.info("freezed w2v_model.feature_extractor.norm_input")

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        if cfg.w2v_groups:
            for p in self.w2v_model.feature_extractor.parameters():
                p.param_group = "w2v_feature_extractor"

            for p in self.w2v_model.post_extract_proj.parameters():
                p.param_group = "w2v_feature_extractor"

            for p in self.w2v_model.encoder.parameters():
                p.param_group = "w2v_encoder"

            self.w2v_model.mask_emb.param_group = "w2v_encoder"

            for p in self.w2v_model.layer_norm.parameters():
                p.param_group = "w2v_encoder"
        else:
            for p in self.w2v_model.parameters():
                p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.get("features_norm_linear", False):
            self.final_norm = LinearNorm(d=d, cfg=cfg.norm_linear)

            for p in self.final_norm.parameters():
                p.param_group = "head"
        else:
            self.final_norm = None

        return d

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
                strict = False
            else:
                strict = True
                state_model = state["model"]

            model.load_state_dict(state_model, strict=strict)

        return model

    def extract_features(self, w2v_args):
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

        x = self.final_dropout(x)
        return x, padding_mask

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # extract features from wav2vec2 model
        x, padding_mask = self.extract_features(w2v_args)

        # apply final normalization if needed
        if self.final_norm is not None:
            x = self.final_norm(x)

        x = self.classifier_head(x, padding_mask=padding_mask)

        return {
            "encoder_out": x,  # B x N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }


class Wav2vec2SeldSequenceClassEncoder(Wav2vec2SequenceClassEncoder):
    def __init__(self, cfg: Wav2Vec2SeldSequeceClassConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        if cfg.proj_before_pooler:
            self.classifier_head = Wav2vec2SeqClassHead(
                input_dim=d,
                inner_dim=cfg.classifier_proj_size,
                num_outs=tgt_len * cfg.doa_size,
                activation_fn=cfg.classifier_activation_fn,
                pooler_dropout_input=cfg.classifier_input_dropout,
                pooler_dropout=cfg.classifier_dropout,
            )
        else:
            self.regression_head = Wav2vec2BertClassHead(
                input_dim=d,
                inner_dim=cfg.regression_proj_size,
                num_outs=tgt_len * cfg.doa_size,
                activation_fn=cfg.regression_activation_fn,
                pooler_dropout_input=cfg.regression_input_dropout,
                pooler_dropout=cfg.regression_dropout,
            )

        for p in self.regression_head.parameters():
            p.param_group = "head"

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # extract features from wav2vec2 model
        x, padding_mask = self.extract_features(w2v_args)

        # apply final normalization if needed
        if self.final_norm is not None:
            x = self.final_norm(x)

        class_logits = self.classifier_head(x, padding_mask=padding_mask)
        regression_logits = self.regression_head(x, padding_mask=padding_mask)

        return {
            "class_encoder_out": class_logits,  # B x N
            "regression_out": regression_logits,  # B x doa_size*N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,  # B x T
        }


class Wav2vec2SeldAudioFrameClassEncoder(Wav2vec2SequenceClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    Wav2Vec2 Model with a frame classification head on top for tasks like
    Speaker Diarization.
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        if cfg.get("align_before_head", True):
            self.setup_alignment(d, cfg)

        self.classifier_head = Wav2vec2AudioFrameClassHead(
            input_dim=d,
            inner_dims=cfg.classifier_proj_size,
            num_outs=tgt_len,
            activation_fn=cfg.classifier_activation_fn,
            out_activation_fn="linear",
            dropout_input=cfg.classifier_input_dropout,
            dropout=cfg.classifier_dropout,
        )
        if not cfg.grad_norm:
            self.classifier_head.compile()

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        if cfg.regression_cat:
            self.regression_head = Wav2vec2AudioFrameClassCatHead(
                input_dim=d,
                cfg=cfg,
                num_outs=tgt_len,
            )
        else:
            self.regression_head = Wav2vec2AudioFrameClassHead(
                input_dim=d,
                inner_dims=cfg.regression_proj_size,
                num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
                activation_fn=cfg.regression_activation_fn,
                out_activation_fn=cfg.regression_out_activation_fn,
                dropout_input=cfg.regression_input_dropout,
                dropout=cfg.regression_dropout,
            )
        if not cfg.grad_norm:
            self.regression_head.compile()

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"

        if cfg.get("align_before_head", True) is False:
            self.setup_alignment(d, cfg)

    def setup_alignment(self, d, cfg):
        """
        Setup alignment for outputs based on the configuration.
        If align_outputs_frames is True, we will pool the outputs to 100ms frames or ~20.1005ms frames
        using either Conv1dCeil or AvgPool1d.
        If align_outputs_frames is False, we will keep the outputs at 20.ms frames.
        """
        if cfg.get("align_outputs_frames", False):
            if cfg.get("label_hop_len_s", 0.02) == 0.1:
                if cfg.get("align_conv_pool", False):
                    if cfg.get("align_conv_pool_norm", None) is not None:
                        norm_type = cfg.align_conv_pool_norm.get("norm_type", "default")
                        activation_fn = cfg.align_conv_pool_norm.get(
                            "activation_fn", "linear"
                        )
                    else:
                        norm_type = None
                        activation_fn = "linear"
                    self.pool = Conv1dCeil(
                        in_channels=d,
                        out_channels=d,
                        kernel_size=5,
                        stride=5,
                        norm_type=norm_type,
                        activation_fn=activation_fn,
                    )

                    for p in self.pool.parameters():
                        p.param_group = "head"
                else:
                    self.pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)
                self.mask_pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)
                self.align_100ms = True
            else:
                self.align_100ms = False
        else:
            assert cfg.get("label_hop_len_s", 0.02) == 0.02, (
                "if align_outputs_frames is False, then label_hop_len_s should be 20ms"
            )
            self.align_100ms = False

    def get_last_shared_layer(self):
        if self.final_norm is not None:
            last_shared_layer = self.final_norm
        else:
            if (
                self.cfg.get("align_outputs_frames", False)
                and self.align_100ms
                and self.cfg.get("align_conv_pool", False)
            ):
                # if final_norm is not used and we take the 100ms pooling
                last_shared_layer = self.pool
            else:
                # if final_norm is not used, we take the last layer of the encoder
                last_shared_layer = self.w2v_model.encoder.layers[-1]
        return last_shared_layer

    def resample_logits_to_ms(
        self, logits, input_length, sample_rate=16000, target_length=0.02
    ):
        """
        Resample logits from ~20.1005ms frames to 20ms frames for uniform input lengths.

        Args:
            logits: Tensor [B, T_src, C] — B=batch, T_src=source frames, C=classes
            input_length: Scalar — waveform length in samples (same for all batch)

        Returns:
            Tensor: [B, T_target, C] — resampled to 20ms grid
        """
        B, T_src, C = logits.shape

        # Compute duration
        duration = input_length / sample_rate  # in seconds

        # Compute target frame count
        T_target = int(duration / target_length)

        # Reshape for interpolate: [B, C, T_src]
        logits_ = logits.permute(0, 2, 1)

        # Apply interpolation
        resampled = F.interpolate(
            logits_,
            size=T_target,
            mode="linear",
            align_corners=True,
        )  # [B, C, T_target]

        # Back to [B, T_target, C]
        resampled = resampled.permute(0, 2, 1)

        return resampled

    def avg_pool_100ms(self, output):
        """
        Downsample wav2vec output to 100ms using AvgPool1d.

        Args:
            output: [B, T, D] Tensor

        Returns:
            [B, T_new, D] Tensor
        """
        out = self.pool(output.transpose(1, 2))  # [B, D, T_new]
        return out.transpose(1, 2)  # [B, T_new, D]

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # extract features from wav2vec2 model
        x, padding_mask = self.extract_features(w2v_args)

        if self.cfg.get("align_before_head", True) and self.align_100ms:
            # pool features and padding_mask to 100ms
            x = self.avg_pool_100ms(x)
            if padding_mask is not None:
                padding_mask = self.mask_pool(padding_mask.to(x.dtype)).bool()

        # apply final normalization if needed
        if self.final_norm is not None:
            x = self.final_norm(x)

        class_logits = self.classifier_head(x, padding_mask)
        regression_logits = self.regression_head(x, padding_mask)

        if self.cfg.get("align_outputs_frames", False):
            # if align_outputs_frames is True, we will pool the outputs to 100ms frames
            if not self.cfg.get("align_before_head", True):
                # if align_before_head is False, we either to resample logits 20ms or pool them to 100ms
                if self.cfg.get("align_only_inference", False):
                    # if align_only_inference is True, we pool to 100ms, only during inference, not during training
                    if self.training:
                        align = False
                    else:
                        align = True
                else:
                    align = True

                if align:
                    if self.align_100ms:
                        class_logits = self.avg_pool_100ms(class_logits)
                        regression_logits = self.avg_pool_100ms(regression_logits)
                    else:
                        class_logits = self.resample_logits_to_ms(
                            class_logits,
                            input_length=source.shape[1],
                            sample_rate=self.cfg.sample_rate,
                            target_length=self.cfg.label_hop_len_s,
                        )
                        regression_logits = self.resample_logits_to_ms(
                            regression_logits,
                            input_length=source.shape[1],
                            sample_rate=self.cfg.sample_rate,
                            target_length=self.cfg.label_hop_len_s,
                        )

        return {
            "class_encoder_out": class_logits,  # B x T x N
            "regression_out": regression_logits,  # B x T x doa_size*N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,  # B x T
        }


class Wav2vec2SeldAudioFrameClassTCNEncoder(Wav2vec2SeldAudioFrameClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    but with TCN header
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassTCNConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        self.classifier_head = Wav2vec2AudioFrameClassTCNHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            cfg=cfg.classifier,
            attention=cfg.attention,
            att_type=cfg.att_type,
            bidirectional=cfg.bidirectional,
            merge_cat=cfg.merge_cat,
            num_outs=tgt_len,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        if cfg.regression_cat:
            if cfg.fc_regression:
                self.regression_head = Wav2vec2AudioFrameClassCatHead(
                    input_dim=d,
                    inner_dims=cfg.regression_proj_size,
                    num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
                    activation_fn=cfg.regression_activation_fn,
                    out_activation_fn=cfg.regression_out_activation_fn,
                    dropout_input=cfg.regression_input_dropout,
                    dropout=cfg.regression_dropout,
                )
            else:
                self.regression_head = Wav2vec2AudioFrameClassTCNCatHead(
                    input_dim=d,
                    cfg=cfg,
                    num_outs=tgt_len,
                )
        else:
            if cfg.fc_regression:
                self.regression_head = Wav2vec2AudioFrameClassHead(
                    input_dim=d,
                    inner_dims=cfg.regression_proj_size,
                    num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
                    activation_fn=cfg.regression_activation_fn,
                    out_activation_fn=cfg.regression_out_activation_fn,
                    dropout_input=cfg.regression_input_dropout,
                    dropout=cfg.regression_dropout,
                )
            else:
                self.regression_head = Wav2vec2AudioFrameClassTCNHead(
                    input_dim=d,
                    dropout_input=cfg.regression_input_dropout,
                    cfg=cfg.regression,
                    attention=cfg.attention,
                    att_type=cfg.att_type,
                    bidirectional=cfg.bidirectional,
                    merge_cat=cfg.merge_cat,
                    num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
                    out_activation_fn=cfg.regression_out_activation_fn,
                )

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class Wav2vec2SeldAudioFrameClassInceptionTimeEncoder(
    Wav2vec2SeldAudioFrameClassEncoder
):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    but with InceptionTime header
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassInceptionTimeConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        self.classifier_head = Wav2vec2AudioFrameClassInceptionTimeHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            filters=cfg.classifier_filters,
            depth=cfg.classifier_depth,
            num_outs=tgt_len,
            activation_fn=cfg.classifier_activation_fn,
            dropout=cfg.classifier_dropout,
            bn=cfg.classifier_bn,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        self.regression_head = Wav2vec2AudioFrameClassInceptionTimeHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            filters=cfg.regression_filters,
            depth=cfg.regression_depth,
            num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
            activation_fn=cfg.regression_activation_fn,
            dropout=cfg.regression_dropout,
            bn=cfg.regression_bn,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class Wav2vec2SeldAudioFrameClassMultiScaleTcnEncoder(
    Wav2vec2SeldAudioFrameClassEncoder
):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    but with InceptionTime header
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassMultiScaleTCNConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        self.classifier_head = Wav2vec2AudioFrameClassMultiScaleTcnHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            bidirectional=cfg.bidirectional,
            merge_cat=cfg.merge_cat,
            num_outs=tgt_len,
            out_activation_fn="linear",
            cfg=cfg.classifier,
        )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        if cfg.regression_cat:
            self.regression_head = Wav2vec2AudioFrameClassMultiScaleTcnCatHead(
                input_dim=d,
                cfg=cfg,
                num_outs=tgt_len,
            )
        else:
            self.regression_head = Wav2vec2AudioFrameClassMultiScaleTcnHead(
                input_dim=d,
                dropout_input=cfg.regression_input_dropout,
                bidirectional=cfg.bidirectional,
                merge_cat=cfg.merge_cat,
                num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
                out_activation_fn=cfg.regression_out_activation_fn,
                cfg=cfg.regression,
            )

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class Wav2vec2SeldAudioFrameClassRNNEncoder(Wav2vec2SeldAudioFrameClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    Wav2Vec2 Model with a frame classification head on top for tasks like
    Speaker Diarization.
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassRNNConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        self.classifier_head = Wav2vec2AudioFrameClassRNNHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            cfg=cfg.classifier,
            num_outs=tgt_len,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        self.regression_head = Wav2vec2AudioFrameClassRNNHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            cfg=cfg.regression,
            num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class Wav2vec2SeldAudioFrameClassConformerEncoder(Wav2vec2SeldAudioFrameClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    Wav2Vec2 Model with a frame conformer classification head on top for tasks like
    Speaker Diarization.
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassConformerConfig, tgt_len=1):
        d = self.overrides_cfg_model(cfg)

        self.classifier_head = ConformerFrameHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            cfg=cfg.classifier_encoder,
            num_outs=tgt_len,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "head"

        self.regression_head = ConformerFrameHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            cfg=cfg.regression_encoder,
            num_outs=tgt_len * cfg.doa_size * cfg.n_bins,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        if mask is not None:
            _MASKING_VALUE = -1e30 if query.dtype == torch.float32 else -1e4
            mask = mask.unsqueeze(1)
            scores.masked_fill_(mask, _MASKING_VALUE)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        hidden_dim (int): dimesion of hidden state vector (D)
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn, smoothing
        - **query** (B, T, D): tensor containing the output features from the decoder.
        - **value** (B, T, D): tensor containing features of the encoded input sequence.
        - **mask** (B, T): tensor containing features of the encoded input sequence.
        - **last_attn** (B, T): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (B, D): tensor containing the feature from encoder outputs
        - **attn** (B, T): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """

    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        mask=None,
        last_attn=None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(
            torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(
                    batch_size, -1, hidden_dim
                )
                + self.value_proj(value.reshape(-1, hidden_dim)).view(
                    batch_size, -1, hidden_dim
                )
                + conv_attn
                + self.bias
            )
        ).squeeze(dim=-1)

        if mask is not None:
            _MASKING_VALUE = -1e30 if query.dtype == torch.float32 else -1e4
            score.masked_fill_(mask, _MASKING_VALUE)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(
            dim=1
        )  # Bx1xT X BxTxD => Bx1xD => BxD

        return context, attn


class LinearNorm(nn.Module):
    """
    Linear projection with Normalization
    """

    def __init__(
        self,
        d,
        cfg: NormLinearConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.proj = nn.Linear(d, d)
        if self.cfg.norm_type == "batch_norm":
            self.norm = Fp32BatchNorm1d(d)
        else:
            self.norm = Fp32LayerNorm(d)
        self.activation_fn = get_activation_fn(cfg.activation_fn)

    def forward(self, features):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = self.proj(features)

            if self.cfg.norm_type == "batch_norm":
                x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

            x = self.norm(x)

            if self.cfg.norm_type == "batch_norm":
                x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            x = self.activation_fn(x)
        return x.to(features.dtype)
