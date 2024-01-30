import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2AsrConfig,
    Wav2VecCtc,
    Wav2VecEncoder,
)
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.tasks import FairseqTask
from omegaconf import open_dict
from torch import Tensor

from seld_wav2vec2.model.conv import InceptionTimeModel
from seld_wav2vec2.model.tcn import (
    MultiScaleTcnConfig,
    TcnConfig,
    TemporalConvNet,
    Wav2vec2AudioFrameClassMultiScaleTcnHead,
)
from seld_wav2vec2.model.utils import SiLU_inplace_to_False, get_activation_fn
from seld_wav2vec2.model.wav2vec2_multi_ch import (
    ConformerEncoderHeader,
    TransformerEncoderHeader,
    Wav2Vec2HeaderConfig,
)

logger = logging.getLogger(__name__)


RNN_CHOICES = ChoiceEnum(["GRU", "LSTM"])
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


class RNNEncoder3D(nn.Module):
    """RNN 3D encoder (B, T, C)."""

    def __init__(
        self,
        layer_type="LSTM",
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.rnn = getattr(nn, str(layer_type))(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(
        self,
        src_input: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
    ):
        """
        Args:
            src_input (FloatTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (FloatTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_input = utils.convert_padding_direction(
                src_input,
                torch.zeros_like(src_input).fill_(self.padding_idx),
                left_to_right=True,
            )

        bsz, seqlen, _ = src_input.size()

        # input dropout
        x = self.dropout_in_module(src_input)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted
        )

        # apply RNN
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        with torch.backends.cudnn.flags(enabled=False):
            packed_outs, (final_hiddens, final_cells) = self.rnn(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
        )
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        # encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple(
            (
                x,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                # encoder_padding_mask,  # seq_len x batch
            )
        )

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(
        self, encoder_out: Tuple[Tensor, Tensor, Tensor, Tensor], new_order
    ):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
            )
        )


@dataclass
class Wav2Vec2SeldConfig(Wav2Vec2AsrConfig):
    remove_pretrained_modules: bool = field(
        default=True, metadata={"help": "whether to remove pretrained modules"}
    )
    ignore_mismatched_sizes: bool = field(
        default=False, metadata={"help": "whether to ignore mismatched sizes"}
    )
    in_channels: int = field(
        default=4, metadata={"help": "number of input channels - CNN"}
    )
    in_conv_groups: int = field(
        default=1, metadata={"help": "number of conv_group channels - CNN"}
    )


@dataclass
class Wav2Vec2SeldClassConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer" "model")
        },
    )
    classifier_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of pooler"},
    )
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout of pooler in ClassifierHead" "applied to input features"
        },
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    classifier_proj_size: int = field(
        default=768, metadata={"help": "inner_dim in ClassifierHead"}
    )
    proj_before_pooler: bool = field(
        default=False,
        metadata={
            "help": "whether to project before of after"
            "mean-pooling in ClassifierHead"
        },
    )


@dataclass
class Wav2Vec2SeldSeqClassConfig(Wav2Vec2SeldClassConfig):
    pass


@dataclass
class Wav2Vec2SeldSequeceClassConfig(Wav2Vec2SeldClassConfig):
    regression_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of pooler"},
    )
    regression_proj_size: int = field(
        default=768, metadata={"help": "inner_dim in ClassifierHead"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in ClassifierHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of pooler in ClassifierHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in ClassifierHead"}
    )


@dataclass
class Wav2Vec2SeqSeldSequeceClassConfig(Wav2Vec2SeldSequeceClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    use_recurrent_block: bool = field(
        default=False, metadata={"help": "use recurrent block"}
    )
    recurrent: Optional[TcnConfig] = TcnConfig()
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the header"}
    )
    classifier_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of head"},
    )
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout of in ClassifierHead applied to input features"},
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    classifier_proj_size: Any = field(
        default=768, metadata={"help": "inner dimensions of classifier"}
    )
    regression_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression inner"},
    )
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    regression_proj_size: Any = field(
        default=768, metadata={"help": "inner dimensions of regression"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassConfig(Wav2Vec2SeldAudioFrameClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassTCNConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    use_recurrent_block: Optional[bool] = field(
        default=False, metadata={"help": "use recurrent block"}
    )
    recurrent: Optional[TcnConfig] = TcnConfig()
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
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout of in ClassifierHead applied to input features"},
    )
    classifier_tcn: TcnConfig = TcnConfig()
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    regression_tcn: TcnConfig = TcnConfig()
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassTCNConfig(Wav2Vec2SeldAudioFrameClassTCNConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassInceptionTimeConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    use_recurrent_block: bool = field(
        default=False, metadata={"help": "use recurrent block"}
    )
    recurrent: Optional[TcnConfig] = TcnConfig()
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the header"}
    )
    classifier_activation_fn: str = field(
        default="relu",
        metadata={"help": " activation function of head"},
    )
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout of in ClassifierHead applied to input features"},
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    classifier_bn: bool = field(
        default=False, metadata={"help": "batch norm in ClassifierHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    classifier_filters: int = field(
        default=32, metadata={"help": "number of filters of classifier"}
    )
    classifier_depth: int = field(
        default=6, metadata={"help": "depth of classifier"}
    )
    regression_activation_fn: str = field(
        default="relu",
        metadata={"help": " activation function of regression inner"},
    )
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    regression_filters: int = field(
        default=32, metadata={"help": "number of filters of regression"}
    )
    regression_depth: int = field(
        default=6, metadata={"help": "depth of regression"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )
    regression_bn: bool = field(
        default=False, metadata={"help": "batch norm in RegressionHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassInceptionTimeConfig(
    Wav2Vec2SeldAudioFrameClassInceptionTimeConfig
):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassMultiScaleTCNConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    use_recurrent_block: bool = field(
        default=False, metadata={"help": "use recurrent block"}
    )
    recurrent: Optional[TcnConfig] = TcnConfig()
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout of in ClassifierHead applied to input features"},
    )
    classifier: MultiScaleTcnConfig = MultiScaleTcnConfig()
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    regression: MultiScaleTcnConfig = MultiScaleTcnConfig()
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassMultiScaleTCNConfig(
    Wav2Vec2SeldAudioFrameClassMultiScaleTCNConfig
):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassRNNConfig(Wav2Vec2SeldAudioFrameClassConfig):
    layer_type: RNN_CHOICES = field(default="GRU", metadata={"help": "rnn hidden type"})
    classifier_hidden_size: int = field(
        default=768, metadata={"help": "rnn hidden_size in ClassifierHead"}
    )
    regression_hidden_size: int = field(
        default=768, metadata={"help": "rnn hidden_size in RegressionHead"}
    )
    classifier_num_layers: int = field(
        default=2, metadata={"help": "rnn number layers in ClassifierHead"}
    )
    regression_num_layers: int = field(
        default=2, metadata={"help": "rnn number layers in ClassifierHead"}
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of rnn in ClassifierHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of rnn in RegressionHead"}
    )
    classifier_bidirectional: bool = field(
        default=False, metadata={"help": "whether to use bidirectional"}
    )
    regression_bidirectional: bool = field(
        default=False, metadata={"help": "whether to use bidirectional"}
    )
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassRNNConfig(Wav2Vec2SeldAudioFrameClassRNNConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConformerConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    use_recurrent_block: bool = field(
        default=False, metadata={"help": "use recurrent block"}
    )
    recurrent: Optional[Wav2Vec2HeaderConfig] = Wav2Vec2HeaderConfig()
    classifier_input_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout of in ClassifierHead applied to input features"},
    )
    classifier_encoder: Wav2Vec2HeaderConfig = Wav2Vec2HeaderConfig()
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in ClassifierHead"}
    )
    regression_encoder: Wav2Vec2HeaderConfig = Wav2Vec2HeaderConfig()
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in ClassifierHead"}
    )
    regression_out_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of regression output"},
    )
    grad_norm: bool = field(default=False, metadata={"help": "apply gradnorm to model"})


@dataclass
class Wav2Vec2SeqSeldAudioFrameClassConformerConfig(
    Wav2Vec2SeldAudioFrameClassConformerConfig
):
    pass


@dataclass
class Wav2Vec2SeldAccDoaAudioFrameClassConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer model")
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the header"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    regression_activation_fn: str = field(
        default="tanh",
        metadata={"help": " activation function of pooler"},
    )
    regression_proj_size: int = field(
        default=768, metadata={"help": "inner_dim in ClassifierHead"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in ClassifierHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in ClassifierHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in ClassifierHead"}
    )


@dataclass
class Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig(
    Wav2Vec2SeldAccDoaAudioFrameClassConfig
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


@register_model(
    "wav2vec2_seld_accdoa_audio_frame_class",
    dataclass=Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig,
)
class Wav2vec2SeqSeldAccDoaAudioFrameClassEncoder(Wav2vecSeqClass):
    def __init__(
        self,
        cfg: Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig,
        w2v_encoder: BaseFairseqModel,
    ):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(
        cls, cfg: Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig, task: FairseqTask
    ):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAccDoaAudioFrameClassEncoder(cfg, cfg.target_length)
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
            self.activation_fn = utils.get_activation_fn(activation_fn)

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
            self.activation_fn = utils.get_activation_fn(activation_fn)
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
        layer_norm_first,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        if layer_norm_first:
            self.layer_norm = LayerNorm(input_dim)

        self.dropout_input = nn.Dropout(p=dropout_input)

        if isinstance(inner_dims, int):
            if inner_dims == 0:
                self.dense = None
            else:
                self.dense = nn.Linear(input_dim, inner_dims)
                self.activation_fn = utils.get_activation_fn(activation_fn)
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
                self.out_activation_fn = utils.get_activation_fn(out_activation_fn)
            else:
                self.out_proj = torch.nn.Linear(inner_dims, num_outs)
                self.out_activation_fn = utils.get_activation_fn(out_activation_fn)
        else:
            self.out_proj = torch.nn.Linear(inner_dims[-1], num_outs)
            self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        if self.layer_norm_first:
            features = self.layer_norm(features)
        x = self.dropout_input(features)

        if self.dense:
            x = self.dense(x)

        if self.activation_fn:
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.out_proj(x)
        x = self.out_activation_fn(x)
        return x


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

        self.tcn = TemporalConvNet(cfg)

        if self.bidirectional:
            self.tcn_r = TemporalConvNet(cfg)

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
        self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

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
        layer_norm_first,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        if layer_norm_first:
            self.layer_norm = LayerNorm(input_dim)

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
        self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask=None):
        if self.layer_norm_first:
            features = self.layer_norm(features)
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
        layer_type,
        input_dim,
        hidden_size,
        inner_dim,
        num_layers,
        num_outs,
        activation_fn,
        dropout_input,
        dropout_rnn,
        dropout,
        bidirectional,
        out_activation_fn,
    ):
        super().__init__()

        self.rnn_encoder = RNNEncoder3D(
            layer_type=layer_type,
            embed_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_in=dropout_input,
            dropout_out=dropout_rnn,
            bidirectional=bidirectional,
            left_pad=False,
        )

        if inner_dim == 0:
            self.dense = None
        else:
            self.dense = nn.Linear(hidden_size, inner_dim)
            self.dropout = nn.Dropout(p=dropout)

        if inner_dim == 0:
            self.out_proj = torch.nn.Linear(hidden_size, num_outs)
        else:
            self.activation_fn = utils.get_activation_fn(activation_fn)
            self.out_proj = torch.nn.Linear(inner_dim, num_outs)
            self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

    def forward(self, features, padding_mask):
        if padding_mask is None:
            padding_mask = torch.zeros(
                (features.size(0), features.size(1)), device=features.device
            ).type(torch.bool)

        input_lengths, _ = (1 - padding_mask.long()).sum(-1).sort(descending=True)

        x, _, _ = self.rnn_encoder(features, src_lengths=input_lengths)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.dense:
            x = self.dense(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

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
            self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

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
            p.param_group = "classifier_head"

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        x = self.classifier_head(x, padding_mask=padding_mask)

        return {
            "encoder_out": x,  # B x N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }


class Wav2vec2SeldSequenceClassEncoder(Wav2vec2SequenceClassEncoder):
    def __init__(self, cfg: Wav2Vec2SeldSequeceClassConfig, tgt_len=1):
        super().__init__(cfg, tgt_len)

        self.cfg = cfg

        d = self.w2v_model.cfg.encoder_embed_dim

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
            p.param_group = "regression_head"

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.use_recurrent_block:
            cfg.recurrent.num_inputs = d
            self.recurrent = TemporalConvNet(cfg.recurrent)
            num_channels = cfg.recurrent.num_channels
            d = num_channels[-1]

            for p in self.recurrent.parameters():
                p.param_group = "recurrent"
        else:
            self.recurrent = None

        self.classifier_head = Wav2vec2AudioFrameClassHead(
            input_dim=d,
            inner_dims=cfg.classifier_proj_size,
            num_outs=tgt_len,
            activation_fn=cfg.classifier_activation_fn,
            out_activation_fn="linear",
            dropout_input=cfg.classifier_input_dropout,
            dropout=cfg.classifier_dropout,
            layer_norm_first=cfg.layer_norm_first,
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassHead(
            input_dim=d,
            inner_dims=cfg.regression_proj_size,
            num_outs=tgt_len * cfg.doa_size,
            activation_fn=cfg.regression_activation_fn,
            out_activation_fn=cfg.regression_out_activation_fn,
            dropout_input=cfg.regression_input_dropout,
            dropout=cfg.regression_dropout,
            layer_norm_first=cfg.layer_norm_first,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"

    def get_last_shared_layer(self):
        if self.recurrent is None:
            last_shared_layer = self.w2v_model.encoder.layers[-1]
        else:
            if hasattr(self.recurrent.encoder, "layers"):
                # last_shared_layer = self.recurrent.out_proj
                last_shared_layer = self.recurrent.encoder.layers[-1]
            else:
                last_shared_layer = self.recurrent.encoder[-1]
        return last_shared_layer

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        if self.recurrent is not None:
            x = self.recurrent(x, padding_mask)

        class_logits = self.classifier_head(x, padding_mask)
        regression_logits = self.regression_head(x, padding_mask)

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.use_recurrent_block:
            cfg.recurrent.num_inputs = d
            self.recurrent = TemporalConvNet(cfg.recurrent)
            num_channels = cfg.recurrent.num_channels
            d = num_channels[-1]

            for p in self.recurrent.parameters():
                p.param_group = "recurrent"
        else:
            self.recurrent = None

        self.classifier_head = Wav2vec2AudioFrameClassTCNHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            cfg=cfg.classifier_tcn,
            attention=cfg.attention,
            att_type=cfg.att_type,
            bidirectional=cfg.bidirectional,
            merge_cat=cfg.merge_cat,
            num_outs=tgt_len,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassTCNHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            cfg=cfg.regression_tcn,
            attention=cfg.attention,
            att_type=cfg.att_type,
            bidirectional=cfg.bidirectional,
            merge_cat=cfg.merge_cat,
            num_outs=tgt_len * cfg.doa_size,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.use_recurrent_block:
            cfg.recurrent.num_inputs = d
            self.recurrent = TemporalConvNet(cfg.recurrent)
            num_channels = cfg.recurrent.num_channels
            d = num_channels[-1]

            for p in self.recurrent.parameters():
                p.param_group = "recurrent"
        else:
            self.recurrent = None

        self.classifier_head = Wav2vec2AudioFrameClassInceptionTimeHead(
            input_dim=d,
            layer_norm_first=cfg.layer_norm_first,
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
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassInceptionTimeHead(
            input_dim=d,
            layer_norm_first=cfg.layer_norm_first,
            dropout_input=cfg.regression_input_dropout,
            filters=cfg.regression_filters,
            depth=cfg.regression_depth,
            num_outs=tgt_len * cfg.doa_size,
            activation_fn=cfg.regression_activation_fn,
            dropout=cfg.regression_dropout,
            bn=cfg.regression_bn,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.use_recurrent_block:
            cfg.recurrent.num_inputs = d
            self.recurrent = TemporalConvNet(cfg.recurrent)
            num_channels = cfg.recurrent.num_channels
            d = num_channels[-1]

            for p in self.recurrent.parameters():
                p.param_group = "recurrent"
        else:
            self.recurrent = None

        self.classifier_head = Wav2vec2AudioFrameClassMultiScaleTcnHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            num_outs=tgt_len,
            out_activation_fn="linear",
            cfg=cfg.classifier
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassMultiScaleTcnHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            num_outs=tgt_len * cfg.doa_size,
            out_activation_fn=cfg.regression_out_activation_fn,
            cfg=cfg.regression
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.classifier_head = Wav2vec2AudioFrameClassRNNHead(
            layer_type=cfg.layer_type,
            input_dim=d,
            hidden_size=cfg.classifier_hidden_size,
            inner_dim=cfg.classifier_proj_size,
            num_layers=cfg.classifier_num_layers,
            num_outs=tgt_len,
            activation_fn=cfg.classifier_activation_fn,
            dropout_input=cfg.classifier_input_dropout,
            dropout_rnn=cfg.classifier_dropout,
            dropout=cfg.classifier_dropout,
            bidirectional=cfg.classifier_bidirectional,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassRNNHead(
            layer_type=cfg.layer_type,
            input_dim=d,
            hidden_size=cfg.regression_hidden_size,
            inner_dim=cfg.regression_proj_size,
            num_layers=cfg.regression_num_layers,
            num_outs=tgt_len * cfg.doa_size,
            activation_fn=cfg.regression_activation_fn,
            dropout_input=cfg.regression_input_dropout,
            dropout_rnn=cfg.regression_dropout,
            dropout=cfg.regression_dropout,
            bidirectional=cfg.regression_bidirectional,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if cfg.use_recurrent_block:
            self.recurrent = ConformerFrameHead(
                input_dim=d,
                dropout_input=0.0,
                cfg=cfg.recurrent,
                num_outs=None,
                out_activation_fn="linear",
            )
            d = cfg.recurrent.encoder_embed_dim

            for p in self.recurrent.parameters():
                p.param_group = "recurrent"
        else:
            self.recurrent = None

        self.classifier_head = ConformerFrameHead(
            input_dim=d,
            dropout_input=cfg.classifier_input_dropout,
            cfg=cfg.classifier_encoder,
            num_outs=tgt_len,
            out_activation_fn="linear",
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = ConformerFrameHead(
            input_dim=d,
            dropout_input=cfg.regression_input_dropout,
            cfg=cfg.regression_encoder,
            num_outs=tgt_len * cfg.doa_size,
            out_activation_fn=cfg.regression_out_activation_fn,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

        if cfg.grad_norm:
            # assign the weights for each task
            self.weights = torch.nn.Parameter(torch.ones(2).float())
            self.weights.param_group = "weights_grad_norm"


class Wav2vec2SeldAccDoaAudioFrameClassEncoder(Wav2vec2SequenceClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    Wav2Vec2 Model with a frame classification head on top for tasks like
    Speaker Diarization.
    """

    def __init__(self, cfg: Wav2Vec2SeldAccDoaAudioFrameClassConfig, tgt_len=1):
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
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        if cfg.remove_pretrained_modules:
            model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        if cfg.n_trans_layers_to_freeze > 0:
            for layer in range(cfg.n_trans_layers_to_freeze):
                freeze_module_params(self.w2v_model.encoder.layers[layer])
                logger.info(f"freezed w2v_model.encoder layer: {layer}")

        for p in self.w2v_model.parameters():
            p.param_group = "w2v_model"

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.regression_head = Wav2vec2AudioFrameClassHead(
            input_dim=d,
            inner_dim=cfg.regression_proj_size,
            num_outs=tgt_len * cfg.doa_size,
            activation_fn=cfg.regression_activation_fn,
            dropout_input=cfg.regression_input_dropout,
            dropout=cfg.regression_dropout,
            layer_norm_first=self.cfg.layer_norm_first,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        regression_logits = self.regression_head(x)

        return {
            "regression_out": regression_logits,  # B x T x doa_size*N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,  # B x T
        }


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
