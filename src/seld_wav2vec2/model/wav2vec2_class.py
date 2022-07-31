import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.lstm import DEFAULT_MAX_SOURCE_POSITIONS, LSTMEncoder
from fairseq.models.wav2vec.wav2vec2 import (EXTRACTOR_MODE_CHOICES,
                                             ConformerEncoder, Wav2Vec2Config)
from fairseq.models.wav2vec.wav2vec2_asr import (Wav2Vec2AsrConfig, Wav2VecCtc,
                                                 Wav2VecEncoder)
from fairseq.modules import LayerNorm
from fairseq.tasks import FairseqTask
from omegaconf import open_dict
from torch import Tensor

from seld_wav2vec2.model.TCN import TemporalConvNet
from seld_wav2vec2.model.utils import get_activation_fn


class DummyDictionary:
    def __init__(
        self, vocab_size
    ):
        self.vocab_len = vocab_size
        self.symbols = []
        for i in range(self.vocab_len):
            self.symbols.append(i)
        self.count = self.vocab_len
        self.indices = {}
        for i in range(self.vocab_len):
            self.indices[i] = i

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def pad(self):
        """Helper to get index of pad symbol"""
        return None


logger = logging.getLogger(__name__)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)
             [None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


class LSTMEncoder3D(LSTMEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
        pretrained_embed=None,
        padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(dictionary,
                         embed_dim=embed_dim,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout_in=dropout_in,
                         dropout_out=dropout_out,
                         bidirectional=bidirectional,
                         left_pad=left_pad,
                         pretrained_embed=pretrained_embed,
                         padding_idx=padding_idx,
                         max_source_positions=max_source_positions)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )

        bsz, seqlen, _ = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x, (h0, c0))

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
            "help": ("number of layers to freeze in the pretrained transformer"
                     "model")
        },
    )
    classifier_activation_fn: str = field(
        default="tanh",
        metadata={
            "help": " activation function of pooler"
        },
    )
    classifier_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of pooler in ClassifierHead"
                               "applied to input features"}
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
        default=False, metadata={"help": "whether to project before of after"
                                 "mean-pooling in ClassifierHead"}
    )


@ dataclass
class Wav2Vec2SeldSeqClassConfig(Wav2Vec2SeldClassConfig):
    pass


@dataclass
class Wav2Vec2SeldSequeceClassConfig(Wav2Vec2SeldClassConfig):
    regression_activation_fn: str = field(
        default="tanh",
        metadata={
            "help": " activation function of pooler"
        },
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


@ dataclass
class Wav2Vec2SeqSeldSequeceClassConfig(Wav2Vec2SeldSequeceClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer"
                     "model")
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the header"}
    )
    classifier_activation_fn: str = field(
        default="tanh",
        metadata={
            "help": " activation function of head"
        },
    )
    classifier_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in ClassifierHead"
                               "applied to input features"}
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    classifier_proj_size: Any = field(
        default=768,
        metadata={"help": "inner dimensions of classifier"})
    regression_activation_fn: str = field(
        default="tanh",
        metadata={
            "help": " activation function of regression inner"
        },
    )
    regression_out_activation_fn: str = field(
        default="linear",
        metadata={
            "help": " activation function of regression output"
        },
    )
    regression_proj_size: Any = field(
        default=768,
        metadata={"help": "inner dimensions of regression"})
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )


@ dataclass
class Wav2Vec2SeqSeldAudioFrameClassConfig(Wav2Vec2SeldAudioFrameClassConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassTCNConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer"
                     "model")
        },
    )
    classifier_inner_channels: Tuple[int, ...] = field(
        default=(100, 100),
        metadata={"help": "inner dimensions of TCN classifier"})
    classifier_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in ClassifierHead"
                               "applied to input features"}
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "dropout of dense in ClassifierHead"}
    )
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    classifier_kernel_size: int = field(
        default=5, metadata={"help": "kernel-size in ClassifierHead"}
    )
    classifier_norm_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "norm type used in block"
        },
    )
    classifier_activation_fn: str = field(
        default="relu",
        metadata={
            "help": " activation function of head"
        },
    )
    regression_inner_channels: Tuple[int, ...] = field(
        default=(100, 100),
        metadata={"help": "inner dimensions of TCN regression"})
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in RegressionHead"}
    )
    regression_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in RegressionHead"}
    )
    regression_kernel_size: int = field(
        default=5, metadata={"help": "kernel-size in RegressionHead"}
    )
    regression_norm_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "norm type used in block"
        },
    )
    regression_activation_fn: str = field(
        default="relu",
        metadata={
            "help": " activation function of regression inner"
        },
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in RegressionHead"}
    )


@ dataclass
class Wav2Vec2SeqSeldAudioFrameClassTCNConfig(Wav2Vec2SeldAudioFrameClassTCNConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassLSTMConfig(Wav2Vec2SeldAudioFrameClassConfig):
    classifier_hidden_size: int = field(
        default=768, metadata={"help": "lstm hidden_size in ClassifierHead"}
    )
    regression_hidden_size: int = field(
        default=768, metadata={"help": "lstm hidden_size in RegressionHead"}
    )
    classifier_num_layers: int = field(
        default=2, metadata={"help": "lstm number layers in ClassifierHead"}
    )
    regression_num_layers: int = field(
        default=2, metadata={"help": "lstm number layers in ClassifierHead"}
    )
    classifier_dropout_lstm: float = field(
        default=0.0, metadata={"help": "dropout of lstm in ClassifierHead"}
    )
    regression_dropout_lstm: float = field(
        default=0.0, metadata={"help": "dropout of lstm in RegressionHead"}
    )
    classifier_bidirectional: bool = field(
        default=False, metadata={"help": "whether to use bidirectional"}
    )
    regression_bidirectional: bool = field(
        default=False, metadata={"help": "whether to use bidirectional"}
    )


@ dataclass
class Wav2Vec2SeqSeldAudioFrameClassLSTMConfig(Wav2Vec2SeldAudioFrameClassLSTMConfig):
    pass


@dataclass
class Wav2Vec2SeldAudioFrameClassConformerConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer"
                     "model")
        },
    )
    classifier_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of in ClassifierHead"
                               "applied to input features"}
    )
    classifier_encoder: Wav2Vec2Config = Wav2Vec2Config()
    regression_encoder: Wav2Vec2Config = Wav2Vec2Config()
    target_length: int = field(
        default=11, metadata={"help": "number of targets in classification"}
    )
    regression_input_dropout: float = field(
        default=0.0, metadata={"help": "dropout of proj in ClassifierHead"}
    )
    doa_size: int = field(
        default=3, metadata={"help": "number of DOA in ClassifierHead"}
    )


@ dataclass
class Wav2Vec2SeqSeldAudioFrameClassConformerConfig(Wav2Vec2SeldAudioFrameClassConformerConfig):
    pass


@dataclass
class Wav2Vec2SeldAccDoaAudioFrameClassConfig(Wav2Vec2SeldConfig):
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": ("number of layers to freeze in the pretrained transformer"
                     "model")
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
        metadata={
            "help": " activation function of pooler"
        },
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


@ dataclass
class Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig(Wav2Vec2SeldAccDoaAudioFrameClassConfig):
    pass


@register_model("wav2vec2_class", dataclass=Wav2Vec2SeldSeqClassConfig)
class Wav2vecSeqClass(Wav2VecCtc):
    def __init__(self, cfg: Wav2Vec2SeldSeqClassConfig,
                 w2v_encoder: BaseFairseqModel):
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


@register_model("wav2vec2_seld_sequence_class",
                dataclass=Wav2Vec2SeqSeldSequeceClassConfig)
class Wav2vec2SeqSeldSequenceClassEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldSequeceClassConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldSequeceClassConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldSequenceClassEncoder(cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model("wav2vec2_seld_audio_frame_class",
                dataclass=Wav2Vec2SeqSeldAudioFrameClassConfig)
class Wav2vec2SeqSeldAudioFrameClassEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldAudioFrameClassConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAudioFrameClassConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassEncoder(
            cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model("wav2vec2_seld_audio_frame_class_tcn",
                dataclass=Wav2Vec2SeqSeldAudioFrameClassTCNConfig)
class Wav2vec2SeqSeldAudioFrameClassTCNEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldAudioFrameClassTCNConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAudioFrameClassTCNConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassTCNEncoder(
            cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model("wav2vec2_seld_audio_frame_class_lstm",
                dataclass=Wav2Vec2SeqSeldAudioFrameClassLSTMConfig)
class Wav2vec2SeqSeldAudioFrameClassLSTMEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldAudioFrameClassLSTMConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAudioFrameClassLSTMConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassLSTMEncoder(
            cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model("wav2vec2_seld_audio_frame_class_conformer",
                dataclass=Wav2Vec2SeqSeldAudioFrameClassConformerConfig)
class Wav2vec2SeqSeldAudioFrameClassConformerEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldAudioFrameClassConformerConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAudioFrameClassConformerConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAudioFrameClassConformerEncoder(
            cfg, cfg.target_length)
        return cls(cfg, w2v_encoder)


@register_model("wav2vec2_seld_accdoa_audio_frame_class",
                dataclass=Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig)
class Wav2vec2SeqSeldAccDoaAudioFrameClassEncoder(Wav2vecSeqClass):
    def __init__(self, cfg: Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig,
                 w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2SeqSeldAccDoaAudioFrameClassConfig,
                    task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vec2SeldAccDoaAudioFrameClassEncoder(
            cfg, cfg.target_length)
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
                layers.append([nn.Linear(in_dim, dim),
                              get_activation_fn(activation_fn),
                              nn.Dropout(p=dropout)])
                in_dim = dim
            layers = sum(layers, [])
            self.dense = nn.Sequential(*layers)
            self.activation_fn = None

        if isinstance(inner_dims, int):
            if inner_dims == 0:
                self.out_proj = torch.nn.Linear(input_dim, num_outs)
                self.out_activation_fn = utils.get_activation_fn(
                    out_activation_fn)
            else:
                self.out_proj = torch.nn.Linear(inner_dims, num_outs)
                self.out_activation_fn = utils.get_activation_fn(
                    out_activation_fn)
        else:
            self.out_proj = torch.nn.Linear(inner_dims[-1], num_outs)
            self.out_activation_fn = utils.get_activation_fn(out_activation_fn)

    def forward(self, features, **kwargs):

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
    Head for audioframe classification tasks. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        inner_channels,
        num_outs,
        dropout_input,
        dropout,
        kernel_size,
        mode,
        activation_fn,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)

        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=inner_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            mode=mode,
            activation_fn=activation_fn)

        self.out_proj = torch.nn.Linear(inner_channels[-1], num_outs)

    def forward(self, features, **kwargs):

        x = self.dropout_input(features)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x)
        return x


class Wav2vec2AudioFrameClassLSTMHead(nn.Module):
    """
    Head for audioframe classification tasks. It does not have the pooler that
    computes the mean of embeddings along the timesteps

    It produces outputs of size (B, T, N)

    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        inner_dim,
        num_layers,
        num_outs,
        activation_fn,
        dropout_input,
        dropout_lstm,
        dropout,
        bidirectional,
    ):
        super().__init__()

        tgt_dict = DummyDictionary(vocab_size=num_outs)

        self.lstm_enc = LSTMEncoder3D(
            dictionary=tgt_dict,
            embed_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_in=dropout_input,
            dropout_out=dropout_lstm,
            bidirectional=bidirectional,
            left_pad=False,
            pretrained_embed=nn.Identity(),  # disable embeddings
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

    def forward(self, features, padding_mask):

        if padding_mask is None:
            padding_mask = torch.zeros(
                (features.size(0), features.size(1)),
                device=features.device).type(torch.bool)

        input_lengths, _ = (1 - padding_mask.long()
                            ).sum(-1).sort(descending=True)

        x, _, _ = self.lstm_enc(features, src_lengths=input_lengths)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.dense:
            x = self.dense(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.out_proj(x)
        return x


class ConformerFrameHead(nn.Module):
    """Head for sentence-level classification tasks using ConformerEncoder.
    """

    def __init__(
        self,
        cfg,
        num_outs,
        dropout_input,
    ):
        super().__init__()

        self.dropout_input = nn.Dropout(p=dropout_input)

        self.transf_enc = ConformerEncoder(cfg)

        self.out_proj = torch.nn.Linear(cfg.encoder_embed_dim, num_outs)

    def forward(self, features, padding_mask, **kwargs):

        x = self.dropout_input(features)

        x, _ = self.transf_enc(x, padding_mask=padding_mask, layer=None)

        x = self.out_proj(x)
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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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

        class_logits = self.classifier_head(x)
        regression_logits = self.regression_head(x)

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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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

        self.classifier_head = Wav2vec2AudioFrameClassTCNHead(
            input_dim=d,
            inner_channels=cfg.classifier_inner_channels,
            num_outs=tgt_len,
            dropout_input=cfg.classifier_input_dropout,
            dropout=cfg.classifier_dropout,
            kernel_size=cfg.classifier_kernel_size,
            activation_fn=cfg.classifier_activation_fn,
            mode=cfg.classifier_norm_mode,
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassTCNHead(
            input_dim=d,
            inner_channels=cfg.regression_inner_channels,
            num_outs=tgt_len * cfg.doa_size,
            dropout_input=cfg.regression_input_dropout,
            dropout=cfg.regression_dropout,
            kernel_size=cfg.regression_kernel_size,
            activation_fn=cfg.regression_activation_fn,
            mode=cfg.regression_norm_mode,
        )

        for p in self.regression_head.parameters():
            p.param_group = "regression_head"


class Wav2vec2SeldAudioFrameClassLSTMEncoder(Wav2vec2SequenceClassEncoder):
    """
    Similar to Wav2Vec2ForAudioFrameClassification
    https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification

    Wav2Vec2 Model with a frame classification head on top for tasks like
    Speaker Diarization.
    """

    def __init__(self, cfg: Wav2Vec2SeldAudioFrameClassLSTMConfig, tgt_len=1):

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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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

        self.classifier_head = Wav2vec2AudioFrameClassLSTMHead(
            input_dim=d,
            hidden_size=cfg.classifier_hidden_size,
            inner_dim=cfg.classifier_proj_size,
            num_layers=cfg.classifier_num_layers,
            num_outs=cfg.target_length,
            activation_fn=cfg.classifier_activation_fn,
            dropout_input=cfg.classifier_input_dropout,
            dropout_lstm=cfg.classifier_dropout_lstm,
            dropout=cfg.classifier_dropout,
            bidirectional=cfg.classifier_bidirectional,
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = Wav2vec2AudioFrameClassLSTMHead(
            input_dim=d,
            hidden_size=cfg.regression_hidden_size,
            inner_dim=cfg.regression_proj_size,
            num_layers=cfg.regression_num_layers,
            num_outs=cfg.target_length*cfg.doa_size,
            activation_fn=cfg.regression_activation_fn,
            dropout_input=cfg.regression_input_dropout,
            dropout_lstm=cfg.regression_dropout_lstm,
            dropout=cfg.regression_dropout,
            bidirectional=cfg.regression_bidirectional,
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

        class_logits = self.classifier_head(x, padding_mask)
        regression_logits = self.regression_head(x, padding_mask)

        return {
            "class_encoder_out": class_logits,  # B x T x N
            "regression_out": regression_logits,  # B x T x doa_size*N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,  # B x T
        }


class Wav2vec2SeldAudioFrameClassConformerEncoder(Wav2vec2SequenceClassEncoder):
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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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

        # d = w2v_args.model.encoder_embed_dim

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

        self.classifier_head = ConformerFrameHead(
            cfg=cfg.classifier_encoder,
            num_outs=tgt_len,
            dropout_input=cfg.classifier_input_dropout,
        )

        for p in self.classifier_head.parameters():
            p.param_group = "classifier_head"

        self.regression_head = ConformerFrameHead(
            cfg=cfg.regression_encoder,
            num_outs=tgt_len * cfg.doa_size,
            dropout_input=cfg.regression_input_dropout,
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

        class_logits = self.classifier_head(x, padding_mask)
        regression_logits = self.regression_head(x, padding_mask)

        return {
            "class_encoder_out": class_logits,  # B x T x N
            "regression_out": regression_logits,  # B x T x doa_size*N
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,  # B x T
        }


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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides)
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
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args)

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
