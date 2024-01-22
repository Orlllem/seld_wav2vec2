import logging
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConformerEncoder,
    TransformerEncoder,
    Wav2Vec2Config,
    Wav2Vec2Model,
)
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    TransposeLast,
)
from fairseq.utils import is_xla_tensor
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


@dataclass
class Wav2Vec2ChSpecConfig(Wav2Vec2ChConfig):
    spectrogram_1d: bool = II("task.spectrogram_1d")
    n_mels: int = II("task.n_mels")


@dataclass
class Wav2Vec2ChSpecMlmConfig(Wav2Vec2ChSpecConfig):
    num_constrastive_layers: int = field(
        default=6, metadata={"help": "num encoder layers for constrastive learning"}
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


@register_model("wav2vec2_spec_ch", dataclass=Wav2Vec2ChSpecConfig)
class Wav2Vec2ChSpecModel(Wav2Vec2ChModel):
    def __init__(self, cfg: Wav2Vec2ChSpecConfig):
        BaseFairseqModel.__init__(self)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        if cfg.spectrogram_1d:
            self.embed = feature_enc_layers[-1][0]
        else:
            self.embed = feature_enc_layers[-1][0] * \
                self._get_feat_extract_output_lengths(
                    torch.tensor(cfg.n_mels)).tolist()

        if cfg.spectrogram_1d:
            self.feature_extractor = ConvFeatureExtractionChModel(
                in_channels=cfg.in_channels,
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                conv_groups=cfg.in_conv_groups,
            )
        else:
            self.feature_extractor = Conv2DFeatureExtractionChModel(
                in_channels=cfg.in_channels,
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                conv_groups=cfg.in_conv_groups,
            )

        self.layer_norm = LayerNorm(self.embed)

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        encoder_cls = TransformerEncoder
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder

        self.encoder = encoder_cls(cfg)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)


@register_model("wav2vec2_ch_spec_mlm", dataclass=Wav2Vec2ChSpecMlmConfig)
class Wav2Vec2ChSpecMlmModel(Wav2Vec2ChModel):
    def __init__(self, cfg: Wav2Vec2ChSpecMlmConfig):
        BaseFairseqModel.__init__(self)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        if cfg.spectrogram_1d:
            self.embed = feature_enc_layers[-1][0]
        else:
            self.embed = feature_enc_layers[-1][0] * \
                self._get_feat_extract_output_lengths(
                    torch.tensor(cfg.n_mels)).tolist()

        if cfg.spectrogram_1d:
            self.feature_extractor = ConvFeatureExtractionChModel(
                in_channels=cfg.in_channels,
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                conv_groups=cfg.in_conv_groups,
            )
        else:
            self.feature_extractor = Conv2DFeatureExtractionChModel(
                in_channels=cfg.in_channels,
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                conv_groups=cfg.in_conv_groups,
            )

        self.layer_norm = LayerNorm(self.embed)

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.num_constrastive_layers = cfg.num_constrastive_layers

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        encoder_cls = TransformerEncoder
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder

        # constrastive learning (cl) and mlm encoder
        self.encoder = encoder_cls(cfg)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.mlm_proj = nn.Linear(
            cfg.encoder_embed_dim, cfg.latent_vars*cfg.latent_groups)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(
                        padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (
                1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        # mlm transformer output
        x_mlm, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x_mlm,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        assert len(
            layer_results) == self.cfg.encoder_layers, f"length of layer_results {len(layer_results)}"
        # get the output of the N-1 layer
        x = layer_results[self.cfg.num_constrastive_layers -
                          1][2].transpose(0, 1)

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=True)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                # labels of mlm
                targets = q["targets"]

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=True)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                # labels of mlm
                targets = q["targets"]

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        # cl context vectors
        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        # mlm context vectors
        mlm_logits = self.mlm_proj(x_mlm[mask_indices])

        result = {
            "x": x,
            "mlm_logits": mlm_logits,
            "targets": targets,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def get_mlm_targets(self, net_output):
        return net_output["targets"].view(-1).long()

    def get_mlm_logits(self, net_output):
        return net_output["mlm_logits"].view(-1, self.cfg.latent_vars).float()


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


@dataclass
class Wav2Vec2ChSpecPretConfig(Wav2Vec2ChPretConfig):
    spectrogram_1d: bool = II("task.spectrogram_1d")
    n_mels: int = II("task.n_mels")


@register_model("wav2vec2_spec_ch_pretr", dataclass=Wav2Vec2ChSpecPretConfig)
class Wav2Vec2ChSpecModelPtrained(Wav2Vec2ChSpecModel):
    def __init__(self, cfg: Wav2Vec2ChSpecPretConfig):
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2ChSpecPretConfig, task=None):

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
        if cfg.normalize != model_normalized:
            logger.info("Fine-tuning works best when data normalization is the same. "
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


@dataclass
class Wav2Vec2ChSpecMlmPretConfig(Wav2Vec2ChSpecMlmConfig):
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


@register_model("wav2vec2_ch_spec_mlm_pretr", dataclass=Wav2Vec2ChSpecMlmPretConfig)
class Wav2Vec2ChSpecMlmModelPtrained(Wav2Vec2ChSpecMlmModel):
    def __init__(self, cfg: Wav2Vec2ChSpecMlmPretConfig):
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2ChSpecMlmPretConfig, task=None):

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


class Conv2DFeatureExtractionChModel(nn.Module):
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
                conv = nn.Conv2d(n_in, n_out, k, stride=stride, bias=conv_bias,
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

        # (B, C, T, D) -> (B, C, D, T)
        x = x.transpose(2, 3)

        # reshape (B, C, D, T) -> (B, C*D, T)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

        return x
