import json
import logging
import os
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal
import torch
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.logging import metrics
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
from omegaconf import II
from torch import Tensor
from torch_audiomentations import Compose, Gain, ShuffleChannels
from torch_audiomentations.augmentations.splice_out import SpliceOut
from torch_audiomentations.utils.object_dict import ObjectDict

import seld_wav2vec2.criterions.parameter as parameter
from seld_wav2vec2.criterions.ccls_feature_class import FeatureClass
from seld_wav2vec2.criterions.cSELD_evaluation_metrics import SELDMetrics
from seld_wav2vec2.criterions.evaluation_metrics import (
    compute_doa_scores_regr_xyz,
    compute_sed_scores,
    early_stopping_metric,
    er_overall_framewise,
    f1_overall_framewise,
)
from seld_wav2vec2.data import (
    AddTargetSeldAudioFrameClassDataset,
    AddTargetSeldSeqClassDataset,
    FileEventDataset,
    FileEventHdfDataset,
    FileEventSpecDataset,
    FileEventSpecHdfDataset,
)
from seld_wav2vec2.data.transforms import AddColoredNoiseFoa

logger = logging.getLogger(__name__)

# label frame resolution (label_frame_res)
nb_label_frames_1s = 50  # 1/label_hop_len_s = 1/0.02
nb_label_frames_1s_100ms = 10  # 1/label_hop_len_s = 1/0.1

eps = np.finfo(np.float32).eps

DCASE_CHOICES = ChoiceEnum(["2018", "2019", "2020"])


def extract_sliding_windows(x, window_size, stride):
    # x: [B, C, T]
    return x.unfold(dimension=-1, size=window_size, step=stride).transpose(1, 2)


class SpliceOutFilterShort(SpliceOut):
    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        spliceout_samples = []

        for i in range(samples.shape[0]):
            random_lengths = self.transform_parameters["splice_lengths"][i]
            sample = samples[i][:, :]
            for j in range(self.num_time_intervals):
                if sample.shape[-1] - random_lengths[j] > 0:
                    start = torch.randint(
                        0,
                        sample.shape[-1] - random_lengths[j],
                        size=(1,),
                    )

                    if random_lengths[j] % 2 != 0:
                        random_lengths[j] += 1

                    hann_window_len = random_lengths[j]
                    hann_window = torch.hann_window(
                        hann_window_len, device=samples.device
                    )
                    hann_window_left, hann_window_right = (
                        hann_window[: hann_window_len // 2],
                        hann_window[hann_window_len // 2:],
                    )

                    fading_out, fading_in = (
                        sample[:, start: start + random_lengths[j] // 2],
                        sample[
                            :,
                            start + random_lengths[j] // 2: start + random_lengths[j],
                        ],
                    )
                    crossfade = (
                        hann_window_right * fading_out + hann_window_left * fading_in
                    )
                    sample = torch.cat(
                        (
                            sample[:, :start],
                            crossfade[:, :],
                            sample[:, start + random_lengths[j]:],
                        ),
                        dim=-1,
                    )

            padding = torch.zeros(
                (samples[i].shape[0], samples[i].shape[-1] - sample.shape[-1]),
                dtype=torch.float32,
                device=sample.device,
            )
            sample = torch.cat((sample, padding), dim=-1)
            spliceout_samples.append(sample.unsqueeze(0))

        return ObjectDict(
            samples=torch.cat(spliceout_samples, dim=0),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class States(UserDict):
    @property
    def state_dict(self) -> Dict[str, Any]:
        return self.data

    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self.data.update(state_dict)


@dataclass
class SoundEventPretrainingConfig(AudioPretrainingConfig):
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Any = None
    norm_per_channel: bool = field(
        default=False,
        metadata={"help": "Normalize per channel when have multiple channels"},
    )
    spectrogram_input: bool = field(
        default=False,
        metadata={"help": ("Whether to feed the model withspectrogram of raw audio")},
    )
    raw_audio_input: bool = field(
        default=True, metadata={"help": "Whether to use raw audio"}
    )
    use_foaiv: bool = field(
        default=True, metadata={"help": "use foa-iv in spectrogram"}
    )
    spec_normalize: bool = field(
        default=False, metadata={"help": "Normalize the resulting spectrogram"}
    )
    log_spec_normalize: bool = field(
        default=True, metadata={"help": "Normalize spectrogram with log-mel"}
    )
    norm_spec_foaiv: bool = field(
        default=False,
        metadata={"help": "Normalize the foa_iv"},
    )
    audio_augm: bool = field(
        default=False, metadata={"help": "Apply data augmentation on the 3D audio"}
    )
    params_augm: Tuple[float, int, int, float] = field(
        default=(0.5, 8, 400, 0.5),
        metadata={
            "help": (
                "Data audio augmentation parameters:"
                "ShuffleChannels"
                "The default parameters are:"
                "shuffle prob: 0.5",
                "spliceout num_time_intervals: 8",
                "spliceout max_width: 400",
                "spliceout prob: 0.5",
            )
        },
    )
    audio_augm_mode: str = field(
        default="per_example", metadata={"help": "Audio augmentation mode"}
    )
    random_crop: bool = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    non_reverberant_prob: float = field(
        default=0.7,
        metadata={"help": "percentage of non-reverberant samples"},
    )
    num_overlap_append: int = field(
        default=2,
        metadata={"help": "maximum number of overlap sounds to append to sample"},
    )
    in_channels: int = II("model.in_channels")
    n_mels: int = field(
        default=64,
        metadata={"help": "number of mel filters when using spectrogram"},
    )
    hop_len_s: float = field(
        default=0.005,
        metadata={"help": "hop length of STFT"},
    )
    spectrogram_1d: bool = field(
        default=False, metadata={"help": ("Use spectrogram as 1d conv or 2d")}
    )
    gammatone_filter_banks: bool = field(
        default=False, metadata={"help": ("Use gammatone_filter_banks or not")}
    )
    persistent_workers: bool = field(
        default=True, metadata={"help": ("Use persistent_workers or not")}
    )


@register_task("sound_event_pretraining", dataclass=SoundEventPretrainingConfig)
class SoundEventPretrainingTask(AudioPretrainingTask):
    cfg: SoundEventPretrainingConfig

    def __init__(
        self,
        cfg: SoundEventPretrainingConfig,
    ):
        super().__init__(cfg)

    def load_dataset(
        self, split: str, task_cfg: SoundEventPretrainingConfig = None, **kwargs
    ):
        task_cfg = task_cfg or self.cfg

        data_path = self.cfg.data

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        if split == "valid" or split == "test":
            shuffle = False
        else:
            shuffle = True

        audio_transforms = None
        if self.cfg.audio_augm:
            assert all(p >= 0 for p in self.cfg.params_augm), (
                "all params_augm must be positive or zero"
            )

            transforms = [
                ShuffleChannels(
                    p=self.cfg.params_augm[0],
                    mode=self.cfg.audio_augm_mode,
                    sample_rate=self.cfg.sample_rate,
                ),
                SpliceOutFilterShort(
                    num_time_intervals=int(self.cfg.params_augm[1]),
                    max_width=int(self.cfg.params_augm[2]),
                    mode=self.cfg.audio_augm_mode,
                    sample_rate=self.cfg.sample_rate,
                    p=self.cfg.params_augm[3],
                ),
            ]

            audio_transforms = Compose(transforms=transforms)

            logger.info(
                f"Using data-augmentation: \n"
                f"mode: {self.cfg.audio_augm_mode}\n"
                f"ShuffleChannels: p: {self.cfg.params_augm[0]}\n"
                "SpliceOut: num_time_intervals:"
                f"{int(self.cfg.params_augm[1])}\n"
                "SpliceOut: max_width:"
                f"{int(self.cfg.params_augm[2])}\n"
                f"SpliceOut: p: {self.cfg.params_augm[3]}\n"
            )

        if self.cfg.spectrogram_input:
            self.datasets[split] = FileEventSpecDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                jit=False,
                raw_audio_input=self.cfg.raw_audio_input,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                shuffle=shuffle,
                pad=task_cfg.enable_padding,
                pad_max=False,
                normalize=task_cfg.normalize,
                spec_normalize=self.cfg.spec_normalize,
                log_spec_normalize=self.cfg.log_spec_normalize,
                norm_spec_foaiv=self.cfg.norm_spec_foaiv,
                use_foaiv=self.cfg.use_foaiv,
                spectrogram_1d=self.cfg.spectrogram_1d,
                norm_per_channel=self.cfg.norm_per_channel,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                audio_transforms=audio_transforms,
                random_crop=self.cfg.random_crop,
                non_reverberant_prob=self.cfg.non_reverberant_prob,
                num_overlap_append=self.cfg.num_overlap_append,
                spec_augment=False,
                n_mels=self.cfg.n_mels,
                hop_len_s=self.cfg.hop_len_s,
                gammatone_filter_banks=self.cfg.gammatone_filter_banks,
            )
        else:
            self.datasets[split] = FileEventDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                pad_max=False,
                normalize=task_cfg.normalize,
                norm_per_channel=self.cfg.norm_per_channel,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                text_compression_level=text_compression_level,
                audio_transforms=audio_transforms,
                random_crop=self.cfg.random_crop,
                non_reverberant_prob=self.cfg.non_reverberant_prob,
                num_overlap_append=self.cfg.num_overlap_append,
            )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.amp.autocast(
                "cuda", enabled=(isinstance(optimizer, AMPOptimizer))
            ):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


@dataclass
class DoaDiscretizerConfig(FairseqDataclass):
    flag: bool = field(
        default=False,
        metadata={"help": "enable the doa discretization"},
    )
    strategy: str = field(
        default="uniform",
        metadata={"help": "strategy type"},
    )
    n_bins: int = II("model.n_bins")


@dataclass
class SoundEventFinetuningConfig(SoundEventPretrainingConfig):
    audio_augm_valid: bool = field(
        default=False,
        metadata={"help": ("Apply audio data augmentation to valid set")},
    )
    rnd_crop_valid: bool = field(
        default=True,
        metadata={"help": "apply random crop to valid set"},
    )
    padding_max: bool = field(
        default=False, metadata={"help": "pad shorter samples to max_sample_size"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq"
            " models); adds 'prev_output_tokens' to input and appends eos to"
            " target"
        },
    )
    seld_audio_frame_class: bool = field(
        default=True, metadata={"help": "use multi-task seld sequence"}
    )
    nb_classes: int = II("model.target_length")
    params_augm: Tuple[float, float, float, float, float] = field(
        default=(5.0, 0.0, 3, 30, 0.3),
        metadata={
            "help": (
                "Data audio augmentation parameters:"
                "Gain, AddColoredNoise"
                "The default parameters are:"
                "gain_in_db: 5.0"
                "gain prob: 0.0"
                "min_snr_in_db: 3"
                "max_snr_in_db: 30"
                "noise prob: 0.3"
            )
        },
    )
    doa_swap_prob: float = field(
        default=0.0,
        metadata={"help": "prob parameter in swap doa augment"},
    )
    shift_prob: float = field(
        default=0.0,
        metadata={"help": "shit-prob parameter in data augment"},
    )
    shift_rollover: bool = field(default=True, metadata={"help": "rollover of shift"})
    eval_seld_score: bool = field(
        default=True, metadata={"help": "evaluate the model with seld_score"}
    )
    optimize_threshold: bool = field(
        default=True, metadata={"help": "optimize threshold during validation"}
    )
    doa_size: int = II("model.doa_size")
    label_hop_len_s: float = field(
        default=0.02,
        metadata={"help": "Label hop length in seconds"},
    )
    eval_dcase: DCASE_CHOICES = field(
        default="2019", metadata={"help": "DCASE competition"}
    )
    hdf_dataset: bool = field(
        default=True,
        metadata={"help": "use hdf dataset in fine-tuning"},
    )
    use_cache_hdf: bool = field(
        default=False,
        metadata={"help": "use hdf cache in fine-tuning"},
    )
    cache_hdf_size: int = field(
        default=64,
        metadata={"help": "hdf cache size in fine-tuning"},
    )
    segment_eval: bool = field(
        default=False,
        metadata={"help": "segment the eval datasets"},
    )
    avg_seld_score: bool = field(
        default=False,
        metadata={"help": "average the seld score"},
    )
    segment_window: float = field(
        default=2.97,
        metadata={"help": "segment window size in seconds"},
    )
    segment_stride: float = field(
        default=1.0,
        metadata={"help": "segment stride size in seconds"},
    )
    opt_threshold_range: Tuple[float, float, float] = field(
        default=(0.1, 1.0, 0.025),
        metadata={"help": ("threshold range: min, max, step")},
    )
    spec_augment: bool = field(
        default=False, metadata={"help": "augment of log-mel spectrogram"}
    )
    augment_foaiv: bool = field(default=False, metadata={"help": "augment of foa-iv"})
    time_mask_F: int = field(
        default=21,
        metadata={"help": "f mask parameter in spec data augment"},
    )
    time_mask_T: int = field(
        default=20,
        metadata={"help": "t mask parameter in spec data augment"},
    )
    n_time_masks: int = field(
        default=1,
        metadata={"help": "number of t mask parameter in spec data augment"},
    )
    n_freq_masks: int = field(
        default=1,
        metadata={"help": "number of freq mask parameter in spec data augment"},
    )
    iid_masks: bool = field(
        default=False,
        metadata={"help": "iid mask parameter in spec data augment"},
    )
    mask_prob: float = field(
        default=1.0,
        metadata={"help": "prob mask parameter in spec data augment"},
    )
    zero_masking: bool = field(
        default=True,
        metadata={"help": "zero-mask mask parameter in spec data augment"},
    )
    doa_discretizer: DoaDiscretizerConfig = DoaDiscretizerConfig()
    apply_sliding_window_overlap: bool = field(
        default=False,
        metadata={"help": "apply sliding window overlap to the dataset"},
    )
    per_task_sliding_window_overlap: bool = field(
        default=False,
        metadata={"help": "apply sliding window overlap per task"},
    )
    apply_record_inference_on_train: bool = field(
        default=False,
        metadata={"help": "apply record inference on train step"},
    )
    align_outputs_frames: bool = II("model.align_outputs_frames")
    align_only_inference: bool = II("model.align_only_inference")


@register_task("sound_event_finetuning", dataclass=SoundEventFinetuningConfig)
class SoundEventFinetuningTask(SoundEventPretrainingTask):
    cfg: SoundEventFinetuningConfig

    def __init__(
        self,
        cfg: SoundEventFinetuningConfig,
    ):
        super().__init__(cfg)

        if self.cfg.eval_dcase == "2020":
            params = parameter.get_params()

            unique_classes = {}
            for i in range(self.cfg.nb_classes):
                unique_classes[i] = i

            params["unique_classes"] = unique_classes
            params["fs"] = self.cfg.sample_rate
            params["label_hop_len_s"] = self.cfg.label_hop_len_s

            self.feat_cls = FeatureClass(params)
            self.cls_new_metric = SELDMetrics(nb_classes=self.cfg.nb_classes)

        self.state = States()
        self.state["best_threshold"] = 0.5
        self.state["best_score"] = None
        self.valid_update = 0

        self.class_probs_list = []
        self.class_labels_list = []
        self.reg_logits_list = []
        self.reg_targets_list = []

        if self.cfg.get("doa_discretizer", None) is None:
            if self.cfg.doa_discretizer.flag:
                assert self.cfg.doa_discretizer.n_bins > 1, (
                    "the number of bins should be greater than 1"
                )
            else:
                if "n_bins" in self.cfg.doa_discretizer:
                    assert self.cfg.doa_discretizer.n_bins == 1, (
                        "n_bins should be 1 in this scenario"
                    )
                else:
                    self.cfg.doa_discretizer["n_bins"] = 1
                self.state["doa_discretizer_tf"] = None

        if self.cfg.avg_seld_score:
            assert self.cfg.segment_eval is False, (
                "when using avg_seld_score, segment_eval should be False"
            )

        if self.cfg.apply_sliding_window_overlap:
            assert self.cfg.segment_eval is False, (
                "when using sliding window overlap, segment_eval should be False"
            )

        if self.cfg.segment_eval is False:
            assert self.cfg.max_sample_size is None, (
                "when segment_eval is False, max_sample_size should be None"
            )

    def load_dataset(
        self, split: str, task_cfg: SoundEventFinetuningConfig = None, **kwargs
    ):
        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None

        data_path = self.cfg.data

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        if split == "valid" or split == "test":
            if self.cfg.audio_augm and self.cfg.audio_augm_valid:
                audio_augm = True
            else:
                audio_augm = False

            if self.cfg.random_crop and self.cfg.rnd_crop_valid:
                random_crop = True
            else:
                random_crop = False

            doa_swap_augment = False
            shift_augment = False
            shuffle = False
            spec_augment = False
            if self.cfg.get("align_only_inference", False):
                assert self.cfg.get("align_outputs_frames", False), (
                    "align_outputs_frames should be True when align_only_inference is True"
                )

            align_outputs_frames = self.cfg.get("align_outputs_frames", False)
            labels_ref = task_cfg.labels
        else:
            audio_augm = self.cfg.audio_augm
            random_crop = self.cfg.random_crop
            shuffle = True
            spec_augment = self.cfg.spec_augment

            if self.cfg.doa_swap_prob > 0.0:
                doa_swap_augment = True
            else:
                doa_swap_augment = False

            if self.cfg.shift_prob > 0.0:
                shift_augment = True
            else:
                shift_augment = False

            if self.cfg.get("align_only_inference", False):
                # training with frames at ~20ms, but inference is at 100ms
                align_outputs_frames = False
                labels_ref = "labels"  # ~20ms
            else:
                labels_ref = task_cfg.labels
                align_outputs_frames = self.cfg.get("align_outputs_frames", False)

        audio_transforms = None
        if audio_augm:
            assert all(p >= 0 for p in self.cfg.params_augm), (
                "all params_augm must be positive or zero"
            )

            transforms = [
                Gain(
                    min_gain_in_db=-self.cfg.params_augm[0],
                    max_gain_in_db=self.cfg.params_augm[0],
                    p=self.cfg.params_augm[1],
                    sample_rate=self.cfg.sample_rate,
                    mode="per_example",
                ),
                AddColoredNoiseFoa(
                    min_snr_in_db=self.cfg.params_augm[2],
                    max_snr_in_db=self.cfg.params_augm[3],
                    min_f_decay=0.0,
                    max_f_decay=1.0,
                    p=self.cfg.params_augm[4],
                    sample_rate=self.cfg.sample_rate,
                    mode="per_example",
                ),
            ]

            audio_transforms = Compose(transforms=transforms)

            logger.info(
                f"Using data-augmentation:\n"
                f"mode: {self.cfg.audio_augm_mode}\n"
                "Gain: min-max gain_in_db:"
                f"{self.cfg.params_augm[0]},\n"
                f"p: {self.cfg.params_augm[1]}\n"
                f"AddColoredNoise: min_snr_in_db:"
                f"{self.cfg.params_augm[2]},\n"
                f"max_snr_in_db: {self.cfg.params_augm[3]},\n"
                f"p: {self.cfg.params_augm[4]}"
            )

        if self.cfg.spectrogram_input:
            if self.cfg.hdf_dataset:
                self.datasets[split] = FileEventSpecHdfDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    raw_audio_input=self.cfg.raw_audio_input,
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    shuffle=shuffle,
                    pad=task_cfg.enable_padding,
                    pad_max=False,
                    normalize=task_cfg.normalize,
                    spec_normalize=self.cfg.spec_normalize,
                    norm_spec_foaiv=self.cfg.norm_spec_foaiv,
                    use_foaiv=self.cfg.use_foaiv,
                    spectrogram_1d=self.cfg.spectrogram_1d,
                    norm_per_channel=self.cfg.norm_per_channel,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    audio_transforms=audio_transforms,
                    random_crop=self.cfg.random_crop,
                    spec_augment=spec_augment,
                    augment_foaiv=self.cfg.augment_foaiv,
                    time_mask_F=self.cfg.time_mask_F,
                    time_mask_T=self.cfg.time_mask_T,
                    n_freq_masks=self.cfg.n_freq_masks,
                    n_time_masks=self.cfg.n_time_masks,
                    iid_masks=self.cfg.iid_masks,
                    mask_prob=self.cfg.mask_prob,
                    zero_masking=self.cfg.zero_masking,
                    n_mels=self.cfg.n_mels,
                    hop_len_s=self.cfg.hop_len_s,
                )
            else:
                self.datasets[split] = FileEventSpecDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    raw_audio_input=self.cfg.raw_audio_input,
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    shuffle=shuffle,
                    pad=task_cfg.enable_padding,
                    pad_max=False,
                    normalize=task_cfg.normalize,
                    spec_normalize=self.cfg.spec_normalize,
                    norm_spec_foaiv=self.cfg.norm_spec_foaiv,
                    use_foaiv=self.cfg.use_foaiv,
                    spectrogram_1d=self.cfg.spectrogram_1d,
                    norm_per_channel=self.cfg.norm_per_channel,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    audio_transforms=audio_transforms,
                    random_crop=self.cfg.random_crop,
                    spec_augment=spec_augment,
                    augment_foaiv=self.cfg.augment_foaiv,
                    time_mask_F=self.cfg.time_mask_F,
                    time_mask_T=self.cfg.time_mask_T,
                    n_freq_masks=self.cfg.n_freq_masks,
                    n_time_masks=self.cfg.n_time_masks,
                    iid_masks=self.cfg.iid_masks,
                    mask_prob=self.cfg.mask_prob,
                    zero_masking=self.cfg.zero_masking,
                    n_mels=self.cfg.n_mels,
                    hop_len_s=self.cfg.hop_len_s,
                )
        else:
            if self.cfg.hdf_dataset:
                self.datasets[split] = FileEventHdfDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    shuffle=shuffle,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    pad_max=self.cfg.padding_max,
                    normalize=task_cfg.normalize,
                    norm_per_channel=self.cfg.norm_per_channel,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    text_compression_level=text_compression_level,
                    audio_transforms=audio_transforms,
                    random_crop=random_crop,
                    use_cache_hdf=self.cfg.use_cache_hdf,
                    cache_size=self.cfg.cache_hdf_size,
                    align_outputs_frames=align_outputs_frames,
                    label_hop_len_s=self.cfg.label_hop_len_s,
                )
            else:
                self.datasets[split] = FileEventDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    shuffle=shuffle,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    pad_max=self.cfg.padding_max,
                    normalize=task_cfg.normalize,
                    norm_per_channel=self.cfg.norm_per_channel,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    text_compression_level=text_compression_level,
                    audio_transforms=audio_transforms,
                    random_crop=random_crop,
                    align_outputs_frames=align_outputs_frames,
                    label_hop_len_s=self.cfg.label_hop_len_s,
                )

        if self.cfg.hdf_dataset:
            labels, doa_discretizer_tf = self.datasets[split].segment_and_get_labels(
                window=int(self.cfg.segment_window * self.cfg.sample_rate),
                stride=int(self.cfg.segment_stride * self.cfg.sample_rate),
                segment_eval=self.cfg.segment_eval,
                split=split,
                doa_discretizer=self.cfg.doa_discretizer
                if self.cfg.doa_discretizer.flag
                else None,
                doa_size=self.cfg.doa_size,
                labels_ref=labels_ref,
                apply_record_inference_on_train=self.cfg.apply_record_inference_on_train,
            )
            self.state["doa_discretizer_tf"] = doa_discretizer_tf
        else:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")

            with open(label_path, "r") as f:
                labels = json.load(f)

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        if self.cfg.seld_audio_frame_class:
            self.datasets[split] = AddTargetSeldAudioFrameClassDataset(
                self.datasets[split],
                labels,
                doa_swap_augment=doa_swap_augment,
                doa_swap_prob=self.cfg.doa_swap_prob,
                shift_augment=shift_augment,
                shift_prob=self.cfg.shift_prob,
                shift_rollover=self.cfg.shift_rollover,
                n_classes=self.cfg.nb_classes,
                spectrogram_input=self.cfg.spectrogram_input,
                spectrogram_1d=self.cfg.spectrogram_1d,
            )
        else:
            self.datasets[split] = AddTargetSeldSeqClassDataset(
                self.datasets[split],
                labels,
                nb_classes=self.cfg.nb_classes,
            )

    def length_spectrogram(self, size):
        """
        Calculate the length of the spectrogram based on the input size.
        Args:
            size (int): The size of the input audio signal.
        Returns:
            int: The length of the spectrogram.
        """
        dset = "valid" if "valid" in self.datasets else "test"
        sz = self.datasets[dset].dataset.length_spectrogram(size)
        return sz

    def sliding_window_overlap_per_task(
        self,
        source: torch.Tensor,           # [1, C, T_samples]
        padding_mask: torch.Tensor,     # [1, T_samples]
        model,
        num_classes: int = 14,
    ):
        """
        Sliding window inference with adaptive per-task weighting (Hann for class, Gaussian for DOA),
        optimized for batch size = 1.
        """
        assert source.shape[0] == 1, "Only batch size = 1 is supported"
        device = source.device
        _, C, T = source.shape

        window = int(self.cfg.segment_window * self.cfg.sample_rate)
        stride = int(self.cfg.segment_stride * self.cfg.sample_rate)
        total_frames = self.length_spectrogram(T)
        doa_size = self.cfg.doa_size

        # Unfold into overlapping chunks
        chunks = source.unfold(-1, window, stride)  # [1, C, N, window]
        chunks = chunks.transpose(1, 2).squeeze(0)  # [N, C, window]
        chunk_masks = padding_mask.unfold(1, window, stride).squeeze(0)  # [N, window]
        N = chunks.shape[0]

        out = model(source=chunks, padding_mask=chunk_masks)
        class_preds = out["class_encoder_out"]  # [N, F, num_classes]
        reg_preds = out["regression_out"]       # [N, F, num_classes * doa_size]
        _, F, _ = class_preds.shape

        # Create weights
        weight_class = torch.hann_window(F, periodic=False, device=device).view(1, F, 1)
        weight_reg = torch.tensor(
            scipy.signal.windows.gaussian(F, std=F // 6),
            dtype=torch.float32, device=device
        ).view(1, F, 1)

        weight_class = weight_class.expand(N, -1, num_classes)
        weight_reg = weight_reg.expand(N, -1, num_classes * doa_size)

        weighted_class = class_preds * weight_class
        weighted_reg = reg_preds * weight_reg

        frame_starts = torch.arange(N, device=device) * self.length_spectrogram(stride)
        frame_offsets = torch.arange(F, device=device).unsqueeze(0)  # [1, F]
        frame_indices = (frame_starts.unsqueeze(1) + frame_offsets).reshape(-1)  # [N*F]

        # Flatten and stitch
        flat_class = weighted_class.reshape(-1, num_classes)
        flat_reg = weighted_reg.reshape(-1, num_classes * doa_size)
        flat_weight_class = weight_class.reshape(-1, num_classes)
        flat_weight_reg = weight_reg.reshape(-1, num_classes * doa_size)

        preds_class_sum = torch.zeros((1, total_frames, num_classes), device=device)
        preds_reg_sum = torch.zeros((1, total_frames, num_classes * doa_size), device=device)
        preds_count_class = torch.zeros((1, total_frames, num_classes), device=device)
        preds_count_reg = torch.zeros((1, total_frames, num_classes * doa_size), device=device)

        preds_class_sum[0].index_add_(0, frame_indices, flat_class)
        preds_reg_sum[0].index_add_(0, frame_indices, flat_reg)
        preds_count_class[0].index_add_(0, frame_indices, flat_weight_class)
        preds_count_reg[0].index_add_(0, frame_indices, flat_weight_reg)

        preds_count_class = preds_count_class.clamp(min=1e-6)
        preds_count_reg = preds_count_reg.clamp(min=1e-6)

        avg_preds_class = preds_class_sum / preds_count_class
        avg_preds_reg = preds_reg_sum / preds_count_reg

        return {
            "class_encoder_out": avg_preds_class,  # [1, T', num_classes]
            "regression_out": avg_preds_reg,       # [1, T', num_classes * doa_size]
        }

    def sliding_window_overlap(
        self,
        source: torch.Tensor,  # [1, C, T]
        padding_mask: torch.Tensor,  # [1, T]
        model,
        num_classes: int = 14,
    ):
        """
        Batched sliding window inference using overlapping windows.

        Args:
            source: Tensor [1, C=4, T]
            padding_mask: Tensor [1, T]
            model: callable model with output keys "class_encoder_out", "regression_out"
            num_classes: number of class labels
        """
        device = source.device
        B, C, T = source.shape
        assert B == 1, "Only batch size 1 is supported."

        window = int(self.cfg.segment_window * self.cfg.sample_rate)
        stride = int(self.cfg.segment_stride * self.cfg.sample_rate)
        total_frames = self.length_spectrogram(T)
        doa_size = self.cfg.doa_size

        # Unfold into overlapping chunks
        chunks = source.unfold(-1, window, stride)         # [1, C, N, window]
        chunks = chunks.transpose(1, 2).reshape(-1, C, window)  # [N, C, window]
        chunk_masks = padding_mask.unsqueeze(1).unfold(-1, window, stride).reshape(-1, window)  # [N, window]

        # Inference on all chunks
        out = model(source=chunks, padding_mask=chunk_masks)
        class_preds = out["class_encoder_out"]  # [N, F, num_classes]
        reg_preds = out["regression_out"]  # [N, F, num_classes * doa_size]

        num_chunks, frames_per_chunk, _ = class_preds.shape

        # Compute frame indices for stitching
        frame_starts = (
            torch.arange(num_chunks, device=device) * self.length_spectrogram(stride)
        ).unsqueeze(1)  # [N, 1]
        frame_offsets = torch.arange(frames_per_chunk, device=device).unsqueeze(0)  # [1, F]
        target_indices = (frame_starts + frame_offsets).reshape(-1)  # [N * F]

        # Flatten predictions
        flat_class = class_preds.reshape(-1, num_classes)  # [N * F, C]
        flat_reg = reg_preds.reshape(-1, num_classes * doa_size)  # [N * F, C * d]
        flat_ones = torch.ones((flat_class.shape[0], 1), device=device)  # [N * F, 1]

        # Preallocate buffers
        preds_class_sum = torch.zeros((1, total_frames, num_classes), device=device)
        preds_reg_sum = torch.zeros((1, total_frames, num_classes * doa_size), device=device)
        preds_count = torch.zeros((1, total_frames, 1), device=device)

        # Stitch predictions using index_add_
        preds_class_sum[0].index_add_(0, target_indices, flat_class)
        preds_reg_sum[0].index_add_(0, target_indices, flat_reg)
        preds_count[0].index_add_(0, target_indices, flat_ones)

        # Normalize by count
        preds_count[preds_count == 0] = 1
        avg_preds_class = preds_class_sum / preds_count
        avg_preds_reg = preds_reg_sum / preds_count

        return {
            "class_encoder_out": avg_preds_class,  # [1, T', C]
            "regression_out": avg_preds_reg,  # [1, T', C * doa_size]
        }

    def inference_overlap_step(self, sample, model):
        """
        Perform inference on a single sample using sliding window overlap.
        Args:
            sample (dict): the mini-batch with bsz = 1. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
        Returns:
            tuple: (net_output, sample)
                - net_output (dict): model output
                - sample (dict): the original sample with additional keys
        """
        source = sample["net_input"]["source"]  # (1, C, T)
        padding_mask = sample["net_input"]["padding_mask"]  # (1, T)

        assert source.shape[0] == 1, (
            f"{source.shape} source batch-size should be 1 in this case"
        )
        assert padding_mask.shape[0] == 1, (
            f"{padding_mask} padding_mask batch-size should be 1"
        )

        if self.cfg.get("per_task_sliding_window_overlap", False):
            net_output = self.sliding_window_overlap_per_task(
                source=source,
                padding_mask=padding_mask,
                model=model,
                num_classes=self.cfg.nb_classes,
            )
        else:
            net_output = self.sliding_window_overlap(
                source=source,
                padding_mask=padding_mask,
                model=model,
                num_classes=self.cfg.nb_classes,
            )

        return net_output, sample

    def inference_non_overlap_step(self, inputs, model):
        """
        Perform inference on a single sample without sliding window overlap.
        Args:
            inputs (dict): the mini-batch with bsz = 1. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
        Returns:
            tuple: (net_output, sample)
                - net_output (dict): model output
                - sample (dict): the original sample with additional keys
        """
        source = inputs["net_input"]["source"]  # (1, C, T)
        padding_mask = inputs["net_input"]["padding_mask"]  # (1, T)

        assert source.shape[0] == 1, (
            f"{source.shape} source batch-size should be 1 in this case"
        )
        assert padding_mask.shape[0] == 1, (
            f"{padding_mask} padding_mask batch-size should be 1"
        )

        T = source.shape[-1]
        source_array = np.arange(T)  # (T)

        net_output = {}
        sample = {}
        sample["ntokens"] = inputs["ntokens"]
        window = int(self.cfg.segment_window * self.cfg.sample_rate)
        sed_labels = inputs["sed_labels"]
        doa_labels = inputs["doa_labels"]
        for i in range(0, T, window):
            source_size = len(source_array[i: i + window])
            source_end = i + source_size
            source_segment = source[:, :, i:source_end]
            padding_mask_segment = padding_mask[:, i:source_end]

            net_output_segment = model(
                source=source_segment, padding_mask=padding_mask_segment
            )
            n_frames = self.length_spectrogram(source_size)

            for k in net_output_segment.keys():
                if net_output_segment[k] is not None:
                    assert net_output_segment[k].shape[1] == n_frames, (
                        f"{net_output_segment[k].shape[1]} != {n_frames}"
                    )
                    if k not in net_output:
                        net_output[k] = net_output_segment[k]
                    else:
                        net_output[k] = torch.cat(
                            (net_output[k], net_output_segment[k]), dim=1
                        )
                else:
                    if k not in net_output:
                        net_output[k] = net_output_segment[k]

            if not self.cfg.get("align_outputs_frames", False):
                # if align_outputs_frames is False, we need to slice the labels
                labels_start = self.length_spectrogram(i)
                if labels_start < 0:
                    labels_start = 0

                labels_end = labels_start + n_frames
                labels_size = sed_labels[:, labels_start:labels_end].shape[1]

                assert labels_size == n_frames, (
                    f"i: {i}, labels_size: {labels_size} != n_frames: {n_frames}, source_size: {source_size}"
                )

                if "sed_labels" not in sample:
                    sample["sed_labels"] = sed_labels[:, labels_start:labels_end]
                else:
                    sample["sed_labels"] = torch.cat(
                        (sample["sed_labels"], sed_labels[:, labels_start:labels_end]),
                        dim=1,
                    )

                if "doa_labels" not in sample:
                    sample["doa_labels"] = doa_labels[:, labels_start:labels_end]
                else:
                    sample["doa_labels"] = torch.cat(
                        (sample["doa_labels"], doa_labels[:, labels_start:labels_end]),
                        dim=1,
                    )

        if self.cfg.get("align_outputs_frames", False):
            sample["sed_labels"] = sed_labels
            sample["doa_labels"] = doa_labels

        return net_output, sample

    def inference_step(self, sample, model):
        """
        Perform inference on a single sample, ie, batch-size = 1.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
        Returns:
            tuple: (net_output, sample)
                - net_output (dict): model output
                - sample (dict): the original sample with additional keys
        """
        if self.cfg.get("apply_sliding_window_overlap", False):
            assert self.cfg.align_outputs_frames, "align_outputs_frames should be True"
            return self.inference_overlap_step(sample, model)
        else:
            return self.inference_non_overlap_step(sample, model)

    def reduce_metrics(self, logging_outputs, criterion):
        """
        Aggregate logging outputs from data parallel training and compute metrics.
        Args:
            logging_outputs (list[dict]): list of dictionaries containing logging outputs
            criterion (~fairseq.criterions.FairseqCriterion): the criterion used for training
        """
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_seld_score and not criterion.training:
            # validation step
            class_probs = np.concatenate(
                [log.get("class_probs", []) for log in logging_outputs], axis=0
            )
            class_labels = np.concatenate(
                [log.get("class_labels", []) for log in logging_outputs], axis=0
            )
            reg_logits = np.concatenate(
                [log.get("reg_logits", []) for log in logging_outputs], axis=0
            )
            reg_targets = np.concatenate(
                [log.get("reg_targets", []) for log in logging_outputs], axis=0
            )
            sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

            # cache batch predictions for validation
            self.class_probs_list.append(class_probs)
            self.class_labels_list.append(class_labels)
            self.reg_logits_list.append(reg_logits)
            self.reg_targets_list.append(reg_targets)

            self.valid_update += sample_size

            finished_valid = self.valid_update == len(self.datasets["valid"])

            # apply when validation finished
            if finished_valid:
                if self.cfg.optimize_threshold:
                    thr_list = [
                        round(float(i), 3)
                        for i in np.arange(*self.cfg.opt_threshold_range)
                    ]

                    if self.cfg.eval_dcase != "2020" and self.cfg.segment_eval:
                        class_probs = np.concatenate(self.class_probs_list, axis=0)
                        class_labels = np.concatenate(self.class_labels_list, axis=0)
                        reg_logits = np.concatenate(self.reg_logits_list, axis=0)
                        reg_targets = np.concatenate(self.reg_targets_list, axis=0)
                    else:
                        self.class_probs_list = tuple(self.class_probs_list)
                        self.class_labels_list = tuple(self.class_labels_list)
                        self.reg_logits_list = tuple(self.reg_logits_list)
                        self.reg_targets_list = tuple(self.reg_targets_list)

                    number_of_samples = len(self.class_labels_list)
                    assert len(self.class_probs_list) == number_of_samples
                    assert len(self.reg_logits_list) == number_of_samples
                    assert len(self.reg_targets_list) == number_of_samples

                    seld_score_list = []
                    for thr_i in thr_list:
                        if self.cfg.eval_dcase == "2020":
                            if self.cfg.avg_seld_score:
                                seld_score_i = []
                            for i in range(number_of_samples):
                                class_probs = self.class_probs_list[i].copy()
                                class_labels = self.class_labels_list[i].copy()
                                reg_logits = self.reg_logits_list[i].copy()
                                reg_targets = self.reg_targets_list[i].copy()

                                class_mask = class_probs > thr_i
                                y_pred_class = class_mask.astype("float32")

                                # ignore padded labels -100
                                class_pad_mask = class_labels < 0
                                class_labels[class_pad_mask] = 0
                                y_true_class = class_labels.astype("float32")

                                class_mask_extended = np.concatenate(
                                    [class_mask] * self.cfg.doa_size, axis=-1
                                )

                                reg_logits[~class_mask_extended] = 0
                                y_pred_reg = reg_logits.astype("float32")
                                y_true_reg = reg_targets.astype("float32")

                                self.eval_seld_score_2020(
                                    y_pred_reg, y_true_reg, y_pred_class, y_true_class
                                )

                                if self.cfg.avg_seld_score:
                                    # average the scores
                                    er, f, de, de_f = (
                                        self.cls_new_metric.compute_seld_scores()
                                    )
                                    seld_score_i.append(
                                        early_stopping_metric([er, f], [de, de_f])
                                    )
                                    self.cls_new_metric.reset_states()

                            if self.cfg.avg_seld_score:
                                assert len(seld_score_i) == number_of_samples, (
                                    f"{len(seld_score_i)} != {number_of_samples}"
                                )
                                # average the scores by number of samples
                                seld_score_i = sum(seld_score_i) / number_of_samples
                            else:
                                er, f, de, de_f = (
                                    self.cls_new_metric.compute_seld_scores()
                                )
                                seld_score_i = early_stopping_metric(
                                    [er, f], [de, de_f]
                                )

                            # clear 2020 seld metrics
                            self.cls_new_metric.reset_states()
                        else:
                            if self.cfg.avg_seld_score:
                                seld_score_i = 0
                                for i in range(number_of_samples):
                                    class_probs = self.class_probs_list[i].copy()
                                    class_labels = self.class_labels_list[i].copy()
                                    reg_logits = self.reg_logits_list[i].copy()
                                    reg_targets = self.reg_targets_list[i].copy()

                                    _, _, _, _, _seld_score = (
                                        self.compute_score_201X_for_thr(
                                            class_probs,
                                            class_labels,
                                            reg_logits,
                                            reg_targets,
                                            thr=thr_i,
                                            eval_dcase=self.cfg.eval_dcase,
                                        )
                                    )
                                    seld_score_i += _seld_score
                                # average the scores by number of samples
                                seld_score_i = seld_score_i / number_of_samples
                            else:
                                _, _, _, _, seld_score_i = (
                                    self.compute_score_201X_for_thr(
                                        class_probs.copy(),
                                        class_labels.copy(),
                                        reg_logits.copy(),
                                        reg_targets.copy(),
                                        thr=thr_i,
                                        eval_dcase=self.cfg.eval_dcase,
                                    )
                                )

                        # append the score for this threshold
                        seld_score_list.append(seld_score_i)

                    seld_score_dict = dict(zip(thr_list, seld_score_list))

                    # obtain the thresold with mininum seld score
                    thr = min(seld_score_dict, key=seld_score_dict.get)

                    # set best threshold
                    min_seld_score = np.min(seld_score_list)
                    if self.best_score:
                        if min_seld_score < self.best_score:
                            self.state["best_threshold"] = thr
                            self.state["best_score"] = min_seld_score
                    else:
                        self.state["best_threshold"] = thr
                        self.state["best_score"] = min_seld_score

                    logger.info(f"optimal threshold: {thr}")
                    logger.info(f"min_seld_score: {min_seld_score}")
                else:
                    thr = 0.5
                    min_seld_score = None

                metrics.log_scalar("prob_threshold", thr, round=3)

                if self.cfg.eval_dcase == "2020":
                    if self.cfg.avg_seld_score:
                        seld_score = []
                        er = []
                        f = []
                        de = []
                        de_f = []
                    for i in range(number_of_samples):
                        class_probs = self.class_probs_list[i]
                        class_labels = self.class_labels_list[i]
                        reg_logits = self.reg_logits_list[i]
                        reg_targets = self.reg_targets_list[i]

                        self.compute_score_2020_for_thr(
                            class_probs, class_labels, reg_logits, reg_targets, thr=thr
                        )

                        if self.cfg.avg_seld_score:
                            # aggregate the scores
                            _er, _f, _de, _de_f = (
                                self.cls_new_metric.compute_seld_scores()
                            )
                            seld_score.append(
                                early_stopping_metric([_er, _f], [_de, _de_f])
                            )
                            er.append(_er)
                            f.append(_f)
                            de.append(_de)
                            de_f.append(_de_f)
                            self.cls_new_metric.reset_states()

                    if self.cfg.avg_seld_score:
                        # check that all lists have the same length
                        assert len(seld_score) == number_of_samples
                        assert len(er) == number_of_samples
                        assert len(f) == number_of_samples
                        assert len(de) == number_of_samples
                        assert len(de_f) == number_of_samples

                        # average the scores per sample
                        seld_score = sum(seld_score) / number_of_samples
                        er = sum(er) / number_of_samples
                        f = sum(f) / number_of_samples
                        de = sum(de) / number_of_samples
                        de_f = sum(de_f) / number_of_samples
                    else:
                        er, f, de, de_f = self.cls_new_metric.compute_seld_scores()
                        seld_score = early_stopping_metric([er, f], [de, de_f])

                    # clear 2020 seld metrics
                    self.cls_new_metric.reset_states()
                else:
                    if not self.cfg.avg_seld_score:
                        er, f, de, de_f, seld_score = self.compute_score_201X_for_thr(
                            class_probs,
                            class_labels,
                            reg_logits,
                            reg_targets,
                            thr=thr,
                            eval_dcase=self.cfg.eval_dcase,
                        )
                    else:
                        er, f, de, de_f, seld_score = 0, 0, 0, 0, 0
                        for i in range(number_of_samples):
                            class_probs = self.class_probs_list[i].copy()
                            class_labels = self.class_labels_list[i].copy()
                            reg_logits = self.reg_logits_list[i].copy()
                            reg_targets = self.reg_targets_list[i].copy()

                            _er, _f, _de, _de_f, _seld_score = (
                                self.compute_score_201X_for_thr(
                                    class_probs,
                                    class_labels,
                                    reg_logits,
                                    reg_targets,
                                    thr=thr,
                                    eval_dcase=self.cfg.eval_dcase,
                                )
                            )
                            seld_score += _seld_score
                            er += _er
                            f += _f
                            de += _de
                            de_f += _de_f

                        seld_score = seld_score / number_of_samples
                        er = er / number_of_samples
                        f = f / number_of_samples
                        de = de / number_of_samples
                        de_f = de_f / number_of_samples

                if min_seld_score is not None:
                    assert seld_score == min_seld_score, (
                        f"{seld_score} != {min_seld_score}"
                    )

                metrics.log_scalar("f1_score", f * 100, round=5)
                metrics.log_scalar("doa_error", de, round=5)
                metrics.log_scalar("frame_recall", de_f * 100, round=5)
                metrics.log_scalar("error_rate", er * 100, round=5)
                metrics.log_scalar("seld_score", seld_score, round=5)

                # reset states
                self.valid_update = 0
                self.class_probs_list = []
                self.class_labels_list = []
                self.reg_logits_list = []
                self.reg_targets_list = []

    @property
    def best_score(self):
        """
        Get the best SELD score.
        Returns:
            float: the best SELD score
        """
        return self.state["best_score"]

    @property
    def best_threshold(self):
        """
        Get the best threshold for SELD score.
        Returns:
            float: the best threshold for SELD score
        """
        return self.state["best_threshold"]

    @property
    def doa_discretizer_tf(self):
        """
        Get the DOA discretizer transformation function.
        Returns:
            function: the DOA discretizer transformation function
        """
        return self.state["doa_discretizer_tf"]

    def compute_score_201X_for_thr(
        self,
        class_probs,
        class_labels,
        reg_logits,
        reg_targets,
        thr=0.5,
        eval_dcase="2019",
    ):
        """
        Compute the SELD score for a given threshold.
        Args:
            class_probs (np.ndarray): predicted class probabilities
            class_labels (np.ndarray): true class labels
            reg_logits (np.ndarray): predicted regression logits
            reg_targets (np.ndarray): true regression targets
            thr (float): threshold for class probabilities
            eval_dcase (str): evaluation DCASE version, either "2018", "2019", or "2020"
        Returns:
            er (float): error rate
            f (float): F1 score
            de (float): DOA error
            de_f (float): frame recall
            seld_score (float): SELD score
        """

        class_mask = class_probs > thr
        y_pred_class = class_mask.astype("float32")

        # ignore padded labels -100
        class_pad_mask = class_labels < 0
        class_labels[class_pad_mask] = 0
        y_true_class = class_labels.astype("float32")

        class_mask_extended = np.concatenate([class_mask] * self.cfg.doa_size, axis=-1)

        reg_logits[~class_mask_extended] = 0
        y_pred_reg = reg_logits.astype("float32")
        y_true_reg = reg_targets.astype("float32")

        if eval_dcase == "2019":
            er, f, de, de_f, seld_score = self.eval_seld_score_2019(
                y_pred_reg, y_true_reg, y_pred_class, y_true_class
            )
        else:
            er, f, de, de_f, seld_score = self.eval_seld_score_2018(
                y_pred_reg, y_true_reg, y_pred_class, y_true_class
            )
        return er, f, de, de_f, seld_score

    def eval_seld_score_2018(self, doa_pred, doa_gt, sed_pred, sed_gt):
        """ Evaluate SELD score for DCASE 2018.
        Args:
            doa_pred (np.ndarray): predicted DOA values
            doa_gt (np.ndarray): ground truth DOA values
            sed_pred (np.ndarray): predicted sound event detection values
            sed_gt (np.ndarray): ground truth sound event detection values
        Returns:
            er (float): error rate
            f (float): F1 score
            doa_err (float): DOA error
            frame_recall (float): frame recall
            seld_score (float): SELD score
        """
        er_metric = compute_doa_scores_regr_xyz(doa_pred, doa_gt, sed_pred, sed_gt)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        _er = er_overall_framewise(sed_pred, sed_gt)
        _f = f1_overall_framewise(sed_pred, sed_gt)
        _seld_scr = early_stopping_metric([_er, _f], [_doa_err, _frame_recall])

        return _er, _f, _doa_err, _frame_recall, _seld_scr

    def eval_seld_score_2019(self, doa_pred, doa_gt, sed_pred, sed_gt):
        """ Evaluate SELD score for DCASE 2019.
        Args:
            doa_pred (np.ndarray): predicted DOA values
            doa_gt (np.ndarray): ground truth DOA values
            sed_pred (np.ndarray): predicted sound event detection values
            sed_gt (np.ndarray): ground truth sound event detection values
        Returns:
            er (float): error rate
            f (float): F1 score
            doa_err (float): DOA error
            frame_recall (float): frame recall
            seld_score (float): SELD score
        """
        er_metric = compute_doa_scores_regr_xyz(doa_pred, doa_gt, sed_pred, sed_gt)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        doa_metric = [_doa_err, _frame_recall]

        sed_metric = compute_sed_scores(sed_pred, sed_gt, nb_label_frames_1s)
        _er = sed_metric[0]
        _f = sed_metric[1]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        return _er, _f, _doa_err, _frame_recall, _seld_scr

    def compute_score_2020_for_thr(
        self, class_probs, class_labels, reg_logits, reg_targets, thr=0.5
    ):
        """
        Compute the SELD score for DCASE 2020.
        Args:
            class_probs (np.ndarray): predicted class probabilities
            class_labels (np.ndarray): true class labels
            reg_logits (np.ndarray): predicted regression logits
            reg_targets (np.ndarray): true regression targets
            thr (float): threshold for class probabilities
        Returns:
            None: this function updates the internal state of the class
        """
        class_mask = class_probs > thr
        y_pred_class = class_mask.astype("float32")

        # ignore padded labels -100
        class_pad_mask = class_labels < 0
        class_labels[class_pad_mask] = 0
        y_true_class = class_labels.astype("float32")

        class_mask_extended = np.concatenate([class_mask] * self.cfg.doa_size, axis=-1)

        reg_logits[~class_mask_extended] = 0
        y_pred_reg = reg_logits.astype("float32")
        y_true_reg = reg_targets.astype("float32")

        self.eval_seld_score_2020(y_pred_reg, y_true_reg, y_pred_class, y_true_class)

    def eval_seld_score_2020(self, doa_pred, doa_gt, sed_pred, sed_gt):
        """
        Evaluate SELD score for DCASE 2020.
        Args:
            doa_pred (np.ndarray): predicted DOA values
            doa_gt (np.ndarray): ground truth DOA values
            sed_pred (np.ndarray): predicted sound event detection values
            sed_gt (np.ndarray): ground truth sound event detection values
        """
        pred_dict = self.feat_cls.regression_label_format_to_output_format(
            sed_pred, doa_pred
        )
        gt_dict = self.feat_cls.regression_label_format_to_output_format(sed_gt, doa_gt)

        pred_blocks_dict = self.feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
        gt_blocks_dict = self.feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

        self.cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.amp.autocast(
                "cuda", enabled=(isinstance(optimizer, AMPOptimizer))
            ):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
