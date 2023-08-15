import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from fairseq import metrics
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.audio_pretraining import (AudioPretrainingConfig,
                                             AudioPretrainingTask)
from omegaconf import II
from torch import Tensor
from torch_audiomentations import (AddColoredNoise, Compose, Gain,
                                   ShuffleChannels)
from torch_audiomentations.augmentations.spliceout import SpliceOut
from torch_audiomentations.utils.object_dict import ObjectDict

import seld_wav2vec2.criterions.cls_feature_class as cls_feature_class
import seld_wav2vec2.criterions.parameter as parameter
from seld_wav2vec2.criterions.evaluation_metrics import (
    compute_doa_scores_regr_xyz, compute_sed_scores, early_stopping_metric,
    er_overall_framewise, f1_overall_framewise)
from seld_wav2vec2.criterions.SELD_evaluation_metrics import SELDMetrics
from seld_wav2vec2.data import (AddTargetSeldAudioFrameClassDataset,
                                AddTargetSeldSeqClassDataset, FileEventDataset)

logger = logging.getLogger(__name__)

# label frame resolution (label_frame_res)
nb_label_frames_1s = 50  # 1/label_hop_len_s = 1/0.02
nb_label_frames_1s_100ms = 10  # 1/label_hop_len_s = 1/0.1

eps = np.finfo(np.float32).eps

AUDIO_AUGM_MODES_CHOICES = ChoiceEnum(["per_example", "per_channel"])
DCASE_CHOICES = ChoiceEnum(["2018", "2019", "2020"])


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
                        hann_window_len, device=samples.device)
                    hann_window_left, hann_window_right = (
                        hann_window[: hann_window_len // 2],
                        hann_window[hann_window_len // 2:],
                    )

                    fading_out, fading_in = (
                        sample[:, start: start + random_lengths[j] // 2],
                        sample[:, start + random_lengths[j] //
                               2: start + random_lengths[j]],
                    )
                    crossfade = hann_window_right * fading_out + hann_window_left * fading_in
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


@ dataclass
class SoundEventPretrainingConfig(AudioPretrainingConfig):
    norm_per_channel: bool = field(
        default=False, metadata={"help": ("Normalize per channel when have"
                                          "multiple channels")}
    )
    audio_augm: bool = field(
        default=False, metadata={"help": "Apply data augmentation on the 3D "
                                         "audio"}
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
    audio_augm_mode: AUDIO_AUGM_MODES_CHOICES = field(
        default="per_example", metadata={"help": "Audio augmentation mode"}
    )
    random_crop: bool = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    in_channels: int = II("model.in_channels")


@ register_task("sound_event_pretraining",
                dataclass=SoundEventPretrainingConfig)
class SoundEventPretrainingTask(AudioPretrainingTask):
    """ """

    cfg: SoundEventPretrainingConfig

    def __init__(
        self,
        cfg: SoundEventPretrainingConfig,
    ):
        super().__init__(cfg)

    def load_dataset(
        self, split: str, task_cfg: SoundEventPretrainingConfig = None,
        **kwargs
    ):

        task_cfg = task_cfg or self.cfg

        data_path = self.cfg.data

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        audio_transforms = None
        if self.cfg.audio_augm:

            assert all(
                p >= 0 for p in self.cfg.params_augm), \
                "all params_augm must be positive or zero"

            transforms = [
                ShuffleChannels(p=self.cfg.params_augm[0],
                                mode=self.cfg.audio_augm_mode,
                                sample_rate=self.cfg.sample_rate),
                SpliceOutFilterShort(num_time_intervals=int(self.cfg.params_augm[1]),
                                     max_width=int(self.cfg.params_augm[2]),
                                     mode=self.cfg.audio_augm_mode,
                                     sample_rate=self.cfg.sample_rate,
                                     p=self.cfg.params_augm[3])
            ]

            audio_transforms = Compose(transforms=transforms)

            logger.info(f"Using data-augmentation: \n"
                        f"mode: {self.cfg.audio_augm_mode}\n"
                        f"ShuffleChannels: p: {self.cfg.params_augm[0]}\n"
                        "SpliceOut: num_time_intervals:"
                        f"{int(self.cfg.params_augm[1])}\n"
                        "SpliceOut: max_width:"
                        f"{int(self.cfg.params_augm[2])}\n"
                        f"SpliceOut: p: {self.cfg.params_augm[3]}\n")

        self.datasets[split] = FileEventDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get(
                "sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.enable_padding,
            pad_max=False,
            normalize=task_cfg.normalize,
            norm_per_channel=self.cfg.norm_per_channel,
            num_buckets=self.cfg.num_batch_buckets or int(
                self.cfg.tpu),
            compute_mask_indices=(
                self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            audio_transforms=audio_transforms if self.cfg.audio_augm else None,
            params_augm=self.cfg.params_augm,
            random_crop=self.cfg.random_crop,
            **self._get_mask_precompute_kwargs(task_cfg),
        )


@ dataclass
class SoundEventFinetuningConfig(SoundEventPretrainingConfig):
    audio_augm_valid: bool = field(
        default=False, metadata={"help": ("Apply audio data augmentation to"
                                          "valid set")}
    )
    rnd_crop_valid: bool = field(
        default=True,
        metadata={"help": "apply random crop to valid set"},
    )
    padding_max: bool = field(
        default=False, metadata={"help": "pad shorter samples to"
                                 "max_sample_size"}
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
    shift_rollover: bool = field(
        default=True, metadata={"help": "rollover of shift"}
    )
    eval_seld_score: bool = field(
        default=True, metadata={"help": "evaluate the model with seld_score"}
    )
    optimize_threshold: bool = field(
        default=True, metadata={"help": "optimize threshold during validation"}
    )
    doa_size: int = II("model.doa_size")
    label_hop_len_s: float = field(
        default=0.02,
        metadata={
            "help": "Label hop length in seconds"},
    )
    eval_dcase: DCASE_CHOICES = field(
        default="2019", metadata={"help": "DCASE competition"}
    )
    opt_threshold_range: Tuple[float, float, float] = field(
        default=(0.1, 1.0, 0.025),
        metadata={
            "help": (
                "threshold range: min, max, step"
            )
        },
    )


@ register_task("sound_event_finetuning", dataclass=SoundEventFinetuningConfig)
class SoundEventFinetuningTask(SoundEventPretrainingTask):
    """ """

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

            params['unique_classes'] = unique_classes
            params['fs'] = self.cfg.sample_rate
            params['label_hop_len_s'] = self.cfg.label_hop_len_s

            self.feat_cls = cls_feature_class.FeatureClass(params)
            self.cls_new_metric = SELDMetrics(nb_classes=self.cfg.nb_classes)

        self.best_threshold = 0.5
        self.best_score = None
        self.valid_update = 0

        self.class_probs_list = []
        self.class_labels_list = []
        self.reg_logits_list = []
        self.reg_targets_list = []

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
        else:
            audio_augm = self.cfg.audio_augm
            random_crop = self.cfg.random_crop
            shuffle = True

            if self.cfg.doa_swap_prob > 0.0:
                doa_swap_augment = True
            else:
                doa_swap_augment = False

            if self.cfg.shift_prob > 0.0:
                shift_augment = True
            else:
                shift_augment = False

        if audio_augm:
            assert all(
                p >= 0 for p in self.cfg.params_augm), \
                "all params_augm must be positive or zero"

            transforms = [
                Gain(
                    min_gain_in_db=-self.cfg.params_augm[0],
                    max_gain_in_db=self.cfg.params_augm[0],
                    p=self.cfg.params_augm[1],
                    sample_rate=self.cfg.sample_rate,
                ),
                AddColoredNoise(min_snr_in_db=self.cfg.params_augm[2],
                                max_snr_in_db=self.cfg.params_augm[3],
                                min_f_decay=-2.0,
                                max_f_decay=2.0,
                                p=self.cfg.params_augm[4],
                                sample_rate=self.cfg.sample_rate),
            ]

            audio_transforms = Compose(transforms=transforms)

            logger.info(f"Using data-augmentation:\n"
                        f"mode: {self.cfg.audio_augm_mode}\n"
                        "Gain: min-max gain_in_db:"
                        f"{self.cfg.params_augm[0]},\n"
                        f"p: {self.cfg.params_augm[1]}\n"
                        f"AddColoredNoise: min_snr_in_db:"
                        f"{self.cfg.params_augm[2]},\n"
                        f"max_snr_in_db: {self.cfg.params_augm[3]},\n"
                        f"p: {self.cfg.params_augm[4]}")

        self.datasets[split] = FileEventDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get(
                "sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            shuffle=shuffle,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            pad_max=self.cfg.padding_max,
            normalize=task_cfg.normalize,
            norm_per_channel=self.cfg.norm_per_channel,
            num_buckets=self.cfg.num_batch_buckets or int(
                self.cfg.tpu),
            compute_mask_indices=(
                self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            audio_transforms=audio_transforms if audio_augm else None,
            random_crop=random_crop,
            **self._get_mask_precompute_kwargs(task_cfg),
        )

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
            )
        else:
            self.datasets[split] = AddTargetSeldSeqClassDataset(
                self.datasets[split],
                labels,
                nb_classes=self.cfg.nb_classes,
            )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_seld_score and not criterion.training:

            # validation step
            class_probs = np.concatenate([log.get("class_probs", 0)
                                          for log in logging_outputs], axis=0)
            class_labels = np.concatenate([log.get("class_labels", 0)
                                           for log in logging_outputs], axis=0)
            reg_logits = np.concatenate([log.get("reg_logits", 0)
                                         for log in logging_outputs], axis=0)
            reg_targets = np.concatenate([log.get("reg_targets", 0)
                                         for log in logging_outputs], axis=0)
            sample_size = sum(log.get("sample_size", 0)
                              for log in logging_outputs)

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
                    thr_list = [round(float(i), 3)
                                for i in np.arange(*self.cfg.opt_threshold_range)]

                    if self.cfg.eval_dcase != "2020":
                        class_probs = np.concatenate(
                            self.class_probs_list, axis=0)
                        class_labels = np.concatenate(
                            self.class_labels_list, axis=0)
                        reg_logits = np.concatenate(
                            self.reg_logits_list, axis=0)
                        reg_targets = np.concatenate(
                            self.reg_targets_list, axis=0)

                        # class_probs.flags.writeable = False
                        # class_labels.flags.writeable = False
                        # reg_logits.flags.writeable = False
                        # reg_targets.flags.writeable = False
                    else:
                        self.class_probs_list = tuple(self.class_probs_list)
                        self.class_labels_list = tuple(self.class_labels_list)
                        self.reg_logits_list = tuple(self.reg_logits_list)
                        self.reg_targets_list = tuple(self.reg_targets_list)

                    seld_score_list = []
                    for thr_i in thr_list:
                        if self.cfg.eval_dcase == "2020":

                            for i in range(len(self.class_labels_list)):

                                class_probs = self.class_probs_list[i].copy()
                                class_labels = self.class_labels_list[i].copy()
                                reg_logits = self.reg_logits_list[i].copy()
                                reg_targets = self.reg_targets_list[i].copy()

                                class_mask = class_probs > thr_i
                                y_pred_class = class_mask.astype('float32')

                                # ignore padded labels -100
                                class_pad_mask = class_labels < 0
                                class_labels[class_pad_mask] = 0
                                y_true_class = class_labels.astype('float32')

                                class_mask_extended = np.concatenate(
                                    [class_mask]*self.cfg.doa_size, axis=-1)

                                reg_logits[~class_mask_extended] = 0
                                y_pred_reg = reg_logits.astype('float32')
                                y_true_reg = reg_targets.astype('float32')

                                self.eval_seld_score_2020(y_pred_reg,
                                                          y_true_reg,
                                                          y_pred_class,
                                                          y_true_class)

                            er, f, de, de_f = self.cls_new_metric.compute_seld_scores()
                            seld_score_i = early_stopping_metric(
                                [er, f], [de, de_f])

                            seld_score_list.append(seld_score_i)

                            # clear 2020 seld metrics
                            self.cls_new_metric.reset_states()
                        else:

                            seld_score_i = self.compute_score_201X_for_thr(class_probs.copy(),
                                                                           class_labels.copy(),
                                                                           reg_logits.copy(),
                                                                           reg_targets.copy(),
                                                                           thr_i)

                            seld_score_list.append(seld_score_i)

                    seld_score_dict = dict(zip(thr_list, seld_score_list))

                    # obtain the thresold with mininum seld score
                    thr = min(seld_score_dict, key=seld_score_dict.get)

                    # metrics.log_scalar("seld_score_default",
                    #                   seld_score_dict[0.5], round=5)

                    # set best threshold
                    min_seld_score = np.min(seld_score_list)
                    if self.best_score:
                        if min_seld_score < self.best_score:
                            self.best_threshold = thr
                            self.best_score = min_seld_score
                    else:
                        self.best_threshold = thr
                        self.best_score = min_seld_score

                    logger.info(f"optimal threshold: {thr}")
                    logger.info(f"min_seld_score: {min_seld_score}")
                else:
                    thr = 0.5
                    min_seld_score = None

                metrics.log_scalar("prob_threshold", thr, round=3)

                if self.cfg.eval_dcase == "2020":
                    for i in range(len(self.class_labels_list)):

                        class_probs = self.class_probs_list[i]
                        class_labels = self.class_labels_list[i]
                        reg_logits = self.reg_logits_list[i]
                        reg_targets = self.reg_targets_list[i]

                        class_mask = class_probs > thr
                        y_pred_class = class_mask.astype('float32')

                        # ignore padded labels -100
                        class_pad_mask = class_labels < 0
                        class_labels[class_pad_mask] = 0
                        y_true_class = class_labels.astype('float32')

                        class_mask_extended = np.concatenate(
                            [class_mask]*self.cfg.doa_size, axis=-1)

                        reg_logits[~class_mask_extended] = 0
                        y_pred_reg = reg_logits.astype('float32')
                        y_true_reg = reg_targets.astype('float32')

                        self.eval_seld_score_2020(y_pred_reg,
                                                  y_true_reg,
                                                  y_pred_class,
                                                  y_true_class)

                    er, f, de, de_f = self.cls_new_metric.compute_seld_scores()
                    seld_score = early_stopping_metric(
                        [er, f], [de, de_f])

                    # clear 2020 seld metrics
                    self.cls_new_metric.reset_states()
                else:
                    class_mask = class_probs > thr
                    y_pred_class = class_mask.astype('float32')

                    # ignore padded labels -100
                    class_pad_mask = class_labels < 0
                    class_labels[class_pad_mask] = 0
                    y_true_class = class_labels.astype('float32')

                    class_mask_extended = np.concatenate(
                        [class_mask]*self.cfg.doa_size, axis=-1)

                    reg_logits[~class_mask_extended] = 0
                    y_pred_reg = reg_logits.astype('float32')
                    y_true_reg = reg_targets.astype('float32')

                    if self.cfg.eval_dcase == "2019":
                        er, f, de, de_f, seld_score = self.eval_seld_score_2019(y_pred_reg,
                                                                                y_true_reg,
                                                                                y_pred_class,
                                                                                y_true_class)
                    else:
                        # TAU - 2018
                        er, f, de, de_f, seld_score = self.eval_seld_score_2018(y_pred_reg,
                                                                                y_true_reg,
                                                                                y_pred_class,
                                                                                y_true_class)

                if min_seld_score is not None:
                    assert seld_score == min_seld_score, f"{seld_score} != {min_seld_score}"

                metrics.log_scalar("f1_score", f * 100, round=5)
                metrics.log_scalar("doa_error", de, round=5)
                metrics.log_scalar(
                    "frame_recall", de_f*100, round=5)
                metrics.log_scalar(
                    "error_rate", er*100, round=5)
                metrics.log_scalar("seld_score", seld_score, round=5)

                # reset states
                self.valid_update = 0
                self.class_probs_list = []
                self.class_labels_list = []
                self.reg_logits_list = []
                self.reg_targets_list = []

    def compute_score_201X_for_thr(self,
                                   class_probs,
                                   class_labels,
                                   reg_logits,
                                   reg_targets,
                                   thr):

        class_mask = class_probs > thr
        y_pred_class = class_mask.astype('float32')

        # ignore padded labels -100
        class_pad_mask = class_labels < 0
        class_labels[class_pad_mask] = 0
        y_true_class = class_labels.astype('float32')

        class_mask_extended = np.concatenate(
            [class_mask]*self.cfg.doa_size, axis=-1)

        reg_logits[~class_mask_extended] = 0
        y_pred_reg = reg_logits.astype('float32')
        y_true_reg = reg_targets.astype('float32')

        if self.cfg.eval_dcase == "2019":
            _, _, _, _, seld_score = self.eval_seld_score_2019(y_pred_reg,
                                                               y_true_reg,
                                                               y_pred_class,
                                                               y_true_class)
        else:
            _, _, _, _, seld_score = self.eval_seld_score_2018(y_pred_reg,
                                                               y_true_reg,
                                                               y_pred_class,
                                                               y_true_class)
        return seld_score

    def eval_seld_score_2018(self, doa_pred, doa_gt, sed_pred, sed_gt):

        er_metric = compute_doa_scores_regr_xyz(doa_pred, doa_gt,
                                                sed_pred, sed_gt)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        _er = er_overall_framewise(sed_pred, sed_gt)
        _f = f1_overall_framewise(sed_pred, sed_gt)
        _seld_scr = early_stopping_metric(
            [_er, _f], [_doa_err, _frame_recall])

        return _er, _f, _doa_err, _frame_recall, _seld_scr

    def eval_seld_score_2019(self, doa_pred, doa_gt, sed_pred, sed_gt):

        er_metric = compute_doa_scores_regr_xyz(
            doa_pred, doa_gt, sed_pred, sed_gt)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        doa_metric = [_doa_err, _frame_recall]

        sed_metric = compute_sed_scores(sed_pred, sed_gt, nb_label_frames_1s)
        _er = sed_metric[0]
        _f = sed_metric[1]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        return _er, _f, _doa_err, _frame_recall, _seld_scr

    def eval_seld_score_2020(self, doa_pred, doa_gt, sed_pred, sed_gt):

        pred_dict = self.feat_cls.regression_label_format_to_output_format(
            sed_pred, doa_pred
        )
        gt_dict = self.feat_cls.regression_label_format_to_output_format(
            sed_gt, doa_gt
        )

        pred_blocks_dict = self.feat_cls.segment_labels(
            pred_dict, sed_pred.shape[0])
        gt_blocks_dict = self.feat_cls.segment_labels(
            gt_dict, sed_gt.shape[0])

        self.cls_new_metric.update_seld_scores_xyz(
            pred_blocks_dict, gt_blocks_dict)
