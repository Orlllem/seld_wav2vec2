import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
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

from seld_wav2vec2.data import (AddTargetSeldAudioFrameClassDataset,
                                AddTargetSeldSeqClassDataset, FileEventDataset)

logger = logging.getLogger(__name__)


AUDIO_AUGM_MODES_CHOICES = ChoiceEnum(["per_example", "per_channel"])


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


@ register_task("sound_event_finetuning", dataclass=SoundEventFinetuningConfig)
class SoundEventFinetuningTask(SoundEventPretrainingTask):
    """ """

    cfg: SoundEventFinetuningConfig

    def __init__(
        self,
        cfg: SoundEventFinetuningConfig,
    ):
        super().__init__(cfg)

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
