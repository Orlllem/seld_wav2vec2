import random
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from cseld_ambisonics import mono_to_foa_dynamic
from torch import Tensor
from torch_audiomentations import AddColoredNoise, Shift
from torch_audiomentations.augmentations.colored_noise import _gen_noise
from torch_audiomentations.augmentations.shift import shift_cpu, shift_gpu
from torch_audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.multichannel import is_multichannel
from torch_audiomentations.utils.object_dict import ObjectDict
from torchaudio import functional as F

from seld_wav2vec2.data.utils import (
    CONV_FEATURE_LAYERS,
    get_feat_extract_output_lengths,
)


def sph2cart(azimuth, elevation, r):
    """
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    """

    x = r * torch.cos(elevation) * torch.cos(azimuth)
    y = r * torch.cos(elevation) * torch.sin(azimuth)
    z = r * torch.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    """
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    """

    azimuth = torch.arctan2(y, x)
    elevation = torch.arctan2(z, torch.sqrt(x**2 + y**2))
    r = torch.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def select_shift_tf(
    p: float = 0.5, rollover: bool = False, spectrogram_input: bool = False
):
    if spectrogram_input:
        shift_tf = SpecShift(p=p, rollover=rollover)
    else:
        shift_tf = RandomSeqShift(p=p, rollover=rollover)
    return shift_tf


def convert_spec2d_to_spec1d(spec):
    # spec (7, 64, T)
    C, N, T = spec.shape
    spec1d = spec.reshape(C * N, T)
    return spec1d  # spec (448, T)


def convert_spec1d_to_spec2d(spec):
    # spec (448, T)
    T = spec.shape[-1]
    spec2d = spec.reshape(7, 64, T)
    return spec2d  # (7, 64, T)


class RandomSwapChannel(nn.Module):
    """
    The data augmentation random swap xyz channel of tfmap of FOA format.
    Adaptation of SALSA (TfmapRandomSwapChannelFoa) to work with torch and raw audio.

    """

    def __init__(self, p: float = 0.5, n_classes: int = 11):
        super().__init__()
        self.p = p
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        x_new = x.clone()
        y_sed_new = y_sed.clone()
        y_doa_new = y_doa.clone()

        idx = np.where(np.random.rand(x.size(0)) < self.p)[0]

        for i in idx:
            x_new[i], y_sed_new[i], y_doa_new[i] = self.apply(
                x=x[i], y_sed=y_sed[i], y_doa=y_doa[i]
            )

        return x_new, y_sed_new, y_doa_new

    def apply_doa_tf(self, doa_labels, tf):
        x = doa_labels[:, : self.n_classes]
        y = doa_labels[:, self.n_classes: 2 * self.n_classes]
        z = doa_labels[:, 2 * self.n_classes:]

        azi, ele, r = cart2sph(x, y, z)

        if tf == 0:  # azi=-azi-pi/2, ele=ele
            x_new, y_new, z_new = sph2cart(-azi - np.pi / 2, ele, r=1)
        elif tf == 1:  # azi=-azi+pi/2, ele=ele
            x_new, y_new, z_new = sph2cart(-azi + np.pi / 2, ele, r=1)
        elif tf == 2:  # azi=azi+pi, ele=ele
            x_new, y_new, z_new = sph2cart(azi + np.pi, ele, r=1)
        elif tf == 3:  # azi=azi-pi/2, ele=-ele
            x_new, y_new, z_new = sph2cart(azi - np.pi / 2, -ele, r=1)
        elif tf == 4:  # azi=azi+pi/2, ele=-ele
            x_new, y_new, z_new = sph2cart(azi + np.pi / 2, -ele, r=1)
        elif tf == 5:  # azi=-azi, ele=-ele
            x_new, y_new, z_new = sph2cart(-azi, -ele, r=1)
        elif tf == 6:  # azi=-azi+pi, ele=-ele
            x_new, y_new, z_new = sph2cart(-azi + np.pi, -ele, r=1)
        else:
            print("only six types of transform are available")

        new_doa_labels = torch.cat([x_new, y_new, z_new], dim=-1)  # (T, 3*N)

        return new_doa_labels

    def apply_doa_tf_spec(self, y_doa_new, m):
        if m[0] == 1:
            y_doa_new[:, 0: self.n_classes] = y_doa_new[
                :, self.n_classes: 2 * self.n_classes
            ]
            y_doa_new[:, self.n_classes: 2 * self.n_classes] = y_doa_new[
                :, : self.n_classes
            ]
        if m[1] == 1:
            y_doa_new[:, 0: self.n_classes] = -y_doa_new[:, 0: self.n_classes]
        if m[2] == 1:
            y_doa_new[:, self.n_classes: 2 * self.n_classes] = -y_doa_new[
                :, self.n_classes: 2 * self.n_classes
            ]
        if m[3] == 1:
            y_doa_new[:, 2 * self.n_classes:] = -y_doa_new[:, 2 * self.n_classes:]
        return y_doa_new

    def apply(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        """

        - channels == 4
        WYZX
        - channels == 7
        WYZXXYZ

        x: torch.Tensor (C, T)
        y_sed: torch.Tensor (T, N)
        y_doa: torch.Tensor (T, 3*N)

        This data augmentation change x_sed and y_doa
        """
        n_input_channels = x.shape[0]

        x_new = x.clone()
        y_doa_new = y_doa.clone()

        if n_input_channels == 4:
            # random method
            m = np.random.randint(7)  # seven type of transformations
            # change input feature
            if m == 0:  # (C1, -C4, C3, -C2)
                x_new[1], x_new[3] = x[3], -x[1]
            elif m == 1:  # (C1, C4, C3, C2)
                x_new[1], x_new[3] = x[3], x[1]
            elif m == 2:  # (C1, -C2, C3, -C4)
                x_new[1], x_new[3] = x[1], -x[3]
            elif m == 3:  # (C1, -C4, -C3, C2)
                x_new[1], x_new[2], x_new[3] = x[3], -x[2], x[1]
            elif m == 4:  # (C1, C4, -C3, -C2)
                x_new[1], x_new[2], x_new[3] = x[3], -x[2], -x[1]
            elif m == 5:  # (C1, -C2, -C3, C4)
                x_new[1], x_new[2] = x[1], -x[2]
            elif m == 6:  # (C1, C2, -C3, -C4)
                x_new[2], x_new[3] = -x[2], -x[3]
            else:
                print("only seven types of transform are available")

        elif n_input_channels == 7:
            # input -> W Y Z X X Y Z: 7 channels
            # random method
            m = np.random.randint(2, size=(4,))
            # change input feature
            if m[0] == 1:  # random swap x, y
                x_new[1] = x[3]
                x_new[3] = x[1]
                x_new[-3] = x[-2]
                x_new[-2] = x[-3]
            if m[1] == 1:  # negate x
                x_new[-3] = -x_new[-3]
            if m[2] == 1:  # negate y
                x_new[-2] = -x_new[-2]
            if m[3] == 1:  # negate z
                x_new[-1] = -x_new[-1]

        else:
            n_mels = int(n_input_channels / 7)

            # random method
            m = np.random.randint(2, size=(4,))

            # change input feature
            if m[0] == 1:  # random swap x, y
                x_new[1 * n_mels: 2 * n_mels] = x[3 * n_mels: 4 * n_mels]
                x_new[3 * n_mels: 4 * n_mels] = x[1 * n_mels: 2 * n_mels]
                x_new[4 * n_mels: 5 * n_mels] = x[5 * n_mels: 6 * n_mels]
                x_new[5 * n_mels: 6 * n_mels] = x[4 * n_mels: 5 * n_mels]
            if m[1] == 1:  # negate x
                x_new[4 * n_mels: 5 * n_mels] = -x_new[4 * n_mels: 5 * n_mels]
            if m[2] == 1:  # negate y
                x_new[5 * n_mels: 6 * n_mels] = -x_new[5 * n_mels: 6 * n_mels]
            if m[3] == 1:  # negate z
                x_new[6 * n_mels: 7 * n_mels] = -x_new[6 * n_mels: 7 * n_mels]

        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if n_input_channels == 4:
                y_doa_new = self.apply_doa_tf(y_doa_new, m)
            else:
                y_doa_new = self.apply_doa_tf_spec(y_doa_new, m)
        else:
            raise NotImplementedError("only cartesian output format is implemented")

        return x_new, y_sed, y_doa_new


class RandomSeqShift(object):
    """
    The data augmentation that shift audio sample.

    """

    def __init__(self, p: float = 0.5, rollover=False, sample_rate=16000):
        self.shift_tf = SeqShift(
            p=p, rollover=rollover, sample_rate=sample_rate, output_type=dict
        )

    def __call__(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        shifted_dict = self.shift_tf(
            x, targets=y_sed, doa_targets=y_doa, target_rate=50
        )

        return (
            shifted_dict["samples"],
            shifted_dict["targets"],
            shifted_dict["doa_targets"],
        )


class SpecShift(nn.Module):
    """
    The data augmentation random shift of tfmap of spectrogram.
    """

    def __init__(
        self,
        p: float = 0.5,
        rollover: bool = True,
        min_shift: int = -74,
        max_shift: int = 74,
    ):
        super().__init__()
        self.p = p
        self.rollover = rollover
        self.min_shift = min_shift
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        x_new = x.clone()
        y_sed_new = y_sed.clone()
        y_doa_new = y_doa.clone()

        idx = np.where(np.random.rand(x.size(0)) < self.p)[0]

        for i in idx:
            x_new[i], y_sed_new[i], y_doa_new[i] = self.apply(
                spec=x[i],
                sed_label=y_sed[i],
                doa_label=y_doa[i],
                num_samples_to_shift=random.randint(self.min_shift, self.max_shift),
            )

        return x_new, y_sed_new, y_doa_new

    def apply(self, spec, sed_label, doa_label, num_samples_to_shift):
        spec = torch.roll(spec, shifts=num_samples_to_shift, dims=-1)
        sed_label = torch.roll(sed_label, shifts=num_samples_to_shift, dims=0)
        doa_label = torch.roll(doa_label, shifts=num_samples_to_shift, dims=0)

        if not self.rollover:
            if num_samples_to_shift > 0:
                spec[:num_samples_to_shift, :] = 0.0
            elif num_samples_to_shift < 0:
                spec[num_samples_to_shift:, :] = 0.0

        return spec, sed_label, doa_label


class SeqShift(Shift):
    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        doa_targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        if not self.training:
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                doa_targets=doa_targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if not isinstance(samples, Tensor) or len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects three-dimensional input tensors, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        batch_size, num_channels, num_samples = samples.shape

        if batch_size * num_channels * num_samples == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(
                    self.__class__.__name__
                )
            )
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                doa_targets=doa_targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if is_multichannel(samples):
            if num_channels > num_samples:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )

            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None and self.is_sample_rate_required():
            raise RuntimeError("sample_rate is required")

        if targets is None and self.is_target_required():
            raise RuntimeError("targets is required")

        has_targets = targets is not None

        if has_targets and not self.supports_target:
            warnings.warn(
                f"Targets are not (yet) supported by {self.__class__.__name__}"
            )

        if has_targets:
            (
                target_batch_size,
                num_frames,
                num_classes,
            ) = targets.shape

            if target_batch_size != batch_size:
                raise RuntimeError(
                    f"samples ({batch_size}) and target ({target_batch_size}) batch sizes must be equal."
                )

            target_rate = target_rate or self.target_rate
            if target_rate is None:
                if num_frames > 1:
                    target_rate = round(sample_rate * num_frames / num_samples)
                    print("target_rate", target_rate)
                    warnings.warn(
                        f"target_rate is required by {self.__class__.__name__}. "
                        f"It has been automatically inferred from targets shape to {target_rate}. "
                        f"If this is incorrect, you can pass it directly."
                    )
                else:
                    # corner case where num_frames == 1, meaning that the target is for the whole sample,
                    # not frame-based. we arbitrarily set target_rate to 0.
                    target_rate = 0

        if not self.are_parameters_frozen:
            if self.p_mode == "per_example":
                p_sample_size = batch_size

            elif self.p_mode == "per_channel":
                p_sample_size = batch_size * num_channels

            elif self.p_mode == "per_batch":
                p_sample_size = 1

            else:
                raise Exception("Invalid mode")

            self.transform_parameters = {
                "should_apply": self.bernoulli_distribution.sample(
                    sample_shape=(p_sample_size,)
                ).to(torch.bool)
            }

        if self.transform_parameters["should_apply"].any():
            cloned_samples = samples.clone()

            if has_targets:
                cloned_targets = targets.clone()
                cloned_doa_targets = doa_targets.clone()
            else:
                cloned_targets = None
                cloned_doa_targets = None
                selected_targets = None

            if self.p_mode == "per_example":
                selected_samples = cloned_samples[
                    self.transform_parameters["should_apply"]
                ]

                if has_targets:
                    selected_targets = cloned_targets[
                        self.transform_parameters["should_apply"]
                    ]
                    selected_doa_targets = cloned_doa_targets[
                        self.transform_parameters["should_apply"]
                    ]

                if self.mode == "per_example":
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=selected_samples,
                            sample_rate=sample_rate,
                            targets=selected_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        samples=selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        doa_targets=selected_doa_targets,
                        target_rate=target_rate,
                    )

                    cloned_samples[self.transform_parameters["should_apply"]] = (
                        perturbed.samples
                    )

                    if has_targets:
                        cloned_targets[self.transform_parameters["should_apply"]] = (
                            perturbed.targets
                        )
                        cloned_doa_targets[
                            self.transform_parameters["should_apply"]
                        ] = perturbed.doa_targets

                    output = ObjectDict(
                        samples=cloned_samples,
                        sample_rate=perturbed.sample_rate,
                        targets=cloned_targets,
                        doa_targets=cloned_doa_targets,
                        target_rate=perturbed.target_rate,
                    )
                    return output.samples if self.output_type == "tensor" else output

                else:
                    raise Exception("Invalid mode/p_mode combination")

            else:
                raise Exception("Invalid p_mode {}".format(self.p_mode))

        output = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            doa_targets=doa_targets,
            target_rate=target_rate,
        )
        return output.samples if self.output_type == "tensor" else output

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        doa_targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        num_samples_to_shift = self.transform_parameters["num_samples_to_shift"]

        # Select fastest implementation based on device
        shift = shift_gpu if samples.device.type == "cuda" else shift_cpu

        shifted_samples = shift(samples, num_samples_to_shift, self.rollover)

        if targets is None or target_rate == 0:
            shifted_targets = targets

        else:
            num_frames_to_shift = torch.round(
                target_rate * num_samples_to_shift / sample_rate
            ).to(torch.int32)
            shifted_targets = shift(
                targets.transpose(-2, -1), num_frames_to_shift, self.rollover
            ).transpose(-2, -1)

            shifted_doa_targets = shift(
                doa_targets.transpose(-2, -1), num_frames_to_shift, self.rollover
            ).transpose(-2, -1)

        return ObjectDict(
            samples=shifted_samples,
            sample_rate=sample_rate,
            targets=shifted_targets,
            doa_targets=shifted_doa_targets,
            target_rate=target_rate,
        )


class AddColoredNoiseFoa(AddColoredNoise):
    """
    Add colored noises to the FOA input audio.
    """

    def length_spectrogram(self, input_length):
        input_length = torch.tensor(input_length)
        output_length = get_feat_extract_output_lengths(
            CONV_FEATURE_LAYERS, input_length
        ).tolist()
        return output_length

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        noise = torch.stack(
            [
                _gen_noise(
                    self.transform_parameters["f_decay"][i],
                    num_samples,
                    sample_rate,
                    samples.device,
                )
                for i in range(batch_size)
            ]
        )

        noise_foa = []
        for i in range(len(noise)):
            sample_foa = mono_to_foa_dynamic(
                x=noise[i].numpy(),
                n_frames=self.length_spectrogram(num_samples) + 1,
                sample_rate=sample_rate,
                n_positions=random.choice([3, 4, 5, 6]),
            )  # (T, C)
            noise_foa.append(sample_foa)
        noise_foa = torch.from_numpy(np.stack(noise_foa)).transpose(1, 2)  # (B, C, T)

        # (batch_size, num_channels)
        noise_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        noise_scaled = noise_rms.unsqueeze(-1) * noise_foa

        if noise_scaled.shape[-1] < num_samples:
            diff = num_samples - noise_scaled.shape[-1]
            noise_scaled = torch.cat(
                [
                    noise_scaled,
                    noise_scaled.new_full(
                        (noise_scaled.shape[0], noise_scaled.shape[1], diff), 0.0
                    ),
                ],
                dim=-1,
            )

        return ObjectDict(
            samples=samples + noise_scaled,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class SpecAugment(nn.Module):
    r"""Apply time and frequency masking to a spectrogram.
    Args:
        n_time_masks (int): Number of time masks. If its value is zero, no time masking will be applied.
        time_mask_param (int): Maximum possible length of the time mask.
        n_freq_masks (int): Number of frequency masks. If its value is zero, no frequency masking will be applied.
        freq_mask_param (int): Maximum possible length of the frequency mask.
        iid_masks (bool, optional): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D. (Default: ``True``)
        p (float, optional): maximum proportion of time steps that can be masked.
            Must be within range [0.0, 1.0]. (Default: 1.0)
        zero_masking (bool, optional): If ``True``, use 0 as the mask value,
            else use mean of the input tensor. (Default: ``False``)
    """

    __constants__ = [
        "n_time_masks",
        "time_mask_param",
        "n_freq_masks",
        "freq_mask_param",
        "iid_masks",
        "p",
        "zero_masking",
    ]

    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        p: float = 1.0,
        zero_masking: bool = False,
    ) -> None:
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks
        self.p = p
        self.zero_masking = zero_masking

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of shape `(..., freq, time)`.
        Returns:
            Tensor: Masked spectrogram of shape `(..., freq, time)`.
        """
        if self.zero_masking:
            mask_value = 0.0
        else:
            mask_value = specgram.mean()
        time_dim = specgram.dim() - 1
        freq_dim = time_dim - 1

        if specgram.dim() > 2 and self.iid_masks is True:
            specgram = specgram.unsqueeze(0)
            for _ in range(self.n_time_masks):
                specgram = F.mask_along_axis_iid(
                    specgram, self.time_mask_param, mask_value, time_dim + 1, p=self.p
                )
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis_iid(
                    specgram, self.freq_mask_param, mask_value, freq_dim + 1, p=self.p
                )
            specgram = specgram.squeeze(0)
        else:
            for _ in range(self.n_time_masks):
                specgram = F.mask_along_axis(
                    specgram, self.time_mask_param, mask_value, time_dim, p=self.p
                )
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis(
                    specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p
                )

        return specgram
