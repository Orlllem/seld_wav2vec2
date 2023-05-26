

import random
import warnings
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_audiomentations import Shift
from torch_audiomentations.augmentations.shift import shift_cpu, shift_gpu
from torch_audiomentations.core.transforms_interface import \
    MultichannelAudioNotSupportedException
from torch_audiomentations.utils.multichannel import is_multichannel
from torch_audiomentations.utils.object_dict import ObjectDict


def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * torch.cos(elevation) * torch.cos(azimuth)
    y = r * torch.cos(elevation) * torch.sin(azimuth)
    z = r * torch.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = torch.arctan2(y, x)
    elevation = torch.arctan2(z, torch.sqrt(x**2 + y**2))
    r = torch.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


class RandomSwapChannel(object):
    """
    The data augmentation random swap xyz channel of tfmap of FOA format.
    Adaptation of SALSA (TfmapRandomSwapChannelFoa) to work with torch and raw audio.

    """

    def __init__(self, p: float = 0.5, n_classes: int = 11):
        self.p = p
        self.n_classes = n_classes

    def __call__(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):

        x_new = x.clone()
        y_sed_new = y_sed.clone()
        y_doa_new = y_doa.clone()
        for i in range(x.size(0)):
            if random.random() < self.p:
                x_new[i], y_sed_new[i], y_doa_new[i] = self.apply(
                    x=x[i], y_sed=y_sed[i], y_doa=y_doa[i])

        return x_new, y_sed_new, y_doa_new

    def apply_doa_tf(self, doa_labels, tf):

        x = doa_labels[:, :self.n_classes]
        y = doa_labels[:, self.n_classes:2*self.n_classes]
        z = doa_labels[:, 2*self.n_classes:]

        azi, ele, r = cart2sph(x, y, z)

        if tf == 0:  # azi=-azi-pi/2, ele=ele
            x_new, y_new, z_new = sph2cart(-azi-np.pi/2, ele, r=1)
        elif tf == 1:  # azi=-azi+pi/2, ele=ele
            x_new, y_new, z_new = sph2cart(-azi+np.pi/2, ele, r=1)
        elif tf == 2:  # azi=azi+pi, ele=ele
            x_new, y_new, z_new = sph2cart(azi+np.pi, ele, r=1)
        elif tf == 3:  # azi=azi-pi/2, ele=-ele
            x_new, y_new, z_new = sph2cart(azi-np.pi/2, -ele, r=1)
        elif tf == 4:  # azi=azi+pi/2, ele=-ele
            x_new, y_new, z_new = sph2cart(azi+np.pi/2, -ele, r=1)
        elif tf == 5:  # azi=-azi, ele=-ele
            x_new, y_new, z_new = sph2cart(-azi, -ele, r=1)
        elif tf == 6:  # azi=-azi+pi, ele=-ele
            x_new, y_new, z_new = sph2cart(-azi+np.pi, -ele, r=1)
        else:
            print("only six types of transform are available")

        new_doa_labels = torch.cat([x_new, y_new, z_new], dim=-1)  # (T, 3*N)

        return new_doa_labels

    def apply(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        """
        :param x < np.ndarray (n_channels, n_time_steps)>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, accdoa
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa
        This data augmentation change x_sed and y_doa
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 4, 'invalid input channel: {}'.format(
            n_input_channels)
        x_new = x.clone()
        # random method
        m = np.random.randint(7)  # six type of transformations
        # change input feature
        if m == 0:  # (C1, -C4, C3, -C2)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], -x[3], x[2], -x[1]
        elif m == 1:  # (C1, C4, C3, C2)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], x[3], x[2], x[1]
        elif m == 2:  # (C1, -C2, C3, -C4)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], -x[1], x[2], -x[3]
        elif m == 3:  # (C1, -C4, -C3, C2)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], -x[3], -x[2], x[1]
        elif m == 4:  # (C1, C4, -C3, -C2)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], x[3], -x[2], -x[1]
        elif m == 5:  # (C1, -C2, -C3, C4)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], -x[1], -x[2], x[3]
        elif m == 6:  # (C1, C2, -C3, -C4)
            x_new[0], x_new[1], x_new[2], x_new[3] = x[0], x[1], -x[2], -x[3]
        else:
            print("only seven types of transform are available")

        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            y_doa_new = self.apply_doa_tf(y_doa, m)
        else:
            raise NotImplementedError(
                'only cartesian output format is implemented')

        return x_new, y_sed, y_doa_new


class Compose(object):
    def __init__(self, list_tfs):
        self.list_tfs = list_tfs

    def __call__(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):

        for tf in self.list_tfs:
            x, y_sed, y_doa = tf(x, y_sed, y_doa)
        return x, y_sed, y_doa


class RandomSeqShift(object):
    """
    The data augmentation that shift audio sample.

    """

    def __init__(self, p: float = 0.5, rollover=False, sample_rate=16000):
        self.shift_tf = SeqShift(
            p=p, sample_rate=sample_rate, rollover=rollover, output_type=dict)

    def __call__(self, x: torch.Tensor, y_sed: torch.Tensor, y_doa: torch.Tensor):
        shifted_dict = self.shift_tf(x,
                                     targets=y_sed,
                                     doa_targets=y_doa,
                                     target_rate=50)

        return shifted_dict["samples"], shifted_dict["targets"], shifted_dict["doa_targets"]


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
                    self.__class__.__name__)
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
                f"Targets are not (yet) supported by {self.__class__.__name__}")

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

                    cloned_samples[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.samples

                    if has_targets:
                        cloned_targets[
                            self.transform_parameters["should_apply"]
                        ] = perturbed.targets
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
                target_rate * num_samples_to_shift / sample_rate).to(torch.int32)
            shifted_targets = shift(
                targets.transpose(-2, -1), num_frames_to_shift, self.rollover).transpose(-2, -1)

            shifted_doa_targets = shift(
                doa_targets.transpose(-2, -1), num_frames_to_shift, self.rollover).transpose(-2, -1)

        return ObjectDict(
            samples=shifted_samples,
            sample_rate=sample_rate,
            targets=shifted_targets,
            doa_targets=shifted_doa_targets,
            target_rate=target_rate,
        )
