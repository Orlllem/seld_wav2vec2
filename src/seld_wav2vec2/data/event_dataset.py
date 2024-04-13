import glob
import logging
import os
import random

import numpy as np
import pyarrow
import scipy
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset, RawAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel
from scipy.interpolate import make_interp_spline
from seld_ambisonics.common import AmbiFormat, AmbisonicArray
from seld_ambisonics.encoder import AmbiEncoder
from seld_ambisonics.position import MovingSource, Position, PositionalSource
from spafe.fbanks import gammatone_fbanks
from torchaudio.transforms import (
    AmplitudeToDB,
    MelScale,
    Spectrogram,
)

from seld_wav2vec2.data.transforms import SpecAugment

logger = logging.getLogger(__name__)

eps = torch.finfo(torch.float32).eps


ROOT_DIR = os.path.dirname(os.path.realpath(__file__)).rsplit(os.sep, 3)[0]

IR_LIST = glob.glob(
    f"{ROOT_DIR}/data/pre-training/IRs/B_format_resampled/*.wav", recursive=True
)

# Phi can vary all around the horizontal plane, so 0 360
# Elevation is common to vary between -60 60, but we can choose different values as well,such as -90 90
# Radius varies from 0 to 10m since is the common approach in sounds localization in close environments
PHI, ELE, Z = np.arange(0, 360, 5), np.arange(-60, 60, 5), np.arange(0, 10, 0.2)

N_POSITIONS = [3, 4, 5, 6, 7]


def mono_to_foa_static(x, n_frames, sample_rate):
    """
    Uses the seld_ambisonics repo to spatialize mono files
    """
    encoder = AmbiEncoder(
        AmbiFormat(ambi_order=1, sample_rate=sample_rate, ordering="ACN")
    )
    # Randomly select a value in the specified range
    coord1 = random.choice(PHI)
    coord2 = random.choice(ELE)
    coord3 = random.choice(Z)
    source = PositionalSource(x, Position(coord1, coord2, coord3, "polar"), sample_rate)
    ambi = encoder.encode(source)
    return ambi.data


def mono_to_foa_dynamic(x, n_frames, sample_rate):
    """
    Uses the seld_ambisonics repo to spatialize mono files with dynamic
    """

    fmt = AmbiFormat(ambi_order=1, sample_rate=sample_rate, ordering="ACN")
    encoder = AmbiEncoder(fmt)

    n_positions = random.choice(N_POSITIONS)

    windows = np.random.dirichlet(np.ones(n_positions), size=1)[0]

    windows = windows * n_frames

    x_positions = []
    positions = []
    p = 0
    for i in range(n_positions):
        # Randomly select a value in the specified range
        coord1 = random.choice(PHI)
        coord2 = random.choice(ELE)
        coord3 = 1.0  # unit radius

        for j in range(round(windows[i])):
            if j > np.random.randint(low=5, high=12):
                positions.append((coord1, coord2, coord3))
                x_positions.append(p)
            p = p + 1

    X_Y_Spline = make_interp_spline(x_positions, positions, k=2)  # order = 2

    _X = np.arange(0, n_frames)

    positions = X_Y_Spline(_X)

    positions = [Position(p[0], p[1], p[2], "polar") for p in positions]

    source = MovingSource(x, positions, rate=sample_rate)

    ambi = AmbisonicArray(np.zeros((x.shape[0], fmt.num_channels)), fmt)
    while source.tic():
        encoder.encode_frame(source, ambi, source.cur_idx)

    return ambi.data


def mono_to_foa_reverb(x, n_frames, sample_rate):
    """
    Uses real Impulse Responses to spatialize mono files
    """
    # Randomly select one of the IRs
    ir_name = random.choice(IR_LIST)
    # Open IR
    ir_input, _ = sf.read(ir_name, dtype="float32")
    # Convert input mono wav into 4 channels to convolve with the 4ch IRs
    wav_4ch = np.repeat(x[:, np.newaxis], 4, axis=1)
    # Convolve the wav signal with the IRs. The mode='same' ensures the output with the same input dimension
    conv = scipy.signal.convolve(wav_4ch, ir_input, mode="same", method="auto")
    return conv


def _next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


class FileEventDataset(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_max=False,
        normalize=False,
        norm_per_channel=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        audio_transforms=None,
        random_crop=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            num_buckets=num_buckets,
            compute_mask_indices=compute_mask_indices,
            text_compression_level=text_compression_level,
            **mask_compute_kwargs,
        )

        self.norm_per_channel = norm_per_channel
        self.random_crop = random_crop
        self.pad_max = pad_max

        self.audio_transforms = audio_transforms

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s[0]) for s in sources]

        if self.pad:
            if self.pad_max:
                assert (
                    self.max_sample_size is not None
                ), "max_sample_size must be defined"
                target_size = self.max_sample_size
            else:
                target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(
            len(sources), sources[0].shape[0], target_size
        )
        padding_mask = (
            torch.BoolTensor(
                collated_sources.shape[0], collated_sources.shape[2]
            ).fill_(False)
            if self.pad
            else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((sources[0].shape[0], -diff), 0.0)], dim=1
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        if self.audio_transforms is not None:
            collated_sources = self.audio_transforms(collated_sources)
            input = {"source": collated_sources}
        else:
            input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def postprocess(self, feats, curr_sample_rate):
        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                if self.norm_per_channel:
                    feats = F.layer_norm(feats, (feats.shape[-1],))
                else:
                    feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = wav.shape[-1]
        diff = size - target_size
        if diff <= 0:
            return wav

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[..., start:end]

    def __getitem__(self, index):
        samples = super().__getitem__(index)

        samples["source"] = samples["source"].T  # (C, T)

        return samples


class FileEventSpecDataset(FileEventDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        raw_audio_input=True,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_max=False,
        normalize=False,
        norm_per_channel=False,
        norm_spec_foaiv=True,
        spec_normalize=True,
        norm_center_scale=False,
        center_scale_spec4ch=[0.0, 1.0],
        center_scale_foaiv=[0.0, 1.0],
        spectrogram_1d=True,
        num_buckets=0,
        audio_transforms=None,
        random_crop=False,
        n_mels=64,
        hop_len_s=0.005,  # 5ms
        spec_augment=False,
        augment_foaiv=True,
        time_mask_F=21,
        time_mask_T=20,
        n_time_masks=1,
        n_freq_masks=1,
        iid_masks=False,
        mask_prob=1.0,
        zero_masking=True,
        gammatone_filter_banks=False,
    ):
        RawAudioDataset.__init__(
            self,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )

        self.raw_audio_input = raw_audio_input

        self.hop_len = int(hop_len_s * self.sample_rate)
        self.win_len = 2 * self.hop_len
        self.n_fft = _next_greater_power_of_2(self.win_len)

        if self.raw_audio_input:
            self.spec_transform = Spectrogram(
                n_fft=self.n_fft,
                win_length=self.win_len,
                hop_length=self.hop_len,
                power=None,
                pad_mode="constant",
            )

            self.melscale_transform = MelScale(
                sample_rate=self.sample_rate,
                mel_scale="slaney",
                n_mels=n_mels,
                norm=None,
                n_stft=self.n_fft // 2 + 1,
            )

            self.amp2db_transform = AmplitudeToDB()

        self.spectrogram_1d = spectrogram_1d
        self.spec_normalize = spec_normalize

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line

                if self.raw_audio_input:
                    # get output size of spectrogram
                    sz = self.length_spectrogram(int(items[1]))
                else:
                    sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.fnames = pyarrow.array(self.fnames)
        self.sizes = np.array(sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)

        self.norm_center_scale = norm_center_scale
        self.center_scale_spec4ch = center_scale_spec4ch
        self.center_scale_foaiv = center_scale_foaiv

        self.norm_spec_foaiv = norm_spec_foaiv
        self.norm_per_channel = norm_per_channel
        self.random_crop = random_crop
        self.pad_max = pad_max

        self.audio_transforms = audio_transforms

        self.augment_foaiv = augment_foaiv
        if spec_augment:
            logger.info(
                f"Using spec-augment:\n"
                f"hop_len: {self.hop_len}\n"
                f"time_mask_F: {time_mask_F},\n"
                f"time_mask_T: {time_mask_T},\n"
                f"n_freq_masks: {n_freq_masks},\n"
                f"n_time_masks: {n_time_masks},\n"
                f"iid_masks: {iid_masks},\n"
                f"mask_prob: {mask_prob}"
            )

            self.spec_aug_transform = SpecAugment(
                n_time_masks=n_time_masks,
                n_freq_masks=n_freq_masks,
                time_mask_param=time_mask_T,
                freq_mask_param=time_mask_F,
                iid_masks=iid_masks,
                p=mask_prob,
                zero_masking=zero_masking,
            )
        else:
            self.spec_aug_transform = None

        if self.raw_audio_input and gammatone_filter_banks:
            fb = gammatone_fbanks.gammatone_filter_banks(
                nfilts=n_mels, nfft=self.n_fft, fs=self.sample_rate, scale="descendant"
            )[0].T
            self.melscale_transform.fb = torch.from_numpy(fb).float()

        if self.raw_audio_input:
            self.mel_wts = self.melscale_transform.fb

    def length_spectrogram(self, wav):
        return int(np.floor(wav / (self.win_len - self.hop_len)) + 1)

    def _get_foa_intensity_vectors(self, linear_spectra):
        IVx = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])

        normal = (
            eps
            + (
                torch.abs(linear_spectra[:, :, 0]) ** 2
                + torch.abs(linear_spectra[:, :, 1]) ** 2
                + torch.abs(linear_spectra[:, :, 2]) ** 2
                + torch.abs(linear_spectra[:, :, 3]) ** 2
            )
            / 2.0
        )

        # normal = torch.sqrt(IVx**2 + IVy**2 + IVz**2) + eps
        IVx = torch.mm(IVx / normal, self.mel_wts)
        IVy = torch.mm(IVy / normal, self.mel_wts)
        IVz = torch.mm(IVz / normal, self.mel_wts)

        # we are doing the following instead of simply concatenating to keep
        # the processing similar to mel_spec and gcc
        foa_iv = torch.stack((IVx, IVy, IVz), dim=-1)  # (T, 64, 3)
        foa_iv = foa_iv.permute(2, 1, 0)  # (3, 64, T)
        return foa_iv

    def postprocess_spec(self, feats, augment=True, foa_iv=False):
        assert feats.dtype == torch.float32, feats.dtype

        if self.spec_normalize:
            with torch.no_grad():
                if self.norm_per_channel:
                    feats = F.layer_norm(feats, (feats.shape[-2], feats.shape[-1]))
                else:
                    if self.norm_center_scale:
                        if foa_iv:
                            feats = (
                                feats - self.center_scale_foaiv[0]
                            ) / self.center_scale_foaiv[1]
                        else:
                            feats = (
                                feats - self.center_scale_spec4ch[0]
                            ) / self.center_scale_spec4ch[1]
                    else:
                        feats = F.layer_norm(feats, feats.shape)

        # apply SpecAugment method
        if (self.spec_aug_transform is not None) and augment:
            feats = self.spec_aug_transform(feats)
        return feats

    def __getitem__(self, index):
        if self.raw_audio_input:
            samples = {"id": index}
            fn = self.fnames[index]

            fname = f"{self.root_dir}/{fn}"

            wav, curr_sample_rate = sf.read(fname)
            if len(wav.shape) == 1 or wav.shape[0] == 1:
                """
                Randomly select one of the functions, mono_to_foa_dynamic which prob of 0.7 and mono_to_foa_reverb 0.3
                The reason of the weighted choices is that there are just 67 IRs each one with one direction, so...
                The mono_to_foa_dynamic generates non-reverberant outputs with much more different directions
                """
                foa_function = random.choices(
                    [mono_to_foa_static, mono_to_foa_reverb], [0.7, 0.3]
                )[0]
                if wav.shape[0] == 1:
                    wav = wav.squeeze(0)
                wav = foa_function(wav, self.length_spectrogram(len(wav)), self.sample_rate)

            assert wav.shape[-1] == 4, wav.shape
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate).T  # (C, T)

            feat = self.extract_feat_spectrogram(feats)
        else:
            samples = {"id": index}
            fn = self.fnames[index]
            path_or_fp = f"{self.root_dir}/{fn}"

            arr = np.load(path_or_fp, mmap_mode="r")  # (7, 64, T)
            feat = torch.from_numpy(arr)

            feat = self.postprocess_spec(feat)

            # apply SpecAugment method
            if self.spec_aug_transform is not None:
                feat = self.spec_aug_transform(feat)

        samples["source"] = feat  # (7, 64, T) or (448, T)

        return samples

    def extract_feat_spectrogram(self, source):
        if self.audio_transforms is not None:
            source = self.audio_transforms(source.unsqueeze(0)).squeeze(0)

        # get spectrogram (4, 513, T)
        spec = self.spec_transform(source)

        # get intensity vector (3, 64, T)
        foa_iv = self._get_foa_intensity_vectors(spec.permute(2, 1, 0))

        if self.norm_spec_foaiv:
            foa_iv = self.postprocess_spec(foa_iv,
                                           augment=self.augment_foaiv,
                                           foa_iv=True)

        # get mel-spectrogram (4, 64, T)
        melscale_spect = self.melscale_transform(spec.abs().pow(2))

        # get logmel-spectrogram (4, 64, T)
        logmel_spec = self.amp2db_transform(melscale_spect)

        # normalize and augment if enabled
        logmel_spec = self.postprocess_spec(logmel_spec, foa_iv=False)

        # concatenate logmel-spectrogram with foa iv (7, 64, T)
        feat = torch.cat((logmel_spec, foa_iv), dim=0)

        # convert (7, 64, T) to (448, T)
        if self.spectrogram_1d:
            C, N, T = feat.shape
            feat = feat.reshape(C * N, T)

        return feat

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[-1] for s in sources]

        if self.pad:
            if self.pad_max:
                assert (
                    self.max_sample_size is not None
                ), "max_sample_size must be defined"
                target_size = self.max_sample_size
            else:
                target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        if self.spectrogram_1d:
            collated_sources = sources[0].new_zeros(
                len(sources), sources[0].shape[0], target_size
            )  # (B, 448, T)
        else:
            collated_sources = sources[0].new_zeros(
                len(sources),
                sources[0].shape[0],
                sources[0].shape[1],
                target_size,
            )  # (B, 7, 64, T)

        padding_mask = (
            torch.BoolTensor(
                collated_sources.shape[0], collated_sources.shape[-1]
            ).fill_(False)
            if self.pad
            else None
        )

        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                if self.spectrogram_1d:
                    collated_sources[i] = torch.cat(
                        [source, source.new_full((sources[0].shape[0], -diff), 0.0)],
                        dim=-1,
                    )
                else:
                    collated_sources[i] = torch.cat(
                        [
                            source,
                            source.new_full(
                                (
                                    sources[0].shape[0],
                                    sources[0].shape[1],
                                    -diff,
                                ),
                                0.0,
                            ),
                        ],
                        dim=-1,
                    )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        out["net_input"] = input
        return out
