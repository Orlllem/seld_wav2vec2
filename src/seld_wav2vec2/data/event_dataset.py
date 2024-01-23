

import logging

import numpy as np
import pyarrow
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset, RawAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel
from spafe.fbanks import gammatone_fbanks
from torchaudio.transforms import (
    AmplitudeToDB,
    FrequencyMasking,
    MelScale,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)

logger = logging.getLogger(__name__)

eps = torch.finfo(torch.float32).eps


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
                assert self.max_sample_size is not None, \
                    "max_sample_size must be defined"
                target_size = self.max_sample_size
            else:
                target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources),
                                                sources[0].shape[0],
                                                target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape[0],
                             collated_sources.shape[2]).fill_(
                False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full(
                        (sources[0].shape[0], -diff), 0.0)],
                    dim=1
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source,
                                                            target_size)

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
                input["source"] = self._bucket_tensor(
                    collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(
                    padding_mask, num_pad, True)

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
            raise Exception(
                f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                if self.norm_per_channel:
                    feats = F.layer_norm(feats, (feats.shape[-1], ))
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
        spec_normalize=False,
        spectrogram_1d=True,
        num_buckets=0,
        audio_transforms=None,
        random_crop=False,
        n_mels=64,
        hop_len_s=0.005,  # 5ms
        spec_augment=False,
        time_mask_F=21,
        time_mask_T=20,
        gammatone_filter_banks=False,
    ):
        RawAudioDataset.__init__(self,
                                 sample_rate=sample_rate,
                                 max_sample_size=max_sample_size,
                                 min_sample_size=min_sample_size,
                                 shuffle=shuffle,
                                 pad=pad,
                                 normalize=normalize
                                 )

        self.raw_audio_input = raw_audio_input

        self.hop_len = int(hop_len_s*self.sample_rate)
        self.win_len = 2 * self.hop_len
        self.n_fft = _next_greater_power_of_2(self.win_len)

        if self.raw_audio_input:
            self.spec_transform = Spectrogram(n_fft=self.n_fft,
                                              win_length=self.win_len,
                                              hop_length=self.hop_len,
                                              power=None,
                                              pad_mode='constant')

            self.melscale_transform = MelScale(sample_rate=self.sample_rate,
                                               mel_scale='slaney',
                                               n_mels=n_mels,
                                               norm=None,
                                               n_stft=self.n_fft // 2 + 1)

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

        self.norm_per_channel = norm_per_channel
        self.random_crop = random_crop
        self.pad_max = pad_max

        self.audio_transforms = audio_transforms

        if spec_augment:
            logger.info(f"Using spec-augment:\n"
                        f"hop_len: {self.hop_len}\n"
                        f"time_mask_F: {time_mask_F},\n"
                        f"time_mask_T: {time_mask_T}")

            self.spec_aug_transform = torch.nn.Sequential(
                TimeStretch(hop_length=self.hop_len,
                            fixed_rate=1.0),
                FrequencyMasking(freq_mask_param=time_mask_F),
                TimeMasking(time_mask_param=time_mask_T),
            )
        else:
            self.spec_aug_transform = None

        if self.raw_audio_input and gammatone_filter_banks:
            fb = gammatone_fbanks.gammatone_filter_banks(
                nfilts=n_mels, nfft=self.n_fft, fs=self.sample_rate, scale='descendant')[0].T
            self.melscale_transform.fb = torch.from_numpy(fb).float()

        if self.raw_audio_input:
            self.mel_wts = self.melscale_transform.fb

    def length_spectrogram(self, wav):
        return int(np.floor(wav/(self.win_len - self.hop_len)) + 1)

    def _get_foa_intensity_vectors(self, linear_spectra):
        IVx = torch.real(torch.conj(
            linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = torch.real(torch.conj(
            linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = torch.real(torch.conj(
            linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])

        normal = torch.sqrt(IVx**2 + IVy**2 + IVz**2) + eps
        IVx = torch.mm(IVx / normal, self.mel_wts)
        IVy = torch.mm(IVy / normal, self.mel_wts)
        IVz = torch.mm(IVz / normal, self.mel_wts)

        # we are doing the following instead of simply concatenating to keep
        # the processing similar to mel_spec and gcc
        foa_iv = torch.stack((IVx, IVy, IVz), dim=-1)
        if self.spectrogram_1d:
            foa_iv = foa_iv.reshape(foa_iv.shape[0], -1)  # (192, T)
        return foa_iv.permute(2, 1, 0)

    def postprocess_spec(self, feats):

        assert feats.dtype == torch.float32, feats.dtype

        if self.spec_normalize:
            with torch.no_grad():
                if self.norm_per_channel:
                    feats = F.layer_norm(feats, (feats.shape[-1], ))
                else:
                    feats = F.layer_norm(feats, feats.shape)
        return feats

    def __getitem__(self, index):

        if self.raw_audio_input:
            samples = {"id": index}
            fn = self.fnames[index]

            fname = f"{self.root_dir}/{fn}"

            wav, curr_sample_rate = sf.read(fname)
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate).T  # (C, T)

            feat = self.extract_feat_spectrogram(feats)
        else:
            samples = {"id": index}
            fn = self.fnames[index]
            path_or_fp = f"{self.root_dir}/{fn}"

            arr = np.load(path_or_fp, mmap_mode="r")  # (7, T, 64)
            feat = torch.from_numpy(arr)

        feat = self.postprocess_spec(feat)

        samples["source"] = feat  # (7, T, 64) or (448, T)

        return samples

    def extract_feat_spectrogram(self, source):

        if self.audio_transforms is not None:
            source = self.audio_transforms(source.unsqueeze(0)).squeeze(0)

        # get spectrogram (4, 513, T)
        spec = self.spec_transform(source)

        # get intensity vector (3, 64, T) or (192, T)
        foa_iv = self._get_foa_intensity_vectors(spec.permute(2, 1, 0))

        # get mel-spectrogram (4, 64, T)
        melscale_spect = self.melscale_transform(spec.abs().pow(2))

        # get logmel-spectrogram (4, 64, T)
        logmel_spec = self.amp2db_transform(melscale_spect)

        # apply SpecAugment method
        if self.spec_aug_transform is not None:
            logmel_spec = self.spec_aug_transform(logmel_spec)

        if self.spectrogram_1d:
            # convert (4, 64, T) to (256, T)
            C, N, T = logmel_spec.shape
            logmel_spec = logmel_spec.reshape(C*N, T)

        # concatenate logmel-spectrogram with foa iv (7, 64, T) or (448, T)
        feat = torch.cat((logmel_spec, foa_iv), dim=0)

        if not self.spectrogram_1d:
            feat = feat.transpose(1, 2)  # (7, 64, T) -> (7, T, 64)

        return feat

    def crop_to_max_size(self, spec, target_size):
        size = spec.shape[1]
        diff = size - target_size
        if diff <= 0:
            return spec

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return spec[:, start:end, :]

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[1] for s in sources]

        if self.pad:
            if self.pad_max:
                assert self.max_sample_size is not None, \
                    "max_sample_size must be defined"
                target_size = self.max_sample_size
            else:
                target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        if self.spectrogram_1d:
            collated_sources = sources[0].new_zeros(len(sources),
                                                    sources[0].shape[0],
                                                    target_size)  # (B, 448, T)
        else:
            # (B, 7, T, 64)
            collated_sources = sources[0].new_zeros(len(sources),
                                                    sources[0].shape[0],
                                                    target_size,
                                                    sources[0].shape[2])

        padding_mask = (
            torch.BoolTensor(collated_sources.shape[0],
                             collated_sources.shape[-2]).fill_(
                False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                if self.spectrogram_1d:
                    collated_sources[i] = torch.cat(
                        [source, source.new_full(
                            (sources[0].shape[0], -diff), 0.0)],
                        dim=1
                    )
                else:
                    collated_sources[i] = torch.cat(
                        [source, source.new_full(
                            (sources[0].shape[0],
                             -diff,
                             sources[0].shape[2]), 0.0)],
                        dim=1
                    )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source,
                                                            target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(
                    collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(
                    padding_mask, num_pad, True)

        out["net_input"] = input
        return out
