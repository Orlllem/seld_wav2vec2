import glob
import logging
import math
import os
import random

import h5py
import numpy as np
import pyarrow
import scipy
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from cachetools import LRUCache
from cseld_ambisonics import (
    AmbiEncoder,
    AmbiFormat,
    Position,
    PositionalSource,
    mono_to_foa_dynamic_overlap,
)
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset, RawAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
from sklearn.preprocessing import KBinsDiscretizer
from spafe.fbanks import gammatone_fbanks
from torchaudio.transforms import (
    AmplitudeToDB,
    MelScale,
    Spectrogram,
)

from seld_wav2vec2.data.transforms import SpecAugment
from seld_wav2vec2.data.utils import (
    CONV_FEATURE_LAYERS,
    get_feat_extract_output_lengths,
)

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


def crop_to_same_size(arrays, size):
    """
    Crop multiple arrays to the same size.

    Parameters:
        arrays (list of numpy.ndarray): A list of NumPy arrays to crop.
        size (int): The size to crop the arrays to.

    Returns:
        list of numpy.ndarray: The cropped arrays, all of the same size.
    """

    cropped_arrays = []
    for arr in arrays:
        if len(arr) >= size:
            cropped_arrays.append(arr[0:size])
        else:
            # print("arr", arr.shape)
            cropped_arrays.append(np.pad(arr, (0, size - len(arr))))

    return cropped_arrays


def mono_to_foa_static(x, sample_rate):
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


def mono_to_foa_reverb(x, sample_rate):
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


def log10(x):
    return torch.log10(x + 1e-6)


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
        text_compression_level=TextCompressionLevel.none,
        audio_transforms=None,
        random_crop=False,
        non_reverberant_prob=0.7,
        positions_choices=[3, 4, 5, 6],
        num_overlap_append=2,
        align_outputs_frames=False,
        label_hop_len_s=0.02,  # 20ms
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

        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        self.mono_fnames = []
        sizes = []
        ch_sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2 or len(items) == 3, line
                sz = int(items[1])
                if len(items) == 3:
                    ch_sz = int(items[2])
                else:
                    ch_sz = 4
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)

                if len(items) == 3:
                    ch_sizes.append(ch_sz)

                if ch_sz == 1:
                    self.mono_fnames.append(self.text_compressor.compress(items[0]))
        total = len(self.fnames)
        multi = len(self.fnames) - len(self.mono_fnames)
        mono = len(self.mono_fnames)
        logger.info(
            f"loaded total {total}, multi: {multi}, mono {mono}, skipped {skipped} samples"
        )

        self.sizes = np.array(sizes, dtype=np.int64)
        self.ch_sizes = np.array(ch_sizes, dtype=np.int64)
        self.fnames = pyarrow.array(self.fnames)
        self.mono_fnames = pyarrow.array(self.mono_fnames)

        self.set_bucket_info(num_buckets)

        self.norm_per_channel = norm_per_channel
        self.random_crop = random_crop
        self.pad_max = pad_max

        self.audio_transforms = audio_transforms
        self.non_reverberant_prob = non_reverberant_prob
        self.positions_choices = positions_choices
        self.num_overlap_choices = [i for i in range(1, num_overlap_append + 1)]

        self.align_outputs_frames = align_outputs_frames
        self.label_hop_len_s = label_hop_len_s

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s[0]) for s in sources]

        if self.pad:
            if self.pad_max:
                assert self.max_sample_size is not None, (
                    "max_sample_size must be defined"
                )
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

    def length_spectrogram(self, input_length):
        if self.align_outputs_frames:
            output_length = math.ceil(
                input_length / (self.sample_rate * self.label_hop_len_s)
            )
        else:
            input_length = torch.tensor(input_length)
            output_length = get_feat_extract_output_lengths(
                CONV_FEATURE_LAYERS, input_length
            ).tolist()
        return output_length

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

    def decompress_to_filepath(self, fn):
        fn = self.decompress_to_filename(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        return path_or_fp

    def decompress_to_filename(self, fn):
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        return fn

    def get_mono_wav_from_fname(self, fn):
        path_or_fp = self.decompress_to_filepath(fn)
        wav, curr_sample_rate = sf.read(path_or_fp)
        return wav.astype(np.float32)

    def __getitem__(self, index):
        samples = {"id": index}
        fn = self.fnames[index]
        path_or_fp = self.decompress_to_filepath(fn)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
        if len(wav.shape) == 1 or wav.shape[0] == 1:
            """
            Randomly select one of the functions, mono_to_foa_dynamic which prob of 0.7 and mono_to_foa_reverb 0.3
            The reason of the weighted choices is that there are just 67 IRs each one with one direction, so...
            The mono_to_foa_dynamic generates non-reverberant outputs with much more different directions
            """

            if wav.shape[0] == 1:
                wav = wav.squeeze(0)

            if random.random() < self.non_reverberant_prob:
                rand_fnames = random.choices(
                    self.mono_fnames, k=random.choice(self.num_overlap_choices)
                )

                waves = [wav]
                for fn in rand_fnames:
                    waves.append(self.get_mono_wav_from_fname(fn))
                waves = crop_to_same_size(waves, size=wav.shape[0])
                wav = mono_to_foa_dynamic_overlap(
                    waves=waves,
                    n_frames=self.length_spectrogram(len(wav)),
                    sample_rate=self.sample_rate,
                    n_positions=random.choice(self.positions_choices),
                )
            else:
                wav = mono_to_foa_reverb(x=wav, sample_rate=self.sample_rate)

        assert wav.shape[-1] == 4, wav.shape  # (T, C)
        feats = torch.from_numpy(wav).float().T  # (C, T)
        feats = self.postprocess(feats, curr_sample_rate)  # (C, T)

        samples["source"] = feats
        return samples


class FileEventSpecDataset(FileEventDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        jit=True,
        raw_audio_input=True,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_max=False,
        normalize=False,
        spec_normalize=False,
        log_spec_normalize=True,
        norm_per_channel=False,
        use_foaiv=True,
        norm_spec_foaiv=False,
        spectrogram_1d=True,
        num_buckets=0,
        text_compression_level=TextCompressionLevel.none,
        audio_transforms=None,
        random_crop=False,
        n_mels=64,
        hop_len_s=0.005,  # 5ms
        spec_augment=False,
        augment_foaiv=False,
        time_mask_F=21,
        time_mask_T=20,
        n_time_masks=1,
        n_freq_masks=1,
        iid_masks=False,
        mask_prob=1.0,
        zero_masking=True,
        gammatone_filter_banks=False,
        non_reverberant_prob=0.7,
        num_overlap_append=2,
        positions_choices=[3, 4, 5, 6, 7],
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            pad_max=pad_max,
            normalize=normalize,
            norm_per_channel=norm_per_channel,
            num_buckets=num_buckets,
            text_compression_level=text_compression_level,
            audio_transforms=audio_transforms,
            random_crop=random_crop,
            non_reverberant_prob=non_reverberant_prob,
            num_overlap_append=num_overlap_append,
            positions_choices=positions_choices,
        )

        self.raw_audio_input = raw_audio_input
        self.spectrogram_1d = spectrogram_1d

        self.hop_len = int(hop_len_s * self.sample_rate)
        self.win_len = 2 * self.hop_len
        self.n_fft = _next_greater_power_of_2(self.win_len)

        if self.raw_audio_input:
            self.log_mel_spec_transform = LogMelSpectrogram(
                sample_rate=sample_rate,
                jit=jit,
                spectrogram_1d=spectrogram_1d,
                n_mels=n_mels,
                gammatone_filter_banks=gammatone_filter_banks,
                audio_transforms=audio_transforms,
                spec_normalize=spec_normalize,
                log_spec_normalize=log_spec_normalize,
                norm_per_channel=norm_per_channel,
                use_foaiv=use_foaiv,
                norm_spec_foaiv=norm_spec_foaiv,
                augment_foaiv=augment_foaiv,
                spec_augment=spec_augment,
                time_mask_F=time_mask_F,
                time_mask_T=time_mask_T,
                n_time_masks=n_time_masks,
                n_freq_masks=n_freq_masks,
                iid_masks=iid_masks,
                mask_prob=mask_prob,
                zero_masking=zero_masking,
            )

    def length_spectrogram(self, wav):
        return int(np.floor(wav / (self.win_len - self.hop_len)) + 1)

    def __getitem__(self, index):
        if self.raw_audio_input:
            samples = {"id": index}
            fn = self.fnames[index]
            fname = self.decompress_to_filepath(fn)

            wav, curr_sample_rate = sf.read(fname, dtype="float32")
            if len(wav.shape) == 1 or wav.shape[0] == 1:
                """
                Randomly select one of the functions, mono_to_foa_dynamic which prob of 0.7 and mono_to_foa_reverb 0.3
                The reason of the weighted choices is that there are just 67 IRs each one with one direction, so...
                The mono_to_foa_dynamic generates non-reverberant outputs with much more different directions
                """

                if wav.shape[0] == 1:
                    wav = wav.squeeze(0)

                if random.random() < self.non_reverberant_prob:
                    rand_fnames = random.choices(
                        self.mono_fnames, k=random.choice(self.num_overlap_choices)
                    )

                    waves = [wav]
                    for fn in rand_fnames:
                        waves.append(self.get_mono_wav_from_fname(fn))
                    waves = crop_to_same_size(waves, size=wav.shape[0])
                    wav = mono_to_foa_dynamic_overlap(
                        waves=waves,
                        n_frames=self.length_spectrogram(len(wav)),
                        sample_rate=self.sample_rate,
                        n_positions=random.choice(self.positions_choices),
                    )
                else:
                    wav = mono_to_foa_reverb(x=wav, sample_rate=self.sample_rate)

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

        samples["source"] = feat  # (7, 64, T) or (448, T)

        return samples

    def extract_feat_spectrogram(self, source):
        return self.log_mel_spec_transform(source)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[-1] for s in sources]

        if self.pad:
            if self.pad_max:
                assert self.max_sample_size is not None, (
                    "max_sample_size must be defined"
                )
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


class FileEventHdfDataset(FileEventDataset):
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
        text_compression_level=TextCompressionLevel.none,
        audio_transforms=None,
        random_crop=False,
        non_reverberant_prob=0.7,
        positions_choices=[3, 4, 5, 6],
        num_overlap_append=2,
        align_outputs_frames=False,
        label_hop_len_s=0.02,  # 20ms
        use_cache_hdf=False,
        cache_size=64,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            pad_max=pad_max,
            normalize=normalize,
            norm_per_channel=norm_per_channel,
            num_buckets=num_buckets,
            text_compression_level=text_compression_level,
            audio_transforms=audio_transforms,
            random_crop=random_crop,
            non_reverberant_prob=non_reverberant_prob,
            num_overlap_append=num_overlap_append,
            positions_choices=positions_choices,
            align_outputs_frames=align_outputs_frames,
            label_hop_len_s=label_hop_len_s,
        )

        self.use_cache_hdf = use_cache_hdf
        if self.use_cache_hdf:
            self.cache = LRUCache(maxsize=cache_size)

    def segment_and_get_labels(
        self,
        window=47520,
        stride=16000,
        segment_eval=False,
        split="train",
        doa_discretizer=None,
        doa_size=3,
        labels_ref="labels",
        apply_record_inference_on_train=False,
    ):
        self.min_sample_size = (
            self.min_sample_size if self.min_sample_size is not None else 0
        )
        self.segment_eval = segment_eval

        if apply_record_inference_on_train:
            segment = False
        else:
            segment = split == "train" or (split != "train" and self.segment_eval)

        labels = []
        self.fnames_segments = []
        if doa_discretizer is not None:
            doa_labels_train = []
        if segment:
            self.sizes = []
        for fn in self.fnames:
            fn = self.decompress_to_filename(fn)
            path_or_fp = os.path.join(self.root_dir, fn)
            with h5py.File(path_or_fp, "r") as f:
                sed_labels = f[f"sed_{labels_ref}"][:]
                doa_labels = f[f"doa_{labels_ref}"][:]
                wav_length = f["wav"].shape[-1]

            sed_labels[np.isnan(sed_labels)] = 0
            doa_labels[np.isnan(doa_labels)] = 0

            wav_array = np.arange(wav_length)

            if doa_discretizer is not None:
                T, N = sed_labels.shape
                doa_labels3D = doa_labels.reshape((T, doa_size, N)).transpose((0, 2, 1))
                doa_labels_train.append(doa_labels3D[sed_labels > 0.5])

            if segment:
                if split == "train":
                    step = stride
                else:
                    step = window

                for i in range(0, wav_length, step):
                    wav_start = i
                    wav_size = len(wav_array[wav_start: i + window])
                    wav_end = wav_start + wav_size

                    if (split == "train" and wav_size > self.min_sample_size) or (
                        split != "train" and wav_size > 0
                    ):
                        self.fnames_segments.append(
                            {
                                "fn": fn,
                                "start": wav_start,
                                "end": wav_end,
                            }
                        )
                        self.sizes.append(wav_size)

                        labels_start = self.length_spectrogram(i)
                        if labels_start < 0:
                            labels_start = 0
                        n_frames = self.length_spectrogram(wav_size)

                        labels_end = labels_start + n_frames
                        labels_size = len(sed_labels[labels_start:labels_end])

                        assert labels_size == n_frames, (
                            f"fn: {fn}, labels_size: {labels_size} != n_frames: {n_frames}, wav_size: {wav_size}, "
                        )

                        labels.append(
                            {
                                "sed_labels": sed_labels[labels_start:labels_end],
                                "doa_labels": doa_labels[labels_start:labels_end],
                            }
                        )

            else:
                self.fnames_segments.append(
                    {
                        "fn": fn,
                        "start": 0,
                        "end": wav_length,
                    }
                )
                labels.append(
                    {
                        "sed_labels": sed_labels,
                        "doa_labels": doa_labels,
                    }
                )
        logger.info(
            f"segmented split: {split} with {len(self.fnames)} to {len(self.fnames_segments)}"
        )

        self.set_bucket_info(self.num_buckets)

        if doa_discretizer is not None and split == "train":
            doa_labels_train = np.concatenate(doa_labels_train).reshape(-1, 1)  # (N, 1)
            doa_discretizer_tf = KBinsDiscretizer(
                n_bins=doa_discretizer.n_bins,
                encode="ordinal",
                strategy=doa_discretizer.strategy,
            )
            doa_discretizer_tf.fit(doa_labels_train)
        else:
            doa_discretizer_tf = None

        return labels, doa_discretizer_tf

    def load_hdf5(self, path_or_fp, start=None, end=None):
        with h5py.File(path_or_fp, "r") as f:
            if start is not None and end is not None:
                wav = f["wav"][:, start:end].astype(np.float32)
            else:
                wav = f["wav"][:].astype(np.float32)
            curr_sample_rate = f["wav"].attrs["sample_rate"]
        return wav, curr_sample_rate

    def __getitem__(self, index):
        samples = {"id": index}

        fname_segment = self.fnames_segments[index]

        fn = fname_segment["fn"]
        start = fname_segment["start"]
        end = fname_segment["end"]

        path_or_fp = os.path.join(self.root_dir, fn)

        if self.use_cache_hdf:
            if fn not in self.cache:
                self.cache[fn] = self.load_hdf5(path_or_fp=path_or_fp)
            wav, curr_sample_rate = self.cache[fn]
            wav = wav[:, start:end]
        else:
            wav, curr_sample_rate = self.load_hdf5(
                path_or_fp=path_or_fp, start=start, end=end
            )

        assert wav.shape[0] == 4, wav.shape
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)  # (C, T)

        samples["source"] = feats
        return samples


class FileEventSpecHdfDataset(FileEventSpecDataset, FileEventHdfDataset):
    def __getitem__(self, index):
        samples = {"id": index}

        fname_segment = self.fnames_segments[index]

        fn = fname_segment["fn"]
        start = fname_segment["start"]
        end = fname_segment["end"]

        fname = f"{self.root_dir}/{fn}"
        if self.raw_audio_input:
            with h5py.File(fname, "r") as f:
                wav = f["wav"][:, start:end].astype(np.float32)
                curr_sample_rate = f["wav"].attrs["sample_rate"]

            assert wav.shape[0] == 4, wav.shape
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)  # (C, T)
            feat = self.extract_feat_spectrogram(feats)
        else:
            arr = np.load(fn, mmap_mode="r")  # (7, 64, T)
            feat = torch.from_numpy(arr)

        samples["source"] = feat  # (7, 64, T) or (448, T)

        return samples


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate,
        jit=True,
        spectrogram_1d=False,
        n_mels=64,
        hop_len_s=0.005,  # 5ms
        gammatone_filter_banks=False,
        audio_transforms=None,
        spec_normalize=False,
        log_spec_normalize=True,
        norm_per_channel=False,
        norm_spec_foaiv=False,
        spec_augment=True,
        use_foaiv=True,
        augment_foaiv=False,
        time_mask_F=21,
        time_mask_T=20,
        n_time_masks=1,
        n_freq_masks=1,
        iid_masks=False,
        mask_prob=1.0,
        zero_masking=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_len = int(hop_len_s * self.sample_rate)
        self.win_len = 2 * self.hop_len
        self.n_fft = _next_greater_power_of_2(self.win_len)

        self.spec_transform = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            power=None,
            pad_mode="constant",
        )

        self.n_mels = n_mels
        self.melscale_transform = MelScale(
            sample_rate=self.sample_rate,
            mel_scale="slaney",
            n_mels=n_mels,
            norm=None,
            n_stft=self.n_fft // 2 + 1,
        )

        if log_spec_normalize:
            self.amp_transform = log10
        else:
            self.amp_transform = AmplitudeToDB()

        if jit:
            self.spec_transform = torch.jit.script(self.spec_transform)
            self.melscale_transform = torch.jit.script(self.melscale_transform)
            self.amp_transform = torch.jit.script(self.amp_transform)

        self.spectrogram_1d = spectrogram_1d
        self.spec_normalize = spec_normalize

        if gammatone_filter_banks:
            fb = gammatone_fbanks.gammatone_filter_banks(
                nfilts=n_mels, nfft=self.n_fft, fs=self.sample_rate, scale="descendant"
            )[0].T
            self.melscale_transform.fb = torch.from_numpy(fb).float()

        self.mel_wts = self.melscale_transform.fb

        self.audio_transforms = audio_transforms

        self.norm_spec_foaiv = norm_spec_foaiv
        self.norm_per_channel = norm_per_channel

        self.use_foaiv = use_foaiv
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

        if gammatone_filter_banks:
            fb = gammatone_fbanks.gammatone_filter_banks(
                nfilts=n_mels, nfft=self.n_fft, fs=self.sample_rate, scale="descendant"
            )[0].T
            self.melscale_transform.fb = torch.from_numpy(fb).float()

        if self.use_foaiv:
            self.mel_wts = self.melscale_transform.fb
        else:
            self.mel_wts = None

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

    def postprocess_spec(self, feats, foa_iv=False):
        assert feats.shape[0] == 3 or feats.shape[0] == 4, (
            feats.shape and feats.shape[1] == self.n_mels
        )

        if self.spec_normalize:
            with torch.no_grad():
                if self.norm_per_channel:  # normalize over (64, T)
                    feats = F.layer_norm(feats, (feats.shape[-2], feats.shape[-1]))
                else:
                    feats = F.layer_norm(feats, feats.shape)
        return feats

    def forward(self, source):
        if self.audio_transforms is not None:
            source = self.audio_transforms(source.unsqueeze(0)).squeeze(0)

        # get spectrogram (4, 513, T)
        spec = self.spec_transform(source)

        # get intensity vector (3, 64, T)
        if self.use_foaiv:
            foa_iv = self._get_foa_intensity_vectors(spec.permute(2, 1, 0))

        if self.norm_spec_foaiv:
            foa_iv = self.postprocess_spec(foa_iv, foa_iv=True)

        # get mel-spectrogram (4, 64, T)
        melscale_spect = self.melscale_transform(spec.abs().pow(2))

        # get logmel-spectrogram (4, 64, T)
        logmel_spec = self.amp_transform(melscale_spect)

        # dB range clipping
        logmel_spec = torch.clamp(logmel_spec, -80, 0)

        # normalize and augment if enabled
        logmel_spec = self.postprocess_spec(logmel_spec, foa_iv=False)

        # apply SpecAugment method on logmel only
        if self.spec_aug_transform is not None and not self.augment_foaiv:
            logmel_spec = self.spec_aug_transform(logmel_spec)

        if self.use_foaiv:
            # concatenate logmel-spectrogram with foa iv (7, 64, T)
            feat = torch.cat((logmel_spec, foa_iv), dim=0)
        else:
            feat = logmel_spec

        # apply SpecAugment method on all spec
        if self.spec_aug_transform is not None and self.augment_foaiv:
            feat = self.spec_aug_transform(feat)

        # convert (7, 64, T) to (448, T)
        if self.spectrogram_1d:
            C, N, T = feat.shape
            feat = feat.reshape(C * N, T)

        return feat
