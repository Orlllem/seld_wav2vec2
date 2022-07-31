

import logging

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel

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
