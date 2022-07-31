

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import BaseWrapperDataset

from seld_wav2vec2.data.transforms import RandomSeqShift, RandomSwapChannel


def check_r_unit(x, y, z):

    r = torch.sqrt(x**2 + y**2 + z**2)

    r_sel = r[r != 0]

    assert np.allclose(
        r_sel, [1.0]*len(r_sel), atol=1e-05), r[0]


class AddTargetEventDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
    ):
        super().__init__(dataset)
        self.labels = labels

    def get_label(self, index):
        return (
            self.labels[index]
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        collated["ntokens"] = len(target)
        collated["target"] = torch.tensor(target)

        return collated


class AddTargetSeldSeqClassDataset(AddTargetEventDataset):
    def __init__(
        self,
        dataset,
        labels,
        nb_classes,
    ):
        super().__init__(dataset, labels)

        self.nb_classes = nb_classes

    def __getitem__(self, index):
        item = self.dataset[index]

        item_dict = self.get_label(index)

        item["event"] = item_dict["event"]

        item_x = torch.zeros(self.nb_classes)
        item_x[item_dict["labels"]] = torch.tensor(item_dict["x"])
        item["x"] = item_x

        if "y" in item_dict:
            item_y = torch.zeros(self.nb_classes)
            item_y[item_dict["labels"]] = torch.tensor(item_dict["y"])
            item["y"] = item_y

        item_z = torch.zeros(self.nb_classes)
        item_z[item_dict["labels"]] = torch.tensor(item_dict["z"])
        item["z"] = item_z

        labels = F.one_hot(torch.LongTensor(item_dict["labels"]),
                           num_classes=self.nb_classes)
        labels = labels.sum(dim=0)
        item["labels"] = labels
        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        labels = [s["labels"] for s in samples if s["id"] in indices]
        event = [s["event"] for s in samples if s["id"] in indices]
        x = [s["x"] for s in samples if s["id"] in indices]
        z = [s["z"] for s in samples if s["id"] in indices]

        if "y" in samples[0]:
            doa_y = True
            y = [s["y"] for s in samples if s["id"] in indices]
        else:
            doa_y = False

        collated["ntokens"] = len(labels)
        collated["labels"] = torch.stack(labels)
        collated["event"] = torch.tensor(event)

        if doa_y:
            collated["doa_labels"] = torch.cat(
                [torch.stack(x), torch.stack(y), torch.stack(z)], dim=-1)
        else:
            collated["doa_labels"] = torch.cat(
                [torch.stack(x), torch.stack(z)], dim=-1)

        return collated


class AddTargetSeldAudioFrameClassDataset(AddTargetEventDataset):
    def __init__(
        self,
        dataset,
        labels,
        doa_swap_augment=False,
        doa_swap_prob=0.5,
        shift_augment=False,
        shift_prob=0.5,
        shift_rollover=False,
        n_classes=11,
    ):
        super().__init__(dataset, labels)

        self.n_classes = n_classes

        if doa_swap_augment and shift_augment:
            self.swap_transform = RandomSwapChannel(
                p=doa_swap_prob, n_classes=n_classes)
            self.shift_transform = RandomSeqShift(
                p=shift_prob, rollover=shift_rollover)
        elif doa_swap_augment and shift_augment is False:
            self.swap_transform = RandomSwapChannel(
                p=doa_swap_prob, n_classes=n_classes)
            self.shift_transform = None
        elif shift_augment and doa_swap_augment is False:
            self.swap_transform = None
            self.shift_transform = RandomSeqShift(
                p=shift_prob, rollover=shift_rollover)
        else:
            self.swap_transform = None
            self.shift_transform = None

    def __getitem__(self, index):
        item = self.dataset[index]

        item_dict = self.get_label(index)

        # x = torch.tensor(item_dict["x"]).T  # (T, N)
        # y = torch.tensor(item_dict["y"]).T  # (T, N)
        # z = torch.tensor(item_dict["z"]).T  # (T, N)

        # doa_labels = torch.cat([x, y, z], dim=-1)  # (T, 3*N)

        sed_labels = torch.tensor(item_dict["sed_labels"])  # (T, N)
        doa_labels = torch.tensor(item_dict["doa_labels"])  # (T, 3*N)

        item["sed_labels"] = sed_labels
        item["doa_labels"] = doa_labels
        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        sed_labels = [s["sed_labels"] for s in samples if s["id"] in indices]
        doa_labels = [s["doa_labels"] for s in samples if s["id"] in indices]

        sizes_sed = [len(t) for t in sed_labels]
        target_size_sed = max(sizes_sed)

        collated_sed_labels = sed_labels[0].new_zeros(len(sed_labels),
                                                      target_size_sed,
                                                      sed_labels[0].shape[1])

        for i, (label, size) in enumerate(zip(sed_labels, sizes_sed)):
            diff = size - target_size_sed
            if diff == 0:
                collated_sed_labels[i] = label
            else:
                collated_sed_labels[i] = torch.cat(
                    [label, label.new_full(
                        (-diff, sed_labels[0].shape[1]), -100)],
                    dim=0
                )

        sizes_doa = [len(doa_i) for doa_i in doa_labels]
        target_size_doa = max(sizes_doa)

        # (B, T, 3N)
        collated_doa_labels = doa_labels[0].new_zeros(len(doa_labels),
                                                      target_size_doa,
                                                      doa_labels[0].shape[1])

        for i, (label, size) in enumerate(zip(doa_labels, sizes_doa)):
            diff = size - target_size_doa
            if diff == 0:
                collated_doa_labels[i] = label
            else:
                collated_doa_labels[i] = torch.cat(
                    [label, label.new_full(
                        (-diff, doa_labels[0].shape[1]), 0.0)],
                    dim=0
                )

        if self.shift_transform is not None or self.swap_transform is not None:
            collated_source = collated["net_input"]["source"]

            if self.shift_transform is not None:
                collated_source, collated_sed_labels, collated_doa_labels = self.shift_transform(
                    collated_source, collated_sed_labels, collated_doa_labels)

            if self.swap_transform is None:
                collated["net_input"]["source"] = collated_source
            else:
                collated_source, collated_sed_labels, collated_doa_labels = self.swap_transform(
                    collated_source, collated_sed_labels, collated_doa_labels)
                collated["net_input"]["source"] = collated_source

        collated["labels_lengths"] = torch.LongTensor(sizes_sed)
        collated["sed_labels"] = collated_sed_labels  # (B, T, N)
        collated["ntokens"] = collated["labels_lengths"].sum().item()
        collated["doa_labels"] = collated_doa_labels  # (B, T, 3N)

        return collated
