import collections
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
from torch.linalg import vector_norm
from torchvision.ops import sigmoid_focal_loss

from seld_wav2vec2.criterions.evaluation_metrics import (
    compute_doa_scores_regr_xyz,
    compute_sed_scores,
    early_stopping_metric,
)

eps = torch.finfo(torch.float32).eps

# label frame resolution (label_frame_res)
nb_label_frames_1s_100ms = 10  # 1/label_hop_len_s = 1/0.1


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SigmoidSchedule:
    def __init__(self, N, x_min=-3, x_max=3, a_min=1.0, a_max=2.0):
        self.a_min = a_min
        self.a_max = a_max
        self.steps = np.log(np.logspace(x_min, x_max, num=N))

    def __call__(self, step):
        x = self.steps[step]
        return self.a_min + sigmoid(x) * (self.a_max - self.a_min)


class AdaFocal(nn.Module):
    def __init__(
        self,
        num_bins=15,
        adafocal_lambda=1.0,
        adafocal_gamma_initial=1.0,
        adafocal_gamma_max=20.0,
        adafocal_gamma_min=-2.0,
        adafocal_switch_pt=0.2,
        update_gamma_every=-1,
        reduction="sum",
    ):
        super().__init__()

        self.num_bins = num_bins
        self.lamda = adafocal_lambda
        self.gamma_initial = adafocal_gamma_initial
        self.switch_pt = adafocal_switch_pt
        self.gamma_max = adafocal_gamma_max
        self.gamma_min = adafocal_gamma_min
        self.update_gamma_every = update_gamma_every
        self.reduction = reduction

        # This initializes the bin_stats variable
        self.bin_stats = collections.defaultdict(dict)
        for bin_no in range(self.num_bins):
            self.bin_stats[bin_no]["lower_boundary"] = bin_no * (1 / self.num_bins)
            self.bin_stats[bin_no]["upper_boundary"] = (bin_no + 1) * (
                1 / self.num_bins
            )
            self.bin_stats[bin_no]["gamma"] = self.gamma_initial

    # This function updates the bin statistics which are used by the Adafocal loss at every epoch.
    def update_bin_stats(self, val_adabin_dict):
        for bin_no in range(self.num_bins):
            # This is the Adafocal gamma update rule
            prev_gamma = self.bin_stats[bin_no]["gamma"]
            exp_term = val_adabin_dict[bin_no]["calibration_gap"]
            if prev_gamma > 0:
                next_gamma = prev_gamma * math.exp(self.lamda * exp_term)
            else:
                next_gamma = prev_gamma * math.exp(-self.lamda * exp_term)
            # This switches between focal and inverse-focal loss when required.
            if abs(next_gamma) < self.switch_pt:
                if next_gamma > 0:
                    next_gamma = -self.switch_pt
                else:
                    next_gamma = self.switch_pt
            self.bin_stats[bin_no]["gamma"] = max(
                min(next_gamma, self.gamma_max), self.gamma_min
            )  # gamma-clipping
            self.bin_stats[bin_no]["lower_boundary"] = val_adabin_dict[bin_no][
                "lower_bound"
            ]
            self.bin_stats[bin_no]["upper_boundary"] = val_adabin_dict[bin_no][
                "upper_bound"
            ]
        # This saves the "bin_stats" to a text file.
        save_file = os.path.join(self.args.save_path, "val_bin_stats.txt")
        with open(save_file, "a") as write_file:
            json.dump(self.bin_stats, write_file)
            write_file.write("\n")
        return

    # This function selects the gammas for each sample based on which bin it falls into.
    def get_gamma_per_sample(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            for bin_no, stats in self.bin_stats.items():
                if bin_no == 0 and pt_sample < stats["upper_boundary"]:
                    break
                elif (
                    bin_no == self.num_bins - 1 and pt_sample >= stats["lower_boundary"]
                ):
                    break
                elif (
                    pt_sample >= stats["lower_boundary"]
                    and pt_sample < stats["upper_boundary"]
                ):
                    break
            gamma_list.append(stats["gamma"])
        return torch.tensor(gamma_list)

    # This computes the loss value to be returned for back-propagation.
    def forward(self, inputs, targets):

        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        ).reshape(-1)

        p = torch.sigmoid(inputs)

        pt = p * targets + (1 - p) * (1 - targets)
        pt = pt.reshape(-1)

        gamma = self.get_gamma_per_sample(pt).to(pt.device)
        gamma_sign = torch.sign(gamma)
        gamma_mag = torch.abs(gamma)
        pt = gamma_sign * pt

        loss = ce_loss * ((1 - pt + eps) ** gamma_mag)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def schedule_weight(current_step, boundaries, values):
    # boundaries.append(float("inf"))
    # values.append(0)

    # check boundaries
    # first index in 'boundaries' greater than current_step
    s = next((x[0] for x in enumerate(boundaries) if x[1] > current_step), -1)
    return values[s - 1]


def reshape_3Dto2D(array):
    """
    Reshape 3D to 2D array (B,T,N) -> (B*T,N)
    """
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2])


def cart2sph_array(array):
    """
    Convert cartesian to spherical coordinates

    :param array x, y, z at dim -1
    :return: azi, ele stacked array in radians
    """

    assert array.shape[-1] == 3

    x = array[:, :, :, 0]
    y = array[:, :, :, 1]
    z = array[:, :, :, 2]

    B = array.shape[0]
    T = array.shape[1]

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    # r = np.sqrt(x**2 + y**2 + z**2)
    return np.stack((elevation, azimuth), axis=-1).reshape(B, T, -1)


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.sum(torch.log(torch.cosh(ey_t + self.eps)))


@dataclass
class MultitaskSedDoaCriterionConfig(FairseqDataclass):
    report_accuracy: bool = field(
        default=True,
        metadata={"help": "report accuracy metric"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    nb_classes: int = II("model.target_length")
    loss_weights: Optional[Tuple[float, float]] = field(
        default=(1, 1),
        metadata={"help": "weights for loss terms"},
    )
    doa_size: int = II("model.doa_size")
    use_labels_mask: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to mask regression using using the labels or "
            "predictions"
        },
    )
    extend_mask: Optional[bool] = field(
        default=True,
        metadata={
            "help": "When mask is extended the model must produced regression"
            "logits of (B, T, doa_size*N_classes)"
        },
    )
    constrain_r_unit: Optional[bool] = field(
        default=False,
        metadata={"help": "Constraint sqrt(x^2 + y^2 + z^2)=1"},
    )
    focal_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use focal loss"},
    )
    focal_ada: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ada-focal"},
    )
    focal_alpha: Optional[float] = field(
        default=0.25,
        metadata={"help": "focal loss alpha"},
    )
    focal_gamma: Optional[float] = field(
        default=2.0,
        metadata={"help": "focal loss gamma"},
    )
    focal_bw: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use focal loss with batching wise"},
    )
    regr_type: Optional[str] = field(
        default="mse",
        metadata={"help": "regression loss type"},
    )
    constrain_r_unit_type: Optional[str] = field(
        default="mse",
        metadata={"help": "regression constraint loss type"},
    )


@dataclass
class MultitaskSedDoaScheduleCriterionConfig(MultitaskSedDoaCriterionConfig):
    boundaries: Tuple[float, ...] = field(
        default=(20000, 30000, 60000),
        metadata={"help": "boundaries of schedule weight for doa"},
    )
    weights_values: Tuple[float, ...] = field(
        default=(1.0, 11.0, 110.0),
        metadata={"help": "values of boundaries of schedule weight for doa"},
    )


@dataclass
class MultitaskSedDoaDwaCriterionConfig(MultitaskSedDoaCriterionConfig):
    dwa_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "DWA temperature"},
    )
    K: Optional[float] = field(
        default=2.0,
        metadata={"help": "DWA scale"},
    )


@dataclass
class MultitaskSedDoaGradNormCriterionConfig(MultitaskSedDoaCriterionConfig):
    grad_norm_alpha: Any = field(
        default=1.0,
        metadata={"help": "gradnorm alpha parameter"},
    )
    skip_grad_norm_for_n_updates: Optional[int] = field(
        default=0,
        metadata={"help": "skip grad-norm for N fine-tuned updates"},
    )
    num_updates: Optional[int] = field(
        default=320000,
        metadata={"help": "num updates of training"},
    )
    grad_norm_x_range: Tuple[int, int] = field(
        default=(-3, 3),
        metadata={"help": "gradnorm x parameter"},
    )


@dataclass
class AccDoataskSedDoaCriterionConfig(FairseqDataclass):
    report_accuracy: bool = field(
        default=True,
        metadata={"help": "report accuracy metric"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    nb_classes: int = II("model.target_length")
    doa_size: int = II("model.doa_size")


@register_criterion(
    "multitask_sed_doa_seqclass", dataclass=MultitaskSedDoaCriterionConfig
)
class MultitaskSedDoaSeqClassCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=2,
        use_labels_mask=True,
        extend_mask=True,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.nb_classes = nb_classes
        self.loss_weights = loss_weights
        self.doa_size = doa_size

        self.labels = np.arange(nb_classes)

        assert len(self.loss_weights) == 2

        self.use_labels_mask = use_labels_mask
        self.extend_mask = extend_mask

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce
        )
        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            with torch.no_grad():
                logits = net_output["class_encoder_out"]
                targets = sample["sed_labels"]
                TP, TN, FP, FN = self.compute_metrics(logits, targets)
                n_correct = np.sum(TP) + np.sum(TN)
                total = n_correct + np.sum(FP) + np.sum(FN)

                logging_output["n_correct"] = n_correct
                logging_output["total"] = total

                logging_output["TP"] = TP
                logging_output["TN"] = TN
                logging_output["FP"] = FP
                logging_output["FN"] = FN
        return loss, sample_size, logging_output

    def compute_metrics(self, logits, target):
        probs = torch.sigmoid(logits.float())

        preds = (probs > 0.5).float().cpu().numpy()

        cm = multilabel_confusion_matrix(
            target.cpu().numpy(), preds, labels=self.labels
        )

        TN, FN, TP, FP = cm[:, 0, 0], cm[:, 1, 0], cm[:, 1, 1], cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True):
        class_logits = net_output["class_encoder_out"]
        reg_logits = net_output["regression_out"]

        class_labels = sample["sed_labels"].to(class_logits)
        reg_targets = sample["doa_labels"].to(reg_logits)

        if self.training:
            multi_label_loss = F.binary_cross_entropy_with_logits(
                class_logits, class_labels, reduction="sum"
            ).float()
            if self.use_labels_mask:
                class_mask = class_labels > 0.5
            else:
                class_mask = torch.sigmoid(class_logits) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits = reg_logits[class_mask_extended]
                reg_targets = reg_targets[class_mask_extended]
            else:
                B, N = class_labels.shape
                reg_logits = reg_logits.reshape((B, self.doa_size, N)).transpose(2, 1)
                reg_targets = reg_targets.reshape((B, self.doa_size, N)).transpose(2, 1)

                reg_logits = reg_logits[class_mask]
                reg_targets = reg_targets[class_mask]

            reg_loss = F.mse_loss(reg_logits, reg_targets, reduction="sum").float()

            loss = (
                self.loss_weights[0] * multi_label_loss
                + self.loss_weights[1] * reg_loss
            )

        else:
            # inference-time
            multi_label_loss = F.binary_cross_entropy_with_logits(
                class_logits, class_labels, reduction="sum"
            ).float()

            class_mask = torch.sigmoid(class_logits.float()) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits = reg_logits[class_mask_extended]
                reg_targets = reg_targets[class_mask_extended]
            else:
                B, N = class_labels.shape
                reg_logits = reg_logits.reshape((B, self.doa_size, N)).transpose(2, 1)
                reg_targets = reg_targets.reshape((B, self.doa_size, N)).transpose(2, 1)

                reg_logits = reg_logits[class_mask]
                reg_targets = reg_targets[class_mask]

            reg_loss = F.mse_loss(reg_logits, reg_targets, reduction="sum").float()

            loss = multi_label_loss + reg_loss

        return loss, multi_label_loss, reg_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(
            log.get("multi_label_loss", 0) for log in logging_outputs
        )
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2),
            ntokens,
            round=3,
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

            tp = np.sum([log.get("TP", 0) for log in logging_outputs], axis=0)
            fp = np.sum([log.get("FP", 0) for log in logging_outputs], axis=0)
            fn = np.sum([log.get("FN", 0) for log in logging_outputs], axis=0)

            f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("f1_score", f1_value * 100.0)


@register_criterion(
    "multitask_sed_doa_seqclass_cart_dcase_2019",
    dataclass=MultitaskSedDoaCriterionConfig,
)
class MultitaskSeldSeqClassCartDcase2019Criterion(MultitaskSedDoaSeqClassCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
    ):
        super().__init__(
            task,
            sentence_avg,
            report_accuracy,
            nb_classes,
            loss_weights,
            doa_size,
            use_labels_mask,
            extend_mask,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce
        )
        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()
                reg_logits = net_output["regression_out"].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits.float())
                class_mask = class_probs > 0.5
                class_preds = class_mask.float()

                # ignore padded labels -100
                class_pad_mask = class_labels < 0
                class_labels[class_pad_mask] = torch.tensor(0).to(class_labels)

                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits[~class_mask_extended] = torch.tensor(0.0).to(reg_targets)
                reg_logits = reg_logits.cpu().numpy()

                class_preds = class_preds.cpu().numpy()
                class_labels = class_labels.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                logging_output["y_pred_class"] = class_preds
                logging_output["y_true_class"] = class_labels
                logging_output["y_pred_reg"] = reg_logits
                logging_output["y_true_reg"] = reg_targets

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(
            log.get("multi_label_loss", 0) for log in logging_outputs
        )
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        y_pred_class = np.concatenate(
            [log.get("y_pred_class", 0) for log in logging_outputs], axis=0
        )
        y_true_class = np.concatenate(
            [log.get("y_true_class", 0) for log in logging_outputs], axis=0
        )
        y_pred_reg = np.concatenate(
            [log.get("y_pred_reg", 0) for log in logging_outputs], axis=0
        )
        y_true_reg = np.concatenate(
            [log.get("y_true_reg", 0) for log in logging_outputs], axis=0
        )

        er_metric = compute_doa_scores_regr_xyz(
            y_pred_reg, y_true_reg, y_pred_class, y_true_class
        )

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        doa_metric = [_doa_err, _frame_recall]

        sed_metric = compute_sed_scores(
            y_pred_class, y_true_class, nb_label_frames_1s_100ms
        )
        _er = sed_metric[0]
        _f = sed_metric[1]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2),
            ntokens,
            round=5,
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens, round=5
        )

        metrics.log_scalar("f1_score", _f * 100, round=5)
        metrics.log_scalar("doa_error", _doa_err, round=5)
        metrics.log_scalar("frame_recall", _frame_recall * 100, round=5)
        if np.isnan(_er):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", _er * 100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)


@register_criterion(
    "multitask_sed_doa_audio_frame_class", dataclass=MultitaskSedDoaCriterionConfig
)
class MultitaskSeldAudioFrameCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
        constrain_r_unit=False,
        focal_loss=False,
        focal_ada=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
        regr_type="mse",
        constrain_r_unit_type="mse",
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.nb_classes = nb_classes
        self.loss_weights = loss_weights
        self.doa_size = doa_size

        self.labels = np.arange(nb_classes)

        assert len(self.loss_weights) == 2

        self.use_labels_mask = use_labels_mask
        self.extend_mask = extend_mask
        self.constrain_r_unit = constrain_r_unit

        self.focal_loss = focal_loss
        self.focal_ada = focal_ada
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_bw = focal_bw

        if regr_type == "logcosh":
            self.regr_loss = LogCoshLoss()
        elif regr_type == "mae":
            self.regr_loss = nn.L1Loss(reduction="sum")
        else:
            self.regr_loss = nn.MSELoss(reduction="sum")

        if constrain_r_unit_type == "logcosh":
            self.regr_r_unit_loss = LogCoshLoss()
        elif constrain_r_unit_type == "mae":
            self.regr_r_unit_loss = nn.L1Loss(reduction="sum")
        else:
            self.regr_r_unit_loss = nn.MSELoss(reduction="sum")

        if self.focal_ada:
            self.ada_focal = AdaFocal(reduction="sum")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce, thr=self.task.best_threshold
        )
        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            with torch.no_grad():
                logits = net_output["class_encoder_out"]
                targets = sample["sed_labels"]

                # ignore padded labels -100
                class_pad_mask = targets < 0
                targets[class_pad_mask] = torch.tensor(0).to(targets.device)
                TP, TN, FP, FN = self.compute_metrics(logits, targets)
                n_correct = np.sum(TP) + np.sum(TN)
                total = n_correct + np.sum(FP) + np.sum(FN)

                logging_output["n_correct"] = n_correct
                logging_output["total"] = total

                logging_output["TP"] = TP
                logging_output["TN"] = TN
                logging_output["FP"] = FP
                logging_output["FN"] = FN
        return loss, sample_size, logging_output

    def compute_metrics(self, logits, targets):
        probs = torch.sigmoid(logits.float())

        preds = (probs > 0.5).float().cpu().numpy()
        targets = targets.cpu().numpy()

        TN, FN, TP, FP = 0, 0, 0, 0
        for i in range(len(targets)):
            cm = multilabel_confusion_matrix(targets[i], preds[i], labels=self.labels)

            TN += cm[:, 0, 0]
            FN += cm[:, 1, 0]
            TP += cm[:, 1, 1]
            FP += cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True, thr=0.5):
        class_logits = net_output["class_encoder_out"]
        reg_logits = net_output["regression_out"]

        class_labels = sample["sed_labels"].to(class_logits)
        reg_targets = sample["doa_labels"].to(reg_logits)

        if self.training:
            # ignore padded labels -100
            weights_pad_mask = class_labels >= 0
            weights = (weights_pad_mask).to(class_logits)

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            if self.focal_loss:
                if self.focal_bw:
                    class_labels_1d = class_labels.reshape(-1).cpu().numpy()
                    unique_labels = np.unique(class_labels_1d)
                    if len(unique_labels) > 1:
                        class_weights = class_weight.compute_class_weight(
                            "balanced", unique_labels, class_labels_1d
                        )
                        focal_alpha = class_weights[1] / sum(class_weights)
                    elif all(unique_labels == [0]):
                        focal_alpha = 0.0
                    else:
                        focal_alpha = 1.0
                else:
                    focal_alpha = self.focal_alpha
                if self.focal_ada:
                    multi_label_loss = self.ada_focal(
                        class_logits,
                        class_labels,
                    )
                else:
                    multi_label_loss = sigmoid_focal_loss(
                        class_logits,
                        class_labels,
                        alpha=focal_alpha,
                        gamma=self.focal_gamma,
                        reduction="sum",
                    )
            else:
                multi_label_loss = F.binary_cross_entropy_with_logits(
                    class_logits, class_labels, weight=weights, reduction="sum"
                )

            if self.use_labels_mask:
                class_mask = class_labels > thr
            else:
                class_mask = torch.sigmoid(class_logits) > thr

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

                if self.constrain_r_unit:
                    B, T, N = class_labels.shape
                    reg_logits = reg_logits.reshape((B, T, self.doa_size, N)).transpose(
                        3, 2
                    )
                    reg_targets = reg_targets.reshape(
                        (B, T, self.doa_size, N)
                    ).transpose(3, 2)

                    reg_logits_mask = reg_logits[class_mask]
                    reg_targets_mask = reg_targets[class_mask]

                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = self.regr_r_unit_loss(
                        reg_norm, torch.ones(reg_norm.shape).to(reg_norm)
                    )
                    reg_loss = reg_loss + reg_unit_loss
            else:
                B, T, N = class_labels.shape
                reg_logits = reg_logits.reshape((B, T, self.doa_size, N)).transpose(
                    3, 2
                )
                reg_targets = reg_targets.reshape((B, T, self.doa_size, N)).transpose(
                    3, 2
                )

                reg_logits_mask = reg_logits[class_mask]
                reg_targets_mask = reg_targets[class_mask]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

                if self.constrain_r_unit:
                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = self.regr_r_unit_loss(
                        reg_norm, torch.ones(reg_norm.shape).to(reg_norm)
                    )
                    reg_loss = reg_loss + reg_unit_loss

            loss = (
                self.loss_weights[0] * multi_label_loss
                + self.loss_weights[1] * reg_loss
            )

        else:
            # inference-time
            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            if self.focal_loss:
                if self.focal_ada:
                    multi_label_loss = self.ada_focal(
                        class_logits,
                        class_labels,
                    )
                else:
                    multi_label_loss = sigmoid_focal_loss(
                        class_logits,
                        class_labels,
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                        reduction="sum",
                    )
            else:
                multi_label_loss = F.binary_cross_entropy_with_logits(
                    class_logits, class_labels, reduction="sum"
                )

            class_mask = torch.sigmoid(class_logits) > thr

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

                if self.constrain_r_unit:
                    B, T, N = class_labels.shape
                    reg_logits = reg_logits.reshape((B, T, self.doa_size, N)).transpose(
                        3, 2
                    )
                    reg_targets = reg_targets.reshape(
                        (B, T, self.doa_size, N)
                    ).transpose(3, 2)

                    reg_logits_mask = reg_logits[class_mask]
                    reg_targets_mask = reg_targets[class_mask]

                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(
                        reg_norm,
                        torch.ones(reg_norm.shape).to(reg_norm),
                        reduction="sum",
                    )
                    reg_loss = reg_loss + reg_unit_loss
            else:
                B, T, N = class_labels.shape
                reg_logits = reg_logits.reshape((B, T, self.doa_size, N)).transpose(
                    3, 2
                )
                reg_targets = reg_targets.reshape((B, T, self.doa_size, N)).transpose(
                    3, 2
                )

                reg_logits_mask = reg_logits[class_mask]
                reg_targets_mask = reg_targets[class_mask]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

                if self.constrain_r_unit:
                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(
                        reg_norm,
                        torch.ones(reg_norm.shape).to(reg_norm),
                        reduction="sum",
                    )
                    reg_loss = reg_loss + reg_unit_loss

            loss = multi_label_loss + reg_loss

        return loss, multi_label_loss, reg_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(
            log.get("multi_label_loss", 0) for log in logging_outputs
        )
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2),
            ntokens,
            round=5,
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens, round=5
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

            tp = np.sum([log.get("TP", 0) for log in logging_outputs], axis=0)
            fp = np.sum([log.get("FP", 0) for log in logging_outputs], axis=0)
            fn = np.sum([log.get("FN", 0) for log in logging_outputs], axis=0)

            f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("f1_score", f1_value * 100.0)


@register_criterion(
    "multitask_sed_doa_audio_frame_class_cart_dcase",
    dataclass=MultitaskSedDoaCriterionConfig,
)
class MultitaskSeldAudioFrameCartDcaseCriterion(MultitaskSeldAudioFrameCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
        constrain_r_unit=False,
        focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
        regr_type="mse",
    ):
        super().__init__(
            task,
            sentence_avg,
            report_accuracy,
            nb_classes,
            loss_weights,
            doa_size,
            use_labels_mask,
            extend_mask,
            constrain_r_unit,
            focal_loss,
            focal_alpha,
            focal_gamma,
            focal_bw,
            regr_type,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce
        )
        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy and not self.training:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()
                reg_logits = net_output["regression_out"].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits)

                class_probs = class_probs.cpu().numpy()
                class_labels = class_labels.cpu().numpy()

                reg_logits = reg_logits.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                class_probs = reshape_3Dto2D(class_probs)
                class_labels = reshape_3Dto2D(class_labels)

                reg_logits = reshape_3Dto2D(reg_logits)
                reg_targets = reshape_3Dto2D(reg_targets)

                logging_output["class_probs"] = class_probs
                logging_output["class_labels"] = class_labels
                logging_output["reg_logits"] = reg_logits
                logging_output["reg_targets"] = reg_targets

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(
            log.get("multi_label_loss", 0) for log in logging_outputs
        )
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2),
            ntokens,
            round=5,
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens, round=5
        )


@register_criterion(
    "multitask_sed_doa_audio_frame_class_cart_dcase_doa_schedule",
    dataclass=MultitaskSedDoaScheduleCriterionConfig,
)
class MultitaskSeldAudioFrameCartDcaseScheduleCriterion(
    MultitaskSeldAudioFrameCartDcaseCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
        constrain_r_unit=False,
        focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
        boundaries=[20000, 30000, 60000],
        weights_values=[1.0, 11.0, 110.0],
    ):
        super().__init__(
            task,
            sentence_avg,
            report_accuracy,
            nb_classes,
            loss_weights,
            doa_size,
            use_labels_mask,
            extend_mask,
            constrain_r_unit,
            focal_loss,
            focal_alpha,
            focal_gamma,
            focal_bw,
        )

        # assert self.loss_weights[0] == 1.0, "weight[0] must be 1.0"
        # assert self.loss_weights[1] == 1.0, "weight[1] must be 1.0"

        self.boundaries = boundaries
        self.weights_values = weights_values

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        current_step = model.w2v_encoder.num_updates
        reg_weight = schedule_weight(current_step, self.boundaries, self.weights_values)

        self.loss_weights[1] = reg_weight

        loss, sample_size, logging_output = super().forward(model, sample, reduce)

        logging_output["reg_weight"] = reg_weight

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        reg_weight = [log.get("reg_weight", 0) for log in logging_outputs]

        metrics.log_scalar(
            "reg_weight", sum(reg_weight) / len(reg_weight), len(reg_weight), round=3
        )


@register_criterion(
    "multitask_sed_doa_audio_frame_class_cart_dcase_doa_dwa",
    dataclass=MultitaskSedDoaDwaCriterionConfig,
)
class MultitaskSeldAudioFrameCartDcaseDwaCriterion(
    MultitaskSeldAudioFrameCartDcaseCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
        constrain_r_unit=False,
        focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
        dwa_temperature=1,
        K=2,
    ):
        super().__init__(
            task,
            sentence_avg,
            report_accuracy,
            nb_classes,
            loss_weights,
            doa_size,
            use_labels_mask,
            extend_mask,
            constrain_r_unit,
            focal_loss,
            focal_alpha,
            focal_gamma,
            focal_bw,
        )
        self.dwa_temperature = dwa_temperature
        self.loss_tm = []
        self.K = K

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        with torch.cuda.amp.autocast(dtype=torch.float32):
            _, multi_label_loss, reg_loss = self.compute_loss(
                net_output, sample, reduce=reduce
            )

            task_loss = torch.stack([multi_label_loss, reg_loss])

            if self.training:
                loss = self.compute_dwa(task_loss)
                self.loss_tm.append(task_loss)
            else:
                loss = multi_label_loss + reg_loss

        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        weights0, weights1 = self.loss_weights[0], self.loss_weights[1]

        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
            "weights0": weights0,
            "weights1": weights1,
        }
        if self.report_accuracy and not self.training:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()
                reg_logits = net_output["regression_out"].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits)

                class_probs = class_probs.cpu().numpy()
                class_labels = class_labels.cpu().numpy()

                reg_logits = reg_logits.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                class_probs = reshape_3Dto2D(class_probs)
                class_labels = reshape_3Dto2D(class_labels)

                reg_logits = reshape_3Dto2D(reg_logits)
                reg_targets = reshape_3Dto2D(reg_targets)

                logging_output["class_probs"] = class_probs
                logging_output["class_labels"] = class_labels
                logging_output["reg_logits"] = reg_logits
                logging_output["reg_targets"] = reg_targets

        return loss, sample_size, logging_output

    def compute_dwa(self, task_loss):
        if len(self.loss_tm) >= 3:
            loss_tm1 = self.loss_tm[-2].clone().detach()
            loss_tm2 = self.loss_tm[-3].clone().detach()

            w_tm1 = loss_tm1 / (loss_tm2 + eps)
            self.loss_weights = self.K * F.softmax(w_tm1 / self.dwa_temperature, dim=-1)

        loss = self.loss_weights[0] * task_loss[0] * self.loss_weights[1] * task_loss[1]
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        weights0 = [log.get("weights0", 0) for log in logging_outputs]
        weights1 = [log.get("weights1", 0) for log in logging_outputs]

        metrics.log_scalar("weights0", weights0[-1], weight=0, round=3)
        metrics.log_scalar("weights1", weights1[-1], weight=0, round=3)


@register_criterion(
    "multitask_sed_doa_audio_frame_class_cart_dcase_grad_norm",
    dataclass=MultitaskSedDoaGradNormCriterionConfig,
)
class MultitaskSeldAudioFrameCartDcaseGradNormCriterion(
    MultitaskSeldAudioFrameCartDcaseCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=True,
        extend_mask=True,
        constrain_r_unit=False,
        focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
        regr_type="mse",
        grad_norm_alpha=0.12,
        grad_norm_x_range=[-3, 3],
        num_updates=320000,
        skip_grad_norm_for_n_updates=0,
    ):
        super().__init__(
            task,
            sentence_avg,
            report_accuracy,
            nb_classes,
            loss_weights,
            doa_size,
            use_labels_mask,
            extend_mask,
            constrain_r_unit,
            focal_loss,
            focal_alpha,
            focal_gamma,
            focal_bw,
            regr_type,
        )

        self.alpha = grad_norm_alpha
        self.initial_task_loss = None
        self.skip_grad_norm_for_n_updates = skip_grad_norm_for_n_updates

        if not isinstance(self.alpha, float):
            assert len(self.alpha) == 2
            self.sigmoid_schedule = SigmoidSchedule(
                N=num_updates,
                x_min=grad_norm_x_range[0],
                x_max=grad_norm_x_range[1],
                a_min=self.alpha[0],
                a_max=self.alpha[1],
            )

    def compute_grad_norm(self, task_loss, model, alpha):
        # compute the weighted loss w_i(t) * L_i(t)
        weighted_task_loss = torch.mul(model.w2v_encoder.weights, task_loss)

        if self.initial_task_loss is None:
            # set L(0)
            self.initial_task_loss = task_loss.data.cpu().numpy()

        # get the total loss
        loss = torch.sum(weighted_task_loss)

        if self.skip_grad_norm:
            return loss

        # This is equivalent to compute each \nabla_W L_i(t)
        loss.backward(retain_graph=True)

        # set the gradients of w_i(t) to zero because
        # these gradients have to be updated using the GradNorm loss
        model.w2v_encoder.weights.grad.data = model.w2v_encoder.weights.grad.data * 0.0

        # get layer of shared weights
        W = model.w2v_encoder.get_last_shared_layer()

        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(
                task_loss[i],
                W.parameters(),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            # compute the norm
            norms.append(torch.norm(torch.mul(model.w2v_encoder.weights[i], gygw[0])))
        norms = torch.stack(norms)

        # compute the inverse training rate r_i(t)
        # \curl{L}_i
        loss_ratio = task_loss.data.cpu().numpy() / self.initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde{G}_w(t)
        mean_norm = np.mean(norms.data.cpu().numpy())

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(
            mean_norm * (inverse_train_rate**alpha), requires_grad=False
        )
        constant_term = constant_term.cuda()
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

        # compute the gradient for the weights
        model.w2v_encoder.weights.grad = torch.autograd.grad(grad_norm_loss,
                                                             model.w2v_encoder.weights,
                                                             retain_graph=True)[0]

        return grad_norm_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        # renormalize
        # normalize_coeff = 2.0 / torch.sum(model.w2v_encoder.weights.data, dim=0)
        # model.w2v_encoder.weights.data = model.w2v_encoder.weights.data * normalize_coeff

        current_step = model.w2v_encoder.num_updates

        if (
            current_step <= model.w2v_encoder.freeze_finetune_updates
            or current_step <= self.skip_grad_norm_for_n_updates
        ):
            self.skip_grad_norm = True
        else:
            self.skip_grad_norm = False

        if self.skip_grad_norm:
            model.w2v_encoder.weights.data = torch.tensor(self.loss_weights).to(
                model.w2v_encoder.weights.device
            )
        else:
            model.w2v_encoder.weights.data = 2 * F.softmax(model.w2v_encoder.weights.data, dim=-1)

        if isinstance(self.alpha, float):
            alpha = self.alpha
        else:
            alpha = self.sigmoid_schedule(current_step)

            metrics.log_scalar("alpha", alpha, weight=0, round=3, priority=10000)

        weights = model.w2v_encoder.weights.data.cpu()

        assert np.isclose(sum(weights), 2.0, atol=1e-3), f"{sum(weights)} != 2.0"
        weights0, weights1 = weights[0], weights[1]

        metrics.log_scalar("weights0", weights0, weight=0, round=3, priority=10000)
        metrics.log_scalar("weights1", weights1, weight=0, round=3, priority=10000)

        # evaluate each task loss L_i(t)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            loss, multi_label_loss, reg_loss = self.compute_loss(
                net_output, sample, reduce=reduce
            )
            task_loss = torch.stack([multi_label_loss, reg_loss])
            if self.training:
                grad_norm_loss = self.compute_grad_norm(task_loss, model, alpha=alpha)
            else:
                grad_norm_loss = torch.tensor(0).to(loss.device)

        sample_size = (
            sample["sed_labels"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "multi_label_loss": multi_label_loss.data,
            "reg_loss": reg_loss.data,
            "grad_norm_loss": grad_norm_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy and not self.training:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()
                reg_logits = net_output["regression_out"].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits)

                class_probs = class_probs.cpu().numpy()
                class_labels = class_labels.cpu().numpy()

                reg_logits = reg_logits.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                class_probs = reshape_3Dto2D(class_probs)
                class_labels = reshape_3Dto2D(class_labels)

                reg_logits = reshape_3Dto2D(reg_logits)
                reg_targets = reshape_3Dto2D(reg_targets)

                logging_output["class_probs"] = class_probs
                logging_output["class_labels"] = class_labels
                logging_output["reg_logits"] = reg_logits
                logging_output["reg_targets"] = reg_targets

        return grad_norm_loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(
            log.get("multi_label_loss", 0) for log in logging_outputs
        )
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        grad_norm_loss = sum(log.get("grad_norm_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2),
            ntokens,
            round=5,
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens, round=5
        )

        metrics.log_scalar(
            "grad_norm_loss", grad_norm_loss / ntokens / math.log(2), ntokens, round=5
        )
