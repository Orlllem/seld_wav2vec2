import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from omegaconf import II
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torchvision.ops import sigmoid_focal_loss

from seld_wav2vec2.utils import torch_to_numpy

eps = torch.finfo(torch.float32).eps

# label frame resolution (label_frame_res)
nb_label_frames_1s_100ms = 10  # 1/label_hop_len_s = 1/0.1


def compute_class_weight_labels(labels, num_classes):
    label_counts = np.bincount(labels, minlength=num_classes)
    weight = np.where(label_counts == 0, 0.0, 1 / label_counts)
    weight = weight / sum(weight)
    return weight


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


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, alpha=-1.0, gamma=2.0, reduction="sum"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


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
    use_labels_mask: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Whether to mask regression using using the labels or predictions"
        },
    )
    extend_mask: Optional[bool] = field(
        default=False,
        metadata={
            "help": "When mask is extended the model must produced regression"
            "logits of (B, T, doa_size*N_classes)"
        },
    )
    class_type: Optional[str] = field(
        default="CE",
        metadata={"help": "type of classification loss (CE, FL, ASL)"},
    )
    gamma: Any = field(
        default=2.0,
        metadata={"help": "gamma parameter of loss"},
    )
    clip: Optional[float] = field(
        default=0.0,
        metadata={"help": "clip parameter of loss"},
    )
    regr_type: Optional[str] = field(
        default="mse",
        metadata={"help": "regression loss type"},
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
    grad_norm_scale: Any = field(
        default=2.0,
        metadata={"help": "gradnorm scale weights parameter"},
    )
    skip_grad_norm_for_n_updates: Optional[int] = field(
        default=0,
        metadata={"help": "skip grad-norm for N fine-tuned updates"},
    )
    num_updates: Optional[int] = field(
        default=II("optimization.max_update"),
        metadata={"help": "num updates of training"},
    )
    grad_norm_x_range: Tuple[int, int] = field(
        default=(-3, 3),
        metadata={"help": "gradnorm x parameter"},
    )


@dataclass
class MultitaskSeldCriterionDiscConfig(MultitaskSedDoaCriterionConfig):
    balance_doa_classes: Optional[bool] = field(
        default=False,
        metadata={"help": "Balance the DOA labels by frequency"},
    )


@dataclass
class MultitaskSeldCriterionDiscGradNormConfig(
    MultitaskSeldCriterionDiscConfig, MultitaskSedDoaGradNormCriterionConfig
):
    pass


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
        use_labels_mask=1.0,
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

        if use_labels_mask >= 1.0:
            use_labels_mask = True
        elif use_labels_mask <= 0:
            use_labels_mask = False

        self.use_labels_mask = use_labels_mask
        self.extend_mask = extend_mask

    def class_mask_type(self):
        if isinstance(self.use_labels_mask, bool):
            mask_type = self.use_labels_mask
        else:
            if random.random() <= self.use_labels_mask:
                mask_type = True
            else:
                mask_type = False
        return mask_type

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
                logits = net_output["class_encoder_out"].float()
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

        preds = torch_to_numpy((probs > 0.5).float())

        cm = multilabel_confusion_matrix(
            torch_to_numpy(target), preds, labels=self.labels
        )

        TN, FN, TP, FP = cm[:, 0, 0], cm[:, 1, 0], cm[:, 1, 1], cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True):
        class_logits = net_output["class_encoder_out"].float()
        reg_logits = net_output["regression_out"].float()
        class_labels = sample["sed_labels"].to(class_logits)
        reg_targets = sample["doa_labels"].to(reg_logits)

        if self.training:
            multi_label_loss = F.binary_cross_entropy_with_logits(
                class_logits, class_labels, reduction="sum"
            ).float()
            if self.class_mask_type():
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
        use_labels_mask=1.0,
        extend_mask=True,
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.nb_classes = nb_classes
        self.loss_weights = loss_weights
        self.doa_size = doa_size

        self.labels = np.arange(nb_classes)

        assert len(self.loss_weights) == 2

        if use_labels_mask >= 1.0:
            use_labels_mask = True
        elif use_labels_mask <= 0:
            use_labels_mask = False

        self.use_labels_mask = use_labels_mask
        self.extend_mask = extend_mask

        if class_type == "FL":
            self.class_loss = SigmoidFocalLoss(alpha=-1.0, gamma=gamma)
        elif class_type == "ASL":
            self.class_loss = AsymmetricLoss(
                gamma_neg=gamma[0], gamma_pos=gamma[1], clip=clip
            )
        else:
            self.class_loss = nn.BCEWithLogitsLoss(reduction="sum")

        if regr_type == "logcosh":
            self.regr_loss = LogCoshLoss()
        elif regr_type == "mae":
            self.regr_loss = nn.L1Loss(reduction="sum")
        else:
            self.regr_loss = nn.MSELoss(reduction="sum")

    def class_mask_type(self):
        if isinstance(self.use_labels_mask, bool):
            mask_type = self.use_labels_mask
        else:
            if random.random() <= self.use_labels_mask:
                mask_type = True
            else:
                mask_type = False
        return mask_type

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
                logits = net_output["class_encoder_out"].float()
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

        preds = torch_to_numpy((probs > 0.5).float())
        targets = torch_to_numpy(targets)

        TN, FN, TP, FP = 0, 0, 0, 0
        for i in range(len(targets)):
            cm = multilabel_confusion_matrix(targets[i], preds[i], labels=self.labels)

            TN += cm[:, 0, 0]
            FN += cm[:, 1, 0]
            TP += cm[:, 1, 1]
            FP += cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True):
        class_logits = net_output["class_encoder_out"].float()
        reg_logits = net_output["regression_out"].float()

        class_labels = sample["sed_labels"].to(class_logits)
        reg_targets = sample["doa_labels"].to(reg_logits)

        if self.training:
            # ignore padded labels -100
            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            multi_label_loss = self.class_loss(class_logits, class_labels)

            if self.class_mask_type():
                class_mask = class_labels > self.task.best_threshold
            else:
                class_mask = torch.sigmoid(class_logits) > self.task.best_threshold

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

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

            loss = (
                self.loss_weights[0] * multi_label_loss
                + self.loss_weights[1] * reg_loss
            )

        else:
            # inference-time
            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            multi_label_loss = self.class_loss(class_logits, class_labels)

            class_mask = torch.sigmoid(class_logits) > self.task.best_threshold

            if self.extend_mask:
                class_mask_extended = torch.cat([class_mask] * self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask, reg_targets_mask)

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
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
        )

    def step_forward_model(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        training = self.training or self.task.cfg.segment_eval

        if self.task.cfg.get("apply_record_inference_on_train", False) or (not training):
            net_output, sample = self.task.inference_step(sample, model)
        else:
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

        return loss, sample_size, logging_output, net_output, sample

    def forward(self, model, inputs, reduce=True):
        loss, sample_size, logging_output, net_output, sample = (
            self.step_forward_model(model, inputs, reduce=True)
        )

        if self.report_accuracy and not self.training:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()  # (B, T, N)
                reg_logits = net_output["regression_out"].float()
                class_labels = sample["sed_labels"].to(class_logits)  # (B, T, N)
                reg_targets = sample["doa_labels"].to(reg_logits)  # (B, T, 3N)

                class_probs = torch.sigmoid(class_logits)

                class_probs = torch_to_numpy(class_probs)
                class_labels = torch_to_numpy(class_labels)

                reg_logits = torch_to_numpy(reg_logits)
                reg_targets = torch_to_numpy(reg_targets)

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
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
        dwa_temperature=1,
        K=2,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
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
                class_logits = net_output["class_encoder_out"].float()  # (B, T, N)
                reg_logits = net_output["regression_out"].float()
                class_labels = sample["sed_labels"].to(class_logits)  # (B, T, N)
                reg_targets = sample["doa_labels"].to(reg_logits)  # (B, T, 3N)

                class_probs = torch.sigmoid(class_logits)

                class_probs = torch_to_numpy(class_probs)
                class_labels = torch_to_numpy(class_labels)

                reg_logits = torch_to_numpy(reg_logits)
                reg_targets = torch_to_numpy(reg_targets)

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
        use_labels_mask=1.0,
        extend_mask=True,
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
        grad_norm_alpha=0.12,
        grad_norm_scale=2.0,
        grad_norm_x_range=[-3, 3],
        num_updates=320000,
        skip_grad_norm_for_n_updates=0,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
        )

        self.alpha = grad_norm_alpha
        self.grad_scale = grad_norm_scale
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
            self.initial_task_loss = torch_to_numpy(task_loss.data)

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
        loss_ratio = torch_to_numpy(task_loss.data) / self.initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde{G}_w(t)
        mean_norm = np.mean(torch_to_numpy(norms.data))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(
            mean_norm * (inverse_train_rate**alpha), requires_grad=False
        )
        constant_term = constant_term.cuda()
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

        # compute the gradient for the weights
        model.w2v_encoder.weights.grad = torch.autograd.grad(
            grad_norm_loss, model.w2v_encoder.weights, retain_graph=True
        )[0]

        return grad_norm_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if self.training or self.task.cfg.segment_eval:
            net_output = model(**sample["net_input"])
        else:
            net_output, sample = self.task.inference_step(sample, model)

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
            # renormalize
            model.w2v_encoder.weights.data = self.grad_scale * F.softmax(
                model.w2v_encoder.weights.data, dim=-1
            )

        if isinstance(self.alpha, float):
            alpha = self.alpha
        else:
            alpha = self.sigmoid_schedule(current_step)

            metrics.log_scalar("alpha", alpha, weight=0, round=3, priority=10000)

        weights = model.w2v_encoder.weights.data.cpu()

        assert np.isclose(sum(weights), self.grad_scale, atol=1e-3), (
            f"{sum(weights)} != 2.0"
        )
        weights0, weights1 = weights[0], weights[1]

        metrics.log_scalar("weights0", weights0, weight=0, round=3, priority=10000)
        metrics.log_scalar("weights1", weights1, weight=0, round=3, priority=10000)

        # evaluate each task loss L_i(t)
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
                class_logits = net_output["class_encoder_out"].float()  # (B, T, N)
                reg_logits = net_output["regression_out"].float()
                class_labels = sample["sed_labels"].float()  # (B, T, N)
                reg_targets = sample["doa_labels"].float()  # (B, T, 3N)

                class_probs = torch.sigmoid(class_logits)

                class_probs = torch_to_numpy(class_probs)
                class_labels = torch_to_numpy(class_labels)

                reg_logits = torch_to_numpy(reg_logits)
                reg_targets = torch_to_numpy(reg_targets)

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


@register_criterion(
    "multitask_frame_cart_dcase_discretizer",
    dataclass=MultitaskSeldCriterionDiscConfig,
)
class MultitaskSeldAudioFrameCartDiscretizerCriterion(
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
        use_labels_mask=1.0,
        extend_mask=True,
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
        balance_doa_classes=False,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
        )

        assert self.task.cfg.doa_discretizer.flag

        self.n_bins = self.task.cfg.doa_discretizer.n_bins
        self.balance_doa_classes = balance_doa_classes

    def compute_metrics(self, preds, targets, labels):
        cm = confusion_matrix(targets, preds, labels)

        TP = np.diag(cm)  # Diagonal elements are the True Positives
        FP = cm.sum(axis=0) - TP  # Column-wise sum minus TP
        FN = cm.sum(axis=1) - TP  # Row-wise sum minus TP
        TN = cm.sum() - (FP + FN + TP)  # Total sum minus FP, FN, TP
        return TP, TN, FP, FN

    def get_class_mask(self, class_logits, class_labels, thr=0.5):
        if self.training:
            if self.class_mask_type():
                class_mask = class_labels > thr
            else:
                class_mask = torch.sigmoid(class_logits) > thr
        else:
            # inference-time
            class_mask = torch.sigmoid(class_logits) > thr
        return class_mask

    def compute_loss(self, net_output, sample, reduce=True):
        class_logits = net_output["class_encoder_out"].float()  # (B, T, N)
        reg_logits = net_output["regression_out"].float()  # (B, T, N, 3, BINS)
        class_labels = sample["sed_labels"].to(class_logits)  # (B, T, N)
        reg_targets = sample["doa_labels"].to(reg_logits)  # (B, T, 3N)

        class_mask = self.get_class_mask(
            class_logits, class_labels, thr=self.task.best_threshold
        )

        # ignore padded labels -100
        class_pad_mask = class_labels < 0
        class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

        multi_label_loss = self.class_loss(class_logits, class_labels)

        B, T, N = class_labels.shape
        reg_targets_3d = reg_targets.reshape((B, T, self.doa_size, N)).transpose(
            3, 2
        )  # (B, T, N, 3)

        doa_logits = reg_logits[class_mask]  # (E, 3, BINS)
        doa_logits = doa_logits.transpose(1, 2)  # (E, BINS, 3)
        reg_targets_mask = torch_to_numpy(reg_targets_3d[class_mask])

        doa_labels = self.task.doa_discretizer_tf.transform(
            reg_targets_mask.reshape(-1, 1)
        )
        doa_labels = doa_labels.reshape(reg_targets_mask.shape)
        doa_labels = torch.from_numpy(doa_labels).to(reg_targets).long()  # (E, 3)

        if self.balance_doa_classes:
            doa_class_weights = compute_class_weight_labels(
                labels=torch_to_numpy(doa_labels).reshape(-1), num_classes=self.n_bins
            )
            doa_class_weights = torch.tensor(doa_class_weights).to(doa_logits)
            doa_class_weights = doa_class_weights / sum(
                doa_class_weights
            )  # sum up to 1

            reg_loss = F.cross_entropy(
                doa_logits, doa_labels, weight=doa_class_weights, reduction="sum"
            )
        else:
            reg_loss = F.cross_entropy(doa_logits, doa_labels, reduction="sum")

        loss = self.loss_weights[0] * multi_label_loss + self.loss_weights[1] * reg_loss

        return loss, multi_label_loss, reg_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, sample_size, logging_output, net_output = self.step_forward_model(
            model, sample, reduce
        )

        logging_output = self.log_output(
            sample=sample, net_output=net_output, logging_output=logging_output
        )

        return loss, sample_size, logging_output

    def log_output(self, sample, net_output, logging_output):
        if self.report_accuracy and not self.training:
            with torch.no_grad():
                class_logits = net_output["class_encoder_out"].float()  # (B, T, N)
                reg_logits = net_output["regression_out"].float()  # (B, T, N, 3, BINS)
                class_labels = sample["sed_labels"].to(class_logits)  # (B, T, N)
                reg_targets = sample["doa_labels"].to(reg_logits)  # (B, T, 3N)

                class_probs = torch.sigmoid(class_logits.float())

                class_mask = class_probs > self.task.best_threshold
                # class_mask = class_labels > self.task.best_threshold

                # check if any event is detected
                if class_mask.any().item():
                    B, T, N = class_labels.shape
                    reg_targets_3d = reg_targets.reshape(
                        (B, T, self.doa_size, N)
                    ).transpose(3, 2)  # (B, T, N, 3)

                    doa_logits = reg_logits[class_mask]  # (E, 3)
                    reg_targets_mask = torch_to_numpy(
                        reg_targets_3d[class_mask]
                    )  # (E, 3)
                    doa_labels = self.task.doa_discretizer_tf.transform(
                        reg_targets_mask.reshape(-1, 1)
                    )  # (E*3,)
                    doa_labels = doa_labels.reshape(reg_targets_mask.shape)

                    doa_preds = torch_to_numpy(
                        F.softmax(doa_logits, dim=-1).argmax(-1)
                    )  # (E, 3)
                    # doa_preds = doa_labels.copy() # (E, 3)

                    reg_logits_mask = self.task.doa_discretizer_tf.inverse_transform(
                        doa_preds.reshape(-1, 1)
                    )  # (E*3,)

                    class_mask = torch_to_numpy(class_mask)
                    class_labels = torch_to_numpy(class_labels)
                    reg_targets = torch_to_numpy(reg_targets).astype(np.float32)
                    class_probs = torch_to_numpy(class_probs).astype(np.float32)

                    reg_logits_preds = np.zeros(reg_targets_3d.shape).astype(
                        np.float32
                    )  # (B, T, N, 3)
                    reg_logits_preds[class_mask] = reg_logits_mask.reshape(
                        doa_preds.shape
                    )
                    reg_logits_preds = reg_logits_preds.transpose((0, 1, 3, 2)).reshape(
                        (B, T, 3 * N)
                    )
                    reg_logits_preds = reg_logits_preds.astype(np.float32)  # (B, T, 3N)

                    TP, TN, FP, FN = self.compute_metrics(
                        preds=doa_preds.reshape(-1),
                        targets=doa_labels.reshape(-1),
                        labels=np.arange(self.n_bins).tolist(),
                    )
                    n_correct = np.sum(TP) + np.sum(TN)
                    total = n_correct + np.sum(FP) + np.sum(FN)

                    class_probs = reshape_3Dto2D(class_probs)
                    class_labels = reshape_3Dto2D(class_labels)
                    reg_logits = reshape_3Dto2D(reg_logits_preds)
                    reg_targets = reshape_3Dto2D(reg_targets)

                    logging_output["class_probs"] = class_probs
                    logging_output["class_labels"] = class_labels
                    logging_output["reg_logits"] = reg_logits
                    logging_output["reg_targets"] = reg_targets

                    logging_output["doa_n_correct"] = n_correct
                    logging_output["doa_total"] = total

                    logging_output["doa_TP"] = TP
                    logging_output["doa_TN"] = TN
                    logging_output["doa_FP"] = FP
                    logging_output["doa_FN"] = FN
                else:
                    # if no event is detected we set tp, tn, fp, fn to zero and regreession logits
                    logging_output["reg_logits"] = np.zeros(reg_targets.shape)

                    logging_output["doa_n_correct"] = 0
                    logging_output["doa_total"] = 0

                    logging_output["doa_TP"] = 0
                    logging_output["doa_TN"] = 0
                    logging_output["doa_FP"] = 0
                    logging_output["doa_FN"] = 0
        return logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        total = utils.item(sum(log.get("doa_total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("doa_total", total)
            n_correct = utils.item(
                sum(log.get("doa_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("doa_n_correct", n_correct)
            metrics.log_derived(
                "doa_accuracy",
                lambda meters: round(
                    meters["doa_n_correct"].sum * 100.0 / meters["doa_total"].sum, 3
                )
                if meters["doa_total"].sum > 0
                else float("nan"),
            )

            tp = np.sum([log.get("doa_TP", 0) for log in logging_outputs], axis=0)
            fp = np.sum([log.get("doa_FP", 0) for log in logging_outputs], axis=0)
            fn = np.sum([log.get("doa_FN", 0) for log in logging_outputs], axis=0)

            f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("doa_f1_score", f1_value * 100.0)


@register_criterion(
    "multitask_frame_cart_dcase_discretizer_gn",
    dataclass=MultitaskSeldCriterionDiscGradNormConfig,
)
class MultitaskSeldAudioFrameCartDiscretizerGradNormCriterion(
    MultitaskSeldAudioFrameCartDiscretizerCriterion,
    MultitaskSeldAudioFrameCartDcaseGradNormCriterion,
):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        loss_weights=(1, 1),
        doa_size=3,
        use_labels_mask=1.0,
        extend_mask=True,
        class_type="CE",
        gamma=2.0,
        clip=0,
        regr_type="mse",
        balance_doa_classes=False,
        grad_norm_alpha=0.12,
        grad_norm_scale=2.0,
        grad_norm_x_range=[-3, 3],
        num_updates=320000,
        skip_grad_norm_for_n_updates=0,
    ):
        MultitaskSeldAudioFrameCartDiscretizerCriterion.__init__(
            self,
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
            balance_doa_classes=balance_doa_classes,
        )

        MultitaskSeldAudioFrameCartDcaseGradNormCriterion.__init__(
            self,
            task=task,
            sentence_avg=sentence_avg,
            report_accuracy=report_accuracy,
            nb_classes=nb_classes,
            loss_weights=loss_weights,
            doa_size=doa_size,
            use_labels_mask=use_labels_mask,
            extend_mask=extend_mask,
            class_type=class_type,
            gamma=gamma,
            clip=clip,
            regr_type=regr_type,
            grad_norm_alpha=grad_norm_alpha,
            grad_norm_scale=grad_norm_scale,
            grad_norm_x_range=grad_norm_x_range,
            num_updates=num_updates,
            skip_grad_norm_for_n_updates=skip_grad_norm_for_n_updates,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if self.training or self.task.cfg.segment_eval:
            net_output = model(**sample["net_input"])
        else:
            net_output, sample = self.task.inference_step(sample, model)

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
            model.w2v_encoder.weights.data = self.grad_scale * F.softmax(
                model.w2v_encoder.weights.data, dim=-1
            )

        if isinstance(self.alpha, float):
            alpha = self.alpha
        else:
            alpha = self.sigmoid_schedule(current_step)

            metrics.log_scalar("alpha", alpha, weight=0, round=3, priority=10000)

        weights = model.w2v_encoder.weights.data.cpu()

        assert np.isclose(sum(weights), self.grad_scale, atol=1e-3), (
            f"{sum(weights)} != 2.0"
        )
        weights0, weights1 = weights[0], weights[1]

        metrics.log_scalar("weights0", weights0, weight=0, round=3, priority=10000)
        metrics.log_scalar("weights1", weights1, weight=0, round=3, priority=10000)

        # evaluate each task loss L_i(t)
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

        logging_output = self.log_output(
            sample=sample, net_output=net_output, logging_output=logging_output
        )

        return grad_norm_loss, sample_size, logging_output
