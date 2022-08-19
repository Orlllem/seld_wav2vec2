import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

import seld_wav2vec2.criterions.cls_feature_class as cls_feature_class
import seld_wav2vec2.criterions.parameter as parameter
from seld_wav2vec2.criterions.evaluation_metrics import (
    compute_doa_scores_regr, compute_doa_scores_regr_xyz, compute_sed_scores,
    early_stopping_metric, er_overall_framewise, f1_overall_framewise)
from seld_wav2vec2.criterions.SELD_evaluation_metrics import SELDMetrics

eps = np.finfo(np.float32).eps

# label frame resolution (label_frame_res)
nb_label_frames_1s = 50  # 1/label_hop_len_s = 1/0.02
nb_label_frames_1s_100ms = 10  # 1/label_hop_len_s = 1/0.1


def schedule_weight(current_step, boundaries, values):
    boundaries.append(float('inf'))
    values.append(0)

    # check boundaries
    # first index in 'boundaries' greater than current_step
    s = next((x[0] for x in enumerate(boundaries) if x[1] > current_step), -1)
    return values[s-1]


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :,
                                                      nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = (np.sqrt(x**2 + y**2 + z**2) > 0.5).astype(float)

    return sed


def reshape_3Dto2D(array):
    '''
    Reshape 3D to 2D array (B,T,N) -> (B*T,N)
    '''
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2])


def cart2sph_array(array):
    '''
    Convert cartesian to spherical coordinates

    :param array x, y, z at dim -1
    :return: azi, ele stacked array in radians
    '''

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
            "predictions"},
    )
    extend_mask: Optional[bool] = field(
        default=True,
        metadata={
            "help": "When mask is extended the model must produced regression"
            "logits of (B, T, doa_size*N_classes)"},
    )
    label_hop_len_s: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "Label hop length in seconds"},
    )
    constrain_r_unit: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Constraint sqrt(x^2 + y^2 + z^2)=1"},
    )
    focal_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use focal loss"},
    )
    focal_alpha: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "focal loss alpha"},
    )
    focal_gamma: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "focal loss gamma"},
    )
    focal_bw: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use focal loss with batching wise"},
    )
    regr_type: Optional[str] = field(
        default="mse",
        metadata={
            "help": "regression loss type"},
    )


@dataclass
class MultitaskSedDoaScheduleCriterionConfig(MultitaskSedDoaCriterionConfig):
    boundaries: Tuple[float, ...] = field(
        default=(20000, 30000, 60000),
        metadata={
            "help": "boundaries of schedule weight for doa"
        },
    )
    weights_values: Tuple[float, ...] = field(
        default=(1.0, 11.0, 110.0),
        metadata={
            "help": "values of boundaries of schedule weight for doa"
        },
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
    "multitask_sed_doa_seqclass",
    dataclass=MultitaskSedDoaCriterionConfig
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
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                logits = net_output['class_encoder_out']
                targets = sample["sed_labels"]
                TP, TN, FP, FN = self.compute_metrics(logits,
                                                      targets)
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

        cm = multilabel_confusion_matrix(target.cpu().numpy(),
                                         preds,
                                         labels=self.labels)

        TN, FN, TP, FP = cm[:, 0, 0], cm[:, 1, 0], cm[:, 1, 1], cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True):

        class_logits = net_output['class_encoder_out']
        reg_logits = net_output['regression_out']

        class_labels = sample["sed_labels"].to(class_logits)
        reg_targets = sample["doa_labels"].to(reg_logits)

        if self.training:

            multi_label_loss = F.binary_cross_entropy_with_logits(class_logits,
                                                                  class_labels,
                                                                  reduction="sum").float()
            if self.use_labels_mask:
                class_mask = class_labels > 0.5
            else:
                class_mask = torch.sigmoid(class_logits) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits = reg_logits[class_mask_extended]
                reg_targets = reg_targets[class_mask_extended]
            else:
                B, N = class_labels.shape
                reg_logits = reg_logits.reshape(
                    (B, self.doa_size, N)).transpose(2, 1)
                reg_targets = reg_targets.reshape(
                    (B, self.doa_size, N)).transpose(2, 1)

                reg_logits = reg_logits[class_mask]
                reg_targets = reg_targets[class_mask]

            reg_loss = F.mse_loss(reg_logits, reg_targets,
                                  reduction='sum').float()

            loss = self.loss_weights[0] * multi_label_loss + \
                self.loss_weights[1] * reg_loss

        else:

            # inference-time
            multi_label_loss = F.binary_cross_entropy_with_logits(class_logits,
                                                                  class_labels,
                                                                  reduction="sum").float()

            class_mask = torch.sigmoid(class_logits.float()) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits = reg_logits[class_mask_extended]
                reg_targets = reg_targets[class_mask_extended]
            else:
                B, N = class_labels.shape
                reg_logits = reg_logits.reshape(
                    (B, self.doa_size, N)).transpose(2, 1)
                reg_targets = reg_targets.reshape(
                    (B, self.doa_size, N)).transpose(2, 1)

                reg_logits = reg_logits[class_mask]
                reg_targets = reg_targets[class_mask]

            reg_loss = F.mse_loss(reg_logits, reg_targets,
                                  reduction='sum').float()

            loss = multi_label_loss + reg_loss

        return loss, multi_label_loss, reg_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "multi_label_loss_sum",
            multi_label_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=3
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

            f1 = (2*tp + eps) / (2*tp + fp + fn + eps)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("f1_score", f1_value * 100.0)


@register_criterion("multitask_sed_doa_seqclass_cart_dcase_2019",
                    dataclass=MultitaskSedDoaCriterionConfig)
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
        super().__init__(task, sentence_avg, report_accuracy, nb_classes,
                         loss_weights, doa_size, use_labels_mask, extend_mask)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                class_logits = net_output['class_encoder_out'].float()
                reg_logits = net_output['regression_out'].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits.float())
                class_mask = class_probs > 0.5
                class_preds = class_mask.float()

                # ignore padded labels -100
                class_pad_mask = class_labels < 0
                class_labels[class_pad_mask] = torch.tensor(0).to(class_labels)

                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits[~class_mask_extended] = torch.tensor(
                    0.0).to(reg_targets)
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
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0)
                           for log in logging_outputs)

        y_pred_class = np.concatenate([log.get("y_pred_class", 0)
                                       for log in logging_outputs], axis=0)
        y_true_class = np.concatenate([log.get("y_true_class", 0)
                                       for log in logging_outputs], axis=0)
        y_pred_reg = np.concatenate([log.get("y_pred_reg", 0)
                                    for log in logging_outputs], axis=0)
        y_true_reg = np.concatenate([log.get("y_true_reg", 0)
                                    for log in logging_outputs], axis=0)

        er_metric = compute_doa_scores_regr_xyz(y_pred_reg, y_true_reg,
                                                y_pred_class, y_true_class)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        doa_metric = [_doa_err, _frame_recall]

        sed_metric = compute_sed_scores(
            y_pred_class, y_true_class, nb_label_frames_1s_100ms)
        _er = sed_metric[0]
        _f = sed_metric[1]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0)
                          for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum", multi_label_loss_sum / ntokens / math.log(2), ntokens, round=5
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=5
        )

        metrics.log_scalar("f1_score", _f * 100, round=5)
        metrics.log_scalar("doa_error", _doa_err, round=5)
        metrics.log_scalar("frame_recall", _frame_recall*100, round=5)
        if np.isnan(_er):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", _er*100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)


@register_criterion(
    "multitask_sed_doa_audio_frame_class",
    dataclass=MultitaskSedDoaCriterionConfig
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
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_bw=False,
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

        self.use_labels_mask = use_labels_mask
        self.extend_mask = extend_mask
        self.constrain_r_unit = constrain_r_unit

        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_bw = focal_bw

        if regr_type == "logcosh":
            self.regr_loss = LogCoshLoss()
        else:
            self.regr_loss = nn.MSELoss(reduction="sum")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                logits = net_output['class_encoder_out']
                targets = sample["sed_labels"]

                # ignore padded labels -100
                class_pad_mask = targets < 0
                targets[class_pad_mask] = torch.tensor(0).to(targets.device)
                TP, TN, FP, FN = self.compute_metrics(logits,
                                                      targets)
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
            cm = multilabel_confusion_matrix(targets[i],
                                             preds[i], labels=self.labels)

            TN += cm[:, 0, 0]
            FN += cm[:, 1, 0]
            TP += cm[:, 1, 1]
            FP += cm[:, 0, 1]

        return TP, TN, FP, FN

    def compute_loss(self, net_output, sample, reduce=True):

        class_logits = net_output['class_encoder_out']
        reg_logits = net_output['regression_out']

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
                    class_weights = class_weight.compute_class_weight(
                        'balanced',
                        np.unique(class_labels_1d),
                        class_labels_1d)
                    focal_alpha = class_weights[1]/sum(class_weights)
                else:
                    focal_alpha = self.focal_alpha
                multi_label_loss = sigmoid_focal_loss(class_logits,
                                                      class_labels,
                                                      alpha=focal_alpha,
                                                      gamma=self.focal_gamma,
                                                      reduction="sum").float()
            else:
                multi_label_loss = F.binary_cross_entropy_with_logits(class_logits,
                                                                      class_labels,
                                                                      weight=weights,
                                                                      reduction="sum").float()

            if self.use_labels_mask:
                class_mask = class_labels > 0.5
            else:
                class_mask = torch.sigmoid(class_logits) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask,
                                          reg_targets_mask).float()

                if self.constrain_r_unit:
                    B, T, N = class_labels.shape
                    reg_logits = reg_logits.reshape(
                        (B, T, self.doa_size, N)).transpose(3, 2)
                    reg_targets = reg_targets.reshape(
                        (B, T, self.doa_size, N)).transpose(3, 2)

                    reg_logits_mask = reg_logits[class_mask]
                    reg_targets_mask = reg_targets[class_mask]

                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(reg_norm,
                                               torch.ones(reg_norm.shape).to(
                                                   reg_norm),
                                               reduction='sum').float()
                    reg_loss = reg_loss + reg_unit_loss
            else:

                B, T, N = class_labels.shape
                reg_logits = reg_logits.reshape(
                    (B, T, self.doa_size, N)).transpose(3, 2)
                reg_targets = reg_targets.reshape(
                    (B, T, self.doa_size, N)).transpose(3, 2)

                reg_logits_mask = reg_logits[class_mask]
                reg_targets_mask = reg_targets[class_mask]

                reg_loss = self.regr_loss(reg_logits_mask,
                                          reg_targets_mask).float()

                if self.constrain_r_unit:
                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(reg_norm,
                                               torch.ones(reg_norm.shape).to(
                                                   reg_norm),
                                               reduction='sum').float()
                    reg_loss = reg_loss + reg_unit_loss

            loss = self.loss_weights[0] * multi_label_loss + \
                self.loss_weights[1] * reg_loss

        else:

            # inference-time
            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            if self.focal_loss:
                multi_label_loss = sigmoid_focal_loss(class_logits,
                                                      class_labels,
                                                      alpha=self.focal_alpha,
                                                      gamma=self.focal_gamma,
                                                      reduction="sum").float()
            else:
                multi_label_loss = F.binary_cross_entropy_with_logits(class_logits,
                                                                      class_labels,
                                                                      reduction="sum").float()

            class_mask = torch.sigmoid(class_logits) > 0.5

            if self.extend_mask:
                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits_mask = reg_logits[class_mask_extended]
                reg_targets_mask = reg_targets[class_mask_extended]

                reg_loss = self.regr_loss(reg_logits_mask,
                                          reg_targets_mask).float()

                if self.constrain_r_unit:
                    B, T, N = class_labels.shape
                    reg_logits = reg_logits.reshape(
                        (B, T, self.doa_size, N)).transpose(3, 2)
                    reg_targets = reg_targets.reshape(
                        (B, T, self.doa_size, N)).transpose(3, 2)

                    reg_logits_mask = reg_logits[class_mask]
                    reg_targets_mask = reg_targets[class_mask]

                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(reg_norm,
                                               torch.ones(reg_norm.shape).to(
                                                   reg_norm),
                                               reduction='sum').float()
                    reg_loss = reg_loss + reg_unit_loss
            else:

                B, T, N = class_labels.shape
                reg_logits = reg_logits.reshape(
                    (B, T, self.doa_size, N)).transpose(3, 2)
                reg_targets = reg_targets.reshape(
                    (B, T, self.doa_size, N)).transpose(3, 2)

                reg_logits_mask = reg_logits[class_mask]
                reg_targets_mask = reg_targets[class_mask]

                reg_loss = self.regr_loss(reg_logits_mask,
                                          reg_targets_mask).float()

                if self.constrain_r_unit:
                    reg_norm = vector_norm(reg_logits_mask, dim=-1)
                    reg_unit_loss = F.mse_loss(reg_norm,
                                               torch.ones(reg_norm.shape).to(
                                                   reg_norm),
                                               reduction='sum').float()
                    reg_loss = reg_loss + reg_unit_loss

            loss = multi_label_loss + reg_loss

        return loss, multi_label_loss, reg_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum", multi_label_loss_sum / ntokens / math.log(2), ntokens, round=5
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=5
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

            f1 = (2*tp + eps) / (2*tp + fp + fn + eps)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("f1_score", f1_value * 100.0)


@register_criterion("multitask_sed_doa_audio_frame_class_cart_dcase_2020",
                    dataclass=MultitaskSedDoaCriterionConfig)
class MultitaskSeldAudioFrameCartDcase2020Criterion(MultitaskSeldAudioFrameCriterion):
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
        label_hop_len_s=0.02,
    ):
        super().__init__(task, sentence_avg, report_accuracy, nb_classes,
                         loss_weights, doa_size, use_labels_mask, extend_mask,
                         constrain_r_unit, focal_loss, focal_alpha, focal_gamma,
                         focal_bw, regr_type)

        params = parameter.get_params()

        params['fs'] = 16000
        params['label_hop_len_s'] = label_hop_len_s

        self.feat_cls = cls_feature_class.FeatureClass(params)
        self.cls_new_metric = SELDMetrics(nb_classes=nb_classes)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                class_logits = net_output['class_encoder_out'].float()
                reg_logits = net_output['regression_out'].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits.float())
                class_mask = class_probs > 0.5
                class_preds = class_mask.float()

                # ignore padded labels -100
                class_pad_mask = class_labels < 0
                class_labels[class_pad_mask] = torch.tensor(0).to(class_labels)

                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits[~class_mask_extended] = torch.tensor(
                    0.0).to(reg_targets)
                reg_logits = reg_logits.cpu().numpy()

                class_preds = class_preds.cpu().numpy()
                class_labels = class_labels.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                sed_pred = reshape_3Dto2D(class_preds)
                sed_gt = reshape_3Dto2D(class_labels)

                doa_pred = reshape_3Dto2D(reg_logits)
                doa_gt = reshape_3Dto2D(reg_targets)

                pred_dict = self.feat_cls.regression_label_format_to_output_format(
                    sed_pred, doa_pred
                )
                gt_dict = self.feat_cls.regression_label_format_to_output_format(
                    sed_gt, doa_gt
                )

                pred_blocks_dict = self.feat_cls.segment_labels(
                    pred_dict, sed_pred.shape[0])
                gt_blocks_dict = self.feat_cls.segment_labels(
                    gt_dict, sed_gt.shape[0])

                self.cls_new_metric.update_seld_scores_xyz(
                    pred_blocks_dict, gt_blocks_dict)

                logging_output["TP"] = self.cls_new_metric._TP
                logging_output["FP"] = self.cls_new_metric._FP
                logging_output["TN"] = self.cls_new_metric._TN
                logging_output["FN"] = self.cls_new_metric._FN

                logging_output["S"] = self.cls_new_metric._S
                logging_output["D"] = self.cls_new_metric._D
                logging_output["I"] = self.cls_new_metric._I

                logging_output["Nref"] = self.cls_new_metric._Nref
                logging_output["Nsys"] = self.cls_new_metric._Nsys

                logging_output["total_DE"] = self.cls_new_metric._total_DE
                logging_output["DE_TP"] = self.cls_new_metric._DE_TP

                # clear metrics
                self.cls_new_metric._TP = 0
                self.cls_new_metric._FP = 0
                self.cls_new_metric._TN = 0
                self.cls_new_metric._FN = 0
                self.cls_new_metric._S = 0
                self.cls_new_metric._D = 0
                self.cls_new_metric._I = 0
                self.cls_new_metric._Nref = 0
                self.cls_new_metric._Nsys = 0
                self.cls_new_metric._total_DE = 0
                self.cls_new_metric._DE_TP = 0

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0)
                           for log in logging_outputs)

        _TP = sum([log.get("TP", 0) for log in logging_outputs])

        _S = sum([log.get("S", 0) for log in logging_outputs])
        _D = sum([log.get("D", 0) for log in logging_outputs])
        _I = sum([log.get("I", 0) for log in logging_outputs])

        _Nref = sum([log.get("Nref", 0) for log in logging_outputs])
        _Nsys = sum([log.get("Nsys", 0) for log in logging_outputs])

        _total_DE = sum([log.get("total_DE", 0) for log in logging_outputs])
        _DE_TP = sum([log.get("DE_TP", 0) for log in logging_outputs])

        # Location-senstive detection performance
        ER = (_S + _D + _I) / float(_Nref + eps)

        prec = float(_TP) / float(_Nsys + eps)
        recall = float(_TP) / float(_Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        if _DE_TP:
            DE = _total_DE / float(_DE_TP + eps)
        else:
            # When the total number of prediction is zero
            DE = 180

        DE_prec = float(_DE_TP) / float(_Nsys + eps)
        DE_recall = float(_DE_TP) / float(_Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        sed_metric = [ER, F]
        doa_metric = [DE, DE_F]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0)
                          for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum", multi_label_loss_sum / ntokens / math.log(2), ntokens, round=5
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=5
        )

        metrics.log_scalar("f1_score", F * 100, round=5)
        metrics.log_scalar("doa_error", DE, round=5)
        metrics.log_scalar("frame_recall", DE_F*100, round=5)
        if np.isnan(ER):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", ER*100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)


@register_criterion("multitask_sed_doa_audio_frame_class_cart_dcase_2019",
                    dataclass=MultitaskSedDoaCriterionConfig)
class MultitaskSeldAudioFrameCartDcase2019Criterion(MultitaskSeldAudioFrameCriterion):
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
        super().__init__(task, sentence_avg, report_accuracy, nb_classes,
                         loss_weights, doa_size, use_labels_mask, extend_mask,
                         constrain_r_unit, focal_loss, focal_alpha, focal_gamma,
                         focal_bw, regr_type)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                class_logits = net_output['class_encoder_out'].float()
                reg_logits = net_output['regression_out'].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits.float())
                class_mask = class_probs > 0.5
                class_preds = class_mask.float()

                # ignore padded labels -100
                class_pad_mask = class_labels < 0
                class_labels[class_pad_mask] = torch.tensor(0).to(class_labels)

                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits[~class_mask_extended] = torch.tensor(
                    0.0).to(reg_targets)
                reg_logits = reg_logits.cpu().numpy()

                class_preds = class_preds.cpu().numpy()
                class_labels = class_labels.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                y_pred_class = reshape_3Dto2D(class_preds)
                y_true_class = reshape_3Dto2D(class_labels)

                y_pred_reg = reshape_3Dto2D(reg_logits)
                y_true_reg = reshape_3Dto2D(reg_targets)

                logging_output["y_pred_class"] = y_pred_class
                logging_output["y_true_class"] = y_true_class
                logging_output["y_pred_reg"] = y_pred_reg
                logging_output["y_true_reg"] = y_true_reg

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0)
                           for log in logging_outputs)

        y_pred_class = np.concatenate([log.get("y_pred_class", 0)
                                       for log in logging_outputs], axis=0)
        y_true_class = np.concatenate([log.get("y_true_class", 0)
                                       for log in logging_outputs], axis=0)
        y_pred_reg = np.concatenate([log.get("y_pred_reg", 0)
                                    for log in logging_outputs], axis=0)
        y_true_reg = np.concatenate([log.get("y_true_reg", 0)
                                    for log in logging_outputs], axis=0)

        er_metric = compute_doa_scores_regr_xyz(y_pred_reg, y_true_reg,
                                                y_pred_class, y_true_class)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        doa_metric = [_doa_err, _frame_recall]

        sed_metric = compute_sed_scores(
            y_pred_class, y_true_class, nb_label_frames_1s)
        _er = sed_metric[0]
        _f = sed_metric[1]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0)
                          for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum", multi_label_loss_sum / ntokens / math.log(2), ntokens, round=5
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=5
        )

        metrics.log_scalar("f1_score", _f * 100, round=5)
        metrics.log_scalar("doa_error", _doa_err, round=5)
        metrics.log_scalar("frame_recall", _frame_recall*100, round=5)
        if np.isnan(_er):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", _er*100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)


@register_criterion("multitask_sed_doa_audio_frame_class_cart_dcase_2019_doa_schedule",
                    dataclass=MultitaskSedDoaScheduleCriterionConfig)
class MultitaskSeldAudioFrameCartDcase2019ScheduleCriterion(MultitaskSeldAudioFrameCartDcase2019Criterion):
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
        super().__init__(task, sentence_avg, report_accuracy, nb_classes,
                         loss_weights, doa_size, use_labels_mask, extend_mask,
                         constrain_r_unit, focal_loss, focal_alpha, focal_gamma,
                         focal_bw)

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
        reg_weight = schedule_weight(
            current_step, self.boundaries, self.weights_values)

        self.loss_weights[1] = reg_weight

        loss, sample_size, logging_output = super().forward(model, sample, reduce)

        logging_output["reg_weight"] = reg_weight

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        reg_weight = [log.get("reg_weight", 0) for log in logging_outputs]

        metrics.log_scalar("reg_weight", sum(reg_weight) / len(reg_weight),
                           len(reg_weight), round=3)


@register_criterion("multitask_sed_doa_audio_frame_class_cart_tut_2018",
                    dataclass=MultitaskSedDoaCriterionConfig)
class MultitaskSeldAudioFrameCartCriterion(MultitaskSeldAudioFrameCriterion):
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
        super().__init__(task, sentence_avg, report_accuracy, nb_classes,
                         loss_weights, doa_size, use_labels_mask, extend_mask,
                         constrain_r_unit, focal_loss, focal_alpha, focal_gamma,
                         focal_bw, regr_type)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, multi_label_loss, reg_loss = self.compute_loss(
            net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
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
                class_logits = net_output['class_encoder_out'].float()
                reg_logits = net_output['regression_out'].float()

                class_labels = sample["sed_labels"].float()
                reg_targets = sample["doa_labels"].float()

                class_probs = torch.sigmoid(class_logits.float())
                class_mask = class_probs > 0.5
                class_preds = class_mask.float()

                # ignore padded labels -100
                class_pad_mask = class_labels < 0
                class_labels[class_pad_mask] = torch.tensor(0).to(class_labels)

                class_mask_extended = torch.cat(
                    [class_mask]*self.doa_size, dim=-1)

                reg_logits[~class_mask_extended] = torch.tensor(
                    0.0).to(reg_targets)
                reg_logits = reg_logits.cpu().numpy()

                class_preds = class_preds.cpu().numpy()
                class_labels = class_labels.cpu().numpy()

                B, T, N = class_labels.shape
                reg_logits = reg_logits.reshape((B, T, N, self.doa_size))
                reg_targets = reg_targets.reshape(
                    reg_logits.shape).cpu().numpy()

                reg_logits_rad = cart2sph_array(reg_logits)
                reg_targets_rad = cart2sph_array(reg_targets)

                y_pred_class = reshape_3Dto2D(class_preds)
                y_true_class = reshape_3Dto2D(class_labels)

                y_pred_reg = reshape_3Dto2D(reg_logits_rad)
                y_true_reg = reshape_3Dto2D(reg_targets_rad)

                logging_output["y_pred_class"] = y_pred_class
                logging_output["y_true_class"] = y_true_class
                logging_output["y_pred_reg"] = y_pred_reg
                logging_output["y_true_reg"] = y_true_reg

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        multi_label_loss_sum = sum(log.get("multi_label_loss", 0)
                                   for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0)
                           for log in logging_outputs)

        y_pred_class = np.concatenate([log.get("y_pred_class", 0)
                                       for log in logging_outputs], axis=0)
        y_true_class = np.concatenate([log.get("y_true_class", 0)
                                       for log in logging_outputs], axis=0)
        y_pred_reg = np.concatenate([log.get("y_pred_reg", 0)
                                    for log in logging_outputs], axis=0)
        y_true_reg = np.concatenate([log.get("y_true_reg", 0)
                                    for log in logging_outputs], axis=0)

        er_metric = compute_doa_scores_regr(y_pred_reg, y_true_reg,
                                            y_pred_class, y_true_class)

        _doa_err, _frame_recall, _, _, _, _ = er_metric
        _er = er_overall_framewise(y_pred_class, y_true_class)
        _f = f1_overall_framewise(y_pred_class, y_true_class)
        _seld_scr = early_stopping_metric(
            [_er, _f], [_doa_err, _frame_recall])

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0)
                          for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "multi_label_loss_sum", multi_label_loss_sum / ntokens / math.log(2), ntokens, round=5
        )
        metrics.log_scalar(
            "reg_loss_sum", reg_loss_sum / ntokens / math.log(2), ntokens,
            round=5
        )

        metrics.log_scalar("f1_score", _f * 100, round=5)
        metrics.log_scalar("doa_error", _doa_err, round=5)
        metrics.log_scalar("frame_recall", _frame_recall*100, round=5)
        if np.isnan(_er):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", _er*100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)


@register_criterion("acc_doa_audio_frame_class_cart_dcase_2020",
                    dataclass=AccDoataskSedDoaCriterionConfig)
class AccDoataskSeldAudioFrameCartDcase2020Criterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=True,
        nb_classes=11,
        doa_size=3,
    ):
        super().__init__(task)

        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.nb_classes = nb_classes
        self.doa_size = doa_size

        self.labels = np.arange(nb_classes)

        params = parameter.get_params()

        params['fs'] = 16000
        params['label_hop_len_s'] = 0.02  # 20ms

        self.feat_cls = cls_feature_class.FeatureClass(params)
        self.cls_new_metric = SELDMetrics(nb_classes=nb_classes)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(net_output, sample, reduce=reduce)
        sample_size = (
            sample["sed_labels"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["sed_labels"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            with torch.no_grad():

                reg_logits = net_output['regression_out'].float()
                reg_targets = sample["doa_labels"].float()
                class_labels = sample["sed_labels"].float().cpu().numpy()

                reg_logits = reg_logits.cpu().numpy()
                reg_targets = reg_targets.cpu().numpy()

                class_preds = get_accdoa_labels(
                    reg_logits, nb_classes=self.nb_classes)

                sed_pred = reshape_3Dto2D(class_preds)
                sed_gt = reshape_3Dto2D(class_labels)

                doa_pred = reshape_3Dto2D(reg_logits)
                doa_gt = reshape_3Dto2D(reg_targets)

                pred_dict = self.feat_cls.regression_label_format_to_output_format(
                    sed_pred, doa_pred
                )
                gt_dict = self.feat_cls.regression_label_format_to_output_format(
                    sed_gt, doa_gt
                )

                pred_blocks_dict = self.feat_cls.segment_labels(
                    pred_dict, sed_pred.shape[0])
                gt_blocks_dict = self.feat_cls.segment_labels(
                    gt_dict, sed_gt.shape[0])

                self.cls_new_metric.update_seld_scores_xyz(
                    pred_blocks_dict, gt_blocks_dict)

                logging_output["TP"] = self.cls_new_metric._TP
                logging_output["FP"] = self.cls_new_metric._FP
                logging_output["TN"] = self.cls_new_metric._TN
                logging_output["FN"] = self.cls_new_metric._FN

                logging_output["S"] = self.cls_new_metric._S
                logging_output["D"] = self.cls_new_metric._D
                logging_output["I"] = self.cls_new_metric._I

                logging_output["Nref"] = self.cls_new_metric._Nref
                logging_output["Nsys"] = self.cls_new_metric._Nsys

                logging_output["total_DE"] = self.cls_new_metric._total_DE
                logging_output["DE_TP"] = self.cls_new_metric._DE_TP

                # clear metrics
                self.cls_new_metric._TP = 0
                self.cls_new_metric._FP = 0
                self.cls_new_metric._TN = 0
                self.cls_new_metric._FN = 0
                self.cls_new_metric._S = 0
                self.cls_new_metric._D = 0
                self.cls_new_metric._I = 0
                self.cls_new_metric._Nref = 0
                self.cls_new_metric._Nsys = 0
                self.cls_new_metric._total_DE = 0
                self.cls_new_metric._DE_TP = 0

        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):

        reg_logits = net_output['regression_out']
        reg_targets = sample["doa_labels"].to(reg_logits)

        loss = F.mse_loss(reg_logits, reg_targets, reduction='sum').float()

        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        _TP = sum([log.get("TP", 0) for log in logging_outputs])

        _S = sum([log.get("S", 0) for log in logging_outputs])
        _D = sum([log.get("D", 0) for log in logging_outputs])
        _I = sum([log.get("I", 0) for log in logging_outputs])

        _Nref = sum([log.get("Nref", 0) for log in logging_outputs])
        _Nsys = sum([log.get("Nsys", 0) for log in logging_outputs])

        _total_DE = sum([log.get("total_DE", 0) for log in logging_outputs])
        _DE_TP = sum([log.get("DE_TP", 0) for log in logging_outputs])

        # Location-senstive detection performance
        ER = (_S + _D + _I) / float(_Nref + eps)

        prec = float(_TP) / float(_Nsys + eps)
        recall = float(_TP) / float(_Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        if _DE_TP:
            DE = _total_DE / float(_DE_TP + eps)
        else:
            # When the total number of prediction is zero
            DE = 180

        DE_prec = float(_DE_TP) / float(_Nsys + eps)
        DE_recall = float(_DE_TP) / float(_Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        sed_metric = [ER, F]
        doa_metric = [DE, DE_F]

        _seld_scr = early_stopping_metric(sed_metric, doa_metric)

        # ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0)
                          for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )

        metrics.log_scalar("f1_score", F * 100, round=5)
        metrics.log_scalar("doa_error", DE, round=5)
        metrics.log_scalar("frame_recall", DE_F*100, round=5)
        if np.isnan(ER):
            metrics.log_scalar("error_rate", 100, round=5)
            metrics.log_scalar("seld_score", 1, round=5)
        else:
            metrics.log_scalar("error_rate", ER*100, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)
            metrics.log_scalar("seld_score", _seld_scr, round=5)
