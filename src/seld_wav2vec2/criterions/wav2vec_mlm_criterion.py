import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.wav2vec_criterion import (
    Wav2vecCriterion,
    Wav2VecCriterionConfig,
)
from fairseq.utils import is_xla_tensor

from seld_wav2vec2.criterions.multi_label_regression import compute_class_weight_labels
from seld_wav2vec2.utils import torch_to_numpy

eps = torch.finfo(torch.float32).eps


def compute_cm_metrics(preds, target, nb_classes):
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    for t, p in zip(target.reshape(-1), preds.reshape(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    TP = torch.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - torch.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - torch.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    return TP, TN, FP, FN


@dataclass
class Wav2VecMlmCriterionConfig(Wav2VecCriterionConfig):
    loss_weights_mlm: Optional[Tuple[float, float]] = field(
        default=(1.0, 1.0),
        metadata={"help": "weights for loss terms"},
    )
    balance_mlm_classes: Optional[bool] = field(
        default=False,
        metadata={"help": "balance weights for mlm loss"},
    )


@register_criterion("wav2vec2_mlm", dataclass=Wav2VecMlmCriterionConfig)
class Wav2vecMlmCriterion(Wav2vecCriterion):
    def __init__(
        self,
        task,
        infonce=False,
        loss_weights=None,
        log_keys=None,
        loss_weights_mlm=(1.0, 1.0),
        balance_mlm_classes=False,
    ):
        super().__init__(task, infonce, loss_weights, log_keys)

        self.loss_weights_mlm = loss_weights_mlm
        self.balance_mlm_classes = balance_mlm_classes

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        self.xla = is_xla_tensor(logits)

        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        # Compute wav2vec 2.0 loss
        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            w2v_loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            w2v_loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample["net_input"]["mask_indices"]
                .transpose(0, 1)  # logits are transposed in `model.get_logits`
                .reshape(logits.size(0))
            )
            w2v_loss = (w2v_loss * mi).sum() if reduce else (w2v_loss * mi)

        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = (
                target.numel() if self.infonce else target.long().sum().detach()
            )
        losses.append(w2v_loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), (
                f"{len(extra_losses)}, {len(self.loss_weights)}"
            )
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    w2v_loss += p
                    losses.append(p)

        # Compute mlm loss
        mlm_targets = model.get_mlm_targets(net_output)
        mlm_logits = model.get_mlm_logits(net_output)

        num_classes = mlm_logits.shape[1]
        if self.balance_mlm_classes:
            class_weights = compute_class_weight_labels(
                labels=torch_to_numpy(mlm_targets).reshape(-1), num_classes=num_classes
            )
            class_weights = torch.tensor(class_weights).to(mlm_logits)
            class_weights = class_weights / sum(class_weights)  # sum up to 1
        else:
            class_weights = None

        mlm_loss = F.cross_entropy(
            mlm_logits,
            mlm_targets,
            reduction=reduction,
            ignore_index=self.padding_idx,
            weight=class_weights,
        )

        mlm_preds = mlm_logits.argmax(1)
        if model.cfg.latent_groups > 1:
            mlm_preds = model.joint_lm_labels(mlm_preds)
            mlm_targets = model.joint_lm_labels(mlm_targets)

        mask = mlm_targets.ne(self.padding_idx)
        n_correct = torch.sum(
            mlm_preds.masked_select(mask).eq(mlm_targets.masked_select(mask))
        )
        total = torch.sum(mask)

        loss = self.loss_weights_mlm[0] * w2v_loss + self.loss_weights_mlm[1] * mlm_loss

        logging_output = {
            "loss": loss.detach() if (reduce and not self.xla) else loss.detach(),
            "w2v_loss": w2v_loss.detach()
            if (reduce and not self.xla)
            else w2v_loss.detach(),
            "mlm_loss": mlm_loss.detach()
            if (reduce and not self.xla)
            else mlm_loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = target
                    logging_output["target"] = original_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, loss_i in enumerate(losses):
                logging_output[f"loss_{i}"] = (
                    loss_i.item() if not self.xla else loss_i.detach()
                )

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    if is_xla_tensor(logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().detach() - both.long().sum().detach()
                        count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        Wav2vecCriterion.reduce_metrics(logging_outputs)

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        w2v_loss_sum = utils.item(
            sum(log.get("w2v_loss", 0) for log in logging_outputs)
        )
        mlm_loss_sum = utils.item(
            sum(log.get("mlm_loss", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "w2v_loss",
            w2v_loss_sum / (sample_size or 1) / math.log(2),
            sample_size,
            round=3,
        )

        metrics.log_scalar(
            "mlm_loss",
            mlm_loss_sum / (sample_size or 1) / math.log(2),
            sample_size,
            round=3,
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "mlm_accuracy",
                lambda meters: round(meters["n_correct"].sum / meters["total"].sum, 3)
                if meters["total"].sum > 0
                else float("nan"),
            )


@dataclass
class Wav2VeclmCriterionConfig(Wav2VecCriterionConfig):
    loss_weights_lm: Optional[Tuple[float, float]] = field(
        default=(1.0, 1.0),
        metadata={"help": "weights for loss terms"},
    )


@register_criterion("wav2vec2_lm", dataclass=Wav2VeclmCriterionConfig)
class Wav2vecLMCriterion(Wav2vecCriterion):
    def __init__(
        self,
        task,
        infonce=False,
        loss_weights=None,
        log_keys=None,
        loss_weights_lm=(1.0, 1.0),
    ):
        super().__init__(task, infonce, loss_weights, log_keys)

        self.loss_weights_lm = loss_weights_lm

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        self.xla = is_xla_tensor(logits)

        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        # Compute wav2vec 2.0 loss
        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            w2v_loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            w2v_loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample["net_input"]["mask_indices"]
                .transpose(0, 1)  # logits are transposed in `model.get_logits`
                .reshape(logits.size(0))
            )
            w2v_loss = (w2v_loss * mi).sum() if reduce else (w2v_loss * mi)

        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = (
                target.numel() if self.infonce else target.long().sum().detach()
            )
        losses.append(w2v_loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), (
                f"{len(extra_losses)}, {len(self.loss_weights)}"
            )
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    w2v_loss += p
                    losses.append(p)

        # Compute autoregressive (GPT similar) loss
        lm_logits = model.get_lm_logits(net_output)
        lm_targets = model.get_lm_targets(net_output)

        lm_loss = F.cross_entropy(
            lm_logits,
            lm_targets,
            reduction=reduction,
            ignore_index=self.padding_idx,
        )

        mask = lm_targets.ne(self.padding_idx)
        n_correct = torch.sum(
            lm_logits.argmax(1).masked_select(mask).eq(lm_targets.masked_select(mask))
        )
        total = torch.sum(mask)

        loss = self.loss_weights_lm[0] * w2v_loss + self.loss_weights_lm[1] * lm_loss

        logging_output = {
            "loss": loss.detach() if (reduce and not self.xla) else loss.detach(),
            "w2v_loss": w2v_loss.detach()
            if (reduce and not self.xla)
            else w2v_loss.detach(),
            "lm_loss": lm_loss.detach()
            if (reduce and not self.xla)
            else lm_loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = target
                    logging_output["target"] = original_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, loss_i in enumerate(losses):
                logging_output[f"loss_{i}"] = (
                    loss_i.item() if not self.xla else loss_i.detach()
                )

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    if is_xla_tensor(logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().detach() - both.long().sum().detach()
                        count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        Wav2vecCriterion.reduce_metrics(logging_outputs)

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        w2v_loss_sum = utils.item(
            sum(log.get("w2v_loss", 0) for log in logging_outputs)
        )
        lm_loss_sum = utils.item(sum(log.get("lm_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "w2v_loss",
            w2v_loss_sum / (sample_size or 1) / math.log(2),
            sample_size,
            round=3,
        )

        metrics.log_scalar(
            "lm_loss",
            lm_loss_sum / (sample_size or 1) / math.log(2),
            sample_size,
            round=3,
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "lm_accuracy",
                lambda meters: round(meters["n_correct"].sum / meters["total"].sum, 3)
                if meters["total"].sum > 0
                else float("nan"),
            )
