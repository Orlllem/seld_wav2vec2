import math
from dataclasses import dataclass, field

import numpy as np
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig)

EPS = 1e-5


@dataclass
class LabelSmoothedCrossEntropyF1CriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    nb_classes: int = field(
        default=3,
        metadata={"help": "number of classes"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_f1",
    dataclass=LabelSmoothedCrossEntropyF1CriterionConfig
)
class LabelSmoothedCrossEntropyF1Criterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        nb_classes=3,
    ):
        super().__init__(task,
                         sentence_avg,
                         label_smoothing,
                         ignore_prefix_size,
                         report_accuracy)

        self.nb_classes = nb_classes

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            TP, TN, FP, FN = self.compute_metrics(model, net_output, sample)
            logging_output["n_correct"] = utils.item(torch.sum(TP).data)
            logging_output["total"] = sample_size

            logging_output["TP"] = TP.tolist()
            logging_output["TN"] = TN.tolist()
            logging_output["FP"] = FP.tolist()
            logging_output["FN"] = FN.tolist()
        return loss, sample_size, logging_output

    def compute_metrics(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        preds = torch.exp(lprobs).argmax(1)

        confusion_matrix = torch.zeros(self.nb_classes, self.nb_classes)

        for t, p in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        TP = torch.diag(confusion_matrix)
        FP = confusion_matrix.sum(axis=0) - torch.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - torch.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        return TP, TN, FP, FN

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
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

            f1 = (2*tp + EPS) / (2*tp + fp + fn + EPS)

            f1_value = np.mean(f1)
            if not np.isnan(f1_value):
                metrics.log_scalar("f1_score", f1_value * 100.0)
