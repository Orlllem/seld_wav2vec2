import json
import logging
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple

import fairseq
import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.utils import move_to_cuda
from omegaconf import OmegaConf, open_dict
from seld_wav2vec2.criterions.ccls_feature_class import FeatureClass
from torch.utils.data import DataLoader

import seld_wav2vec2.criterions.parameter as parameter
from seld_wav2vec2.criterions.evaluation_metrics import early_stopping_metric
from seld_wav2vec2.criterions.multi_label_regression import reshape_3Dto2D
from seld_wav2vec2.criterions.SELD_evaluation_metrics import SELDMetrics

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate-2020")

SAMPLE_RATE = 16000


def evaluate(
    ckpt_path,
    subset,
    batch_size,
    output_file_path,
    user_dir,
    doa_size=3,
    label_hop_len_s=None,
    data=None,
    labels=None,
    thr=0.5,
    use_best_threshold=False,
    segment_in_seconds=True,
    evaluate_at_100ms=False,
    segment_window=None,
    segment_stride=None,
):
    Arg = namedtuple("Arg", ["user_dir"])
    arg = Arg(user_dir.__str__())
    fairseq.utils.import_user_module(arg)

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval().cuda()

    if data is not None:
        task.cfg.data = data

    if labels is not None:
        task.cfg.labels = labels

    if label_hop_len_s is not None:
        task.cfg.label_hop_len_s = label_hop_len_s

    if segment_in_seconds:
        logger.info("Using segment window and stride in seconds.")
    else:
        logger.info("Using segment window and stride in number of labels.")
        OmegaConf.set_struct(task.cfg, True)
        with open_dict(task.cfg):
            task.cfg.segment_in_seconds = False

        task.cfg.segment_window = segment_window
        task.cfg.segment_stride = segment_stride

    model.w2v_encoder.apply_mask = False
    task.cfg.audio_augm = False
    task.cfg.random_crop = False
    task.cfg.max_sample_size = None

    if evaluate_at_100ms:
        logger.info("Evaluating at 100ms hop length.")
        task.cfg.align_outputs_frames = True
        model.w2v_encoder.align_100ms = True
        model.w2v_encoder.pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)
        model.w2v_encoder.mask_pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)

    task.load_dataset(subset)

    if use_best_threshold:
        thr = task.best_threshold
        logger.info(f"evaluating using best_threshold: {thr}")

    params = parameter.get_params()

    params["fs"] = SAMPLE_RATE
    params["label_hop_len_s"] = task.cfg.label_hop_len_s  # 20ms or 100ms

    feat_cls = FeatureClass(params)
    cls_new_metric = SELDMetrics(nb_classes=task.cfg.nb_classes)

    logger.info(task.cfg)

    logger.info("iterating over data")
    batch_itr = DataLoader(
        task.dataset(subset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task.dataset(subset).collater,
    )

    for batch in batch_itr:
        with torch.no_grad():
            batch = move_to_cuda(batch)

            net_output, batch = task.inference_step(batch, model)

            class_logits = net_output["class_encoder_out"].float()
            reg_logits = net_output["regression_out"].float()

            class_probs = torch.sigmoid(class_logits.float())

            class_mask = class_probs > thr

            class_preds = class_mask.float()

            class_labels = batch["sed_labels"].float()
            reg_targets = batch["doa_labels"].float()

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels.device)

            class_mask_extended = torch.cat([class_mask] * doa_size, dim=-1)

            reg_logits[~class_mask_extended] = torch.tensor(0.0).to(reg_targets)
            reg_logits = reg_logits.cpu().numpy()
            reg_targets = reg_targets.cpu().numpy()

            class_preds = class_preds.cpu().numpy()
            class_labels = class_labels.cpu().numpy()

            sed_pred = reshape_3Dto2D(class_preds)
            sed_gt = reshape_3Dto2D(class_labels)

            doa_pred = reshape_3Dto2D(reg_logits)
            doa_gt = reshape_3Dto2D(reg_targets)

            pred_dict = feat_cls.regression_label_format_to_output_format(
                sed_pred, doa_pred
            )
            gt_dict = feat_cls.regression_label_format_to_output_format(sed_gt, doa_gt)

            pred_blocks_dict = feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
            gt_blocks_dict = feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

            cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)

    logger.info("evaluating ...")

    er, f, de, de_f = cls_new_metric.compute_seld_scores()

    seld_scr = early_stopping_metric([er, f], [de, de_f])

    logger.info(f"f1: {f}")
    logger.info(f"er: {er}")
    logger.info(f"doa: {de}")
    logger.info(f"recall: {de_f}")
    logger.info(f"seld score: {seld_scr}")

    results = {
        "f1_mean": str(f),
        "er_mean": str(er),
        "doa_error": str(de),
        "frame_recall": str(de_f),
        "seld_score": str(seld_scr),
    }

    # Creates Output folder if it doesn't exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path", required=True, help="path to model checkpoint file."
    )
    parser.add_argument("--subset", default="test", help="subset to evaluate.")
    parser.add_argument(
        "--batch_size", default=64, type=int, help="max number of tokens in each batch."
    )
    parser.add_argument(
        "--output_file_path",
        default="reports/metrics.json",
        type=str,
        help=("path and name of output metrics file."),
    )
    parser.add_argument(
        "--user_dir",
        default="src/seld_wav2vec2",
        type=str,
        help=("path to a python module containing custom", "extensions."),
    )
    parser.add_argument("--doa_size", default=3, type=int, help="Size of DOA.")
    parser.add_argument(
        "--label_hop_len_s", default=None, type=float, help="hop len of labels"
    )
    parser.add_argument("--data", type=str, help=("path to the data."))
    parser.add_argument("--labels", type=str, default="labels", help="labels path")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold to be used in evaluation.")
    parser.add_argument("--use_best_threshold", action="store_true")
    parser.add_argument("--segment_in_seconds", action="store_true")
    parser.add_argument(
        "--not_segment_in_seconds", dest="segment_in_seconds", action="store_false"
    )
    parser.add_argument(
        "--segment_window",
        default=None,
        type=int,
        help="segment window size (in seconds or number of samples).",
    )
    parser.add_argument(
        "--segment_stride",
        default=None,
        type=int,
        help="segment_stride size (in seconds or number of samples).",
    )
    parser.add_argument("--evaluate_at_100ms", action="store_true")

    args = parser.parse_args()
    evaluate(
        args.ckpt_path,
        args.subset,
        args.batch_size,
        args.output_file_path,
        args.user_dir,
        doa_size=args.doa_size,
        label_hop_len_s=args.label_hop_len_s,
        data=args.data,
        labels=args.labels,
        thr=args.threshold,
        use_best_threshold=args.use_best_threshold,
        segment_in_seconds=args.segment_in_seconds,
        segment_window=args.segment_window,
        segment_stride=args.segment_stride,
        evaluate_at_100ms=args.evaluate_at_100ms,
    )
