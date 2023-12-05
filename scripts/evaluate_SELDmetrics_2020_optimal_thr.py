import logging
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple

import fairseq
import numpy as np
import torch
from fairseq import checkpoint_utils
from fairseq.utils import move_to_cuda
from torch.utils.data import DataLoader

from seld_wav2vec2.criterions.evaluation_metrics import (
    compute_doa_scores_regr_xyz, compute_sed_scores, early_stopping_metric)
from seld_wav2vec2.criterions.multi_label_regression import (
    nb_label_frames_1s, reshape_3Dto2D)

from .evaluate_SELDmetrics_2019 import evaluate

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")


def evaluate_list_thr(ckpt_path, batch_size, output_path, user_dir,
                      doa_size=3, data=None):

    Arg = namedtuple("Arg", ["user_dir"])
    arg = Arg(user_dir.__str__())
    fairseq.utils.import_user_module(arg)

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([
                                                                    ckpt_path])
    model = models[0].eval().cuda()

    if data is not None:
        task.cfg.data = data

    model.w2v_encoder.apply_mask = False
    task.cfg.audio_augm = False
    task.cfg.random_crop = False
    task.load_dataset("valid")

    logger.info("iterating over data")
    batch_itr = DataLoader(task.dataset("valid"),
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=task.dataset("valid").collater)

    logger.info("evaluating ...")
    class_labels_list, class_probs_list, reg_logits_list, reg_targets_list = [], [], [], []
    for batch in batch_itr:
        with torch.no_grad():
            batch = move_to_cuda(batch)
            net_output = model(**batch["net_input"])

            class_logits = net_output['class_encoder_out'].float()
            reg_logits = net_output['regression_out'].float()

            class_probs = torch.sigmoid(class_logits.float())

            class_labels = batch['sed_labels'].float()
            reg_targets = batch['doa_labels'].float()

            class_labels_list.append(class_labels)
            class_probs_list.append(class_probs)
            reg_logits_list.append(reg_logits)
            reg_targets_list.append(reg_targets)

    assert set([len(i) for i in [class_labels_list, class_probs_list,
               reg_logits_list, reg_targets_list]]) == 1

    thr_list = [round(float(i), 3) for i in np.arange(0.1, 1.0, 0.025)]

    results_list = []

    for thr in thr_list:

        y_true_reg_list, y_pred_reg_list, y_true_list, y_pred_list = [], [], [], []

        for i in range(len(class_labels_list)):

            class_labels = class_labels_list[i]
            class_probs = class_probs_list[i]
            reg_logits = reg_logits_list[i]
            reg_targets = reg_targets_list[i]

            class_mask = class_probs > thr

            class_preds = class_mask.float()

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(
                0.0).to(class_labels.device)

            class_mask_extended = torch.cat([class_mask]*doa_size, dim=-1)

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

            y_pred_list.append(y_pred_class)
            y_true_list.append(y_true_class)

            y_pred_reg_list.append(y_pred_reg)
            y_true_reg_list.append(y_true_reg)

        sed_pred = np.concatenate(y_pred_list)
        sed_gt = np.concatenate(y_true_list)

        doa_pred = np.concatenate(y_pred_reg_list)
        doa_gt = np.concatenate(y_true_reg_list)

        sed_metric = compute_sed_scores(
            sed_pred, sed_gt, nb_label_frames_1s)

        logger.info(f"sed_metric: {sed_metric}")

        _er, _f = sed_metric

        doa_metric = compute_doa_scores_regr_xyz(
            doa_pred, doa_gt, sed_pred, sed_gt)

        logger.info(f"doa_metric: {doa_metric}")

        _doa_err, _frame_recall, _, _, _, _ = doa_metric

        _seld_scr = early_stopping_metric([_er, _f], [_doa_err, _frame_recall])

        logger.info(f"f1: {_f}")
        logger.info(f"er: {_er}")
        logger.info(f"doa: {_doa_err}")
        logger.info(f"recall: {_frame_recall}")
        logger.info(f"seld score: {_seld_scr}")

        results = {"f1_mean": str(_f),
                   "er_mean": str(_er),
                   "doa_error": str(_doa_err),
                   "frame_recall": str(_frame_recall),
                   "seld_score": str(_seld_scr)}

        results_list.append(results)

    f1_mean_list = []
    er_mean_list = []
    doa_error_list = []
    frame_recall_list = []
    seld_score_list = []
    for i in range(len(results_list)):
        for key in list(results.keys()):

            if key == "f1_mean":
                f1_mean_list.append(results_list[i][key])

            if key == "er_mean":
                er_mean_list.append(results_list[i][key])

            if key == "doa_error":
                doa_error_list.append(results_list[i][key])

            if key == "frame_recall":
                frame_recall_list.append(results_list[i][key])

            if key == "seld_score":
                seld_score_list.append(results_list[i][key])

    f1_mean_list = [float(i) for i in f1_mean_list]
    er_mean_list = [float(i) for i in er_mean_list]
    frame_recall_list = [float(i) for i in frame_recall_list]
    doa_error_list = [float(i) for i in doa_error_list]
    seld_score_list = [float(i) for i in seld_score_list]

    min_x = np.argmin(seld_score_list)
    print("best seld-score", seld_score_list[min_x])

    thr_best = thr_list[min_x]
    print("best threshold", thr_best)

    evaluate(ckpt_path, "test", batch_size,
             f"{output_path}/base_metrics.json", user_dir, doa_size, data, thr=0.5)

    evaluate(ckpt_path, "test", batch_size,
             f"{output_path}/best_metrics.json", user_dir, doa_size, data, thr=thr_best)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path',
                        required=True,
                        help="path to model checkpoint file.")
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='max number of tokens in each batch.')
    parser.add_argument('--output_path',
                        default="reports",
                        type=str,
                        help=("path output metrics file."))
    parser.add_argument('--user_dir',
                        default="src/seld_wav2vec2",
                        type=str,
                        help=("path to a python module containing custom",
                              "extensions."))
    parser.add_argument('--doa_size',
                        default=3,
                        type=int,
                        help='Size of DOA.')
    parser.add_argument('--data',
                        type=str,
                        help=("path and name of output metrics file."))

    args = parser.parse_args()
    evaluate_list_thr(args.ckpt_path, args.batch_size,
                      args.output_file_path, args.user_dir, args.doa_size, args.data)
