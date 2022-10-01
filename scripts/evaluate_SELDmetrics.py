import json
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
    compute_doa_scores_regr, compute_seld_metric, er_overall_framewise,
    f1_overall_framewise)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")


def evaluate(ckpt_path, set, batch_size, output_file_path, user_dir,
             doa_size=2, range_value=(-1, 1), range_ele=(-60, 50),
             range_azi=(-180, 160)):

    Arg = namedtuple("Arg", ["user_dir"])
    arg = Arg(user_dir.__str__())
    fairseq.utils.import_user_module(arg)

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([
                                                                    ckpt_path])
    model = models[0].eval().cuda()

    model.w2v_encoder.apply_mask = False
    task.cfg.audio_augm = False
    task.cfg.random_crop = False
    task.load_dataset(set)

    logger.info("iterating over data")
    batch_itr = DataLoader(task.dataset(set),
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=task.dataset(set).collater)

    y_true_reg, y_pred_reg, y_true, y_pred = [], [], [], []

    min_value = range_value[0]
    max_value = range_value[1]

    feature_range_min_ele = range_ele[0]
    feature_range_max_ele = range_ele[1]

    feature_range_min_azi = range_azi[0]
    feature_range_max_azi = range_azi[1]

    logger.info(f"preprocessed range values: ({min_value}, {max_value})")
    logger.info(
        f"range ele: ({feature_range_min_ele}, {feature_range_max_ele})")
    logger.info(
        f"range azi: ({feature_range_min_azi}, {feature_range_max_azi})")

    for batch in batch_itr:
        with torch.no_grad():
            batch = move_to_cuda(batch)
            net_output = model(**batch["net_input"])

            class_logits = net_output['class_encoder_out']
            reg_logits = net_output['regression_out']

            class_probs = torch.sigmoid(class_logits.float())

            class_mask = class_probs > 0.5

            class_preds = class_mask.float()

            class_labels = batch['sed_labels'].float()
            reg_targets = batch['doa_labels'].float()

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(
                0.0).to(class_labels.device)

            class_mask_extended = torch.cat([class_mask]*doa_size, dim=-1)

            reg_targets_np = reg_targets.cpu().numpy()
            reg_targets_np = np.where(reg_targets_np == 0, np.nan,
                                      reg_targets_np)

            reg_targets_norm_x = (
                reg_targets_np[:, :, 0:11] - min_value)/(max_value - min_value)
            reg_targets_norm_x = reg_targets_norm_x * \
                (feature_range_max_ele - feature_range_min_ele) + \
                feature_range_min_ele
            reg_targets_norm_x = reg_targets_norm_x*np.pi / 180

            reg_targets_norm_z = (
                reg_targets_np[:, :, 11:] - min_value)/(max_value - min_value)
            reg_targets_norm_z = reg_targets_norm_z * \
                (feature_range_max_azi - feature_range_min_azi) + \
                feature_range_min_azi
            reg_targets_norm_z = reg_targets_norm_z*np.pi / 180

            reg_logits_norm_x = (reg_logits[:, :, 0:11].cpu(
            ).numpy() - min_value)/(max_value - min_value)
            reg_logits_norm_x = reg_logits_norm_x * \
                (feature_range_max_ele - feature_range_min_ele) + \
                feature_range_min_ele
            reg_logits_norm_x = reg_logits_norm_x*np.pi / 180

            reg_logits_norm_z = (reg_logits[:, :, 11:].cpu(
            ).numpy() - min_value)/(max_value - min_value)
            reg_logits_norm_z = reg_logits_norm_z * \
                (feature_range_max_azi - feature_range_min_azi) + \
                feature_range_min_azi
            reg_logits_norm_z = reg_logits_norm_z*np.pi / 180

            reg_logits_norm = np.concatenate((reg_logits_norm_x,
                                              reg_logits_norm_z), axis=-1)
            reg_targets_norm = np.concatenate((reg_targets_norm_x,
                                              reg_targets_norm_z), axis=-1)

            reg_targets_norm[np.isnan(reg_targets_norm)] = 0

            reg_logits_norm = torch.from_numpy(reg_logits_norm)
            reg_logits_norm[~class_mask_extended] = torch.tensor(
                0.0).to(reg_targets.device)
            reg_logits_norm = reg_logits_norm.cpu().numpy()

        y_true.append(class_labels.cpu().numpy().reshape(
            (class_labels.shape[0]*class_labels.shape[1], class_labels.shape[2])))
        y_pred.append(class_preds.cpu().numpy().reshape(
            (class_preds.shape[0]*class_preds.shape[1], class_preds.shape[2])))

        y_true_reg.append(reg_targets_norm.reshape(
            (reg_targets.shape[0]*reg_targets.shape[1], reg_targets.shape[2])))
        y_pred_reg.append(reg_logits_norm.reshape(
            (reg_logits.shape[0]*reg_logits.shape[1], reg_logits.shape[2])))

    logger.info("evaluating ...")

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_true_reg = np.concatenate(y_true_reg)
    y_pred_reg = np.concatenate(y_pred_reg)

    er_metric = compute_doa_scores_regr(y_pred_reg,
                                        y_true_reg,
                                        y_pred,
                                        y_true)

    logger.info(f"er_metric: {er_metric}")

    _doa_err, _frame_recall, _, _, _, _ = er_metric
    _er = er_overall_framewise(y_pred, y_true)
    _f = f1_overall_framewise(y_pred, y_true)
    _seld_scr = compute_seld_metric([_er, _f], [_doa_err, _frame_recall])

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

    # Creates Output folder if it doesn't exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as output_file:
        json.dump(results, output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path',
                        required=True,
                        help="path to model checkpoint file.")
    parser.add_argument('--set',
                        default="test",
                        help="set to evaluate.")
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='max number of tokens in each batch.')
    parser.add_argument('--output_file_path',
                        default="reports/metrics.json",
                        type=str,
                        help=("path and name of output metrics file."))
    parser.add_argument('--user_dir',
                        default="src/seld_wav2vec2",
                        type=str,
                        help=("path to a python module containing custom",
                              "extensions."))
    parser.add_argument('--doa_size',
                        default=2,
                        type=int,
                        help='Size of DOA.')
    parser.add_argument('--range_value',
                        default=[-1, 1],
                        nargs='+',
                        type=float,
                        help='Range of preprocessed values in regression.')
    parser.add_argument('--range_ele',
                        default=[-60, 50],
                        nargs='+',
                        type=float,
                        help='Range of x values in regression.')
    parser.add_argument('--range_azi',
                        default=[-180, 160],
                        nargs='+',
                        type=float,
                        help='Range of z values in regression.')

    args = parser.parse_args()
    evaluate(args.ckpt_path, args.set, args.batch_size, args.output_file_path,
             args.user_dir, args.doa_size, tuple(args.range_value),
             tuple(args.range_ele),
             tuple(args.range_azi))
