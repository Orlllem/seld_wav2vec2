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
    compute_doa_scores_regr, early_stopping_metric, er_overall_framewise,
    f1_overall_framewise)
from seld_wav2vec2.criterions.multi_label_regression import reshape_3Dto2D

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")


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


def evaluate(ckpt_path, set, batch_size, output_file_path, user_dir,
             doa_size=2, cartesian_coordinates=False):

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

    for batch in batch_itr:
        with torch.no_grad():
            batch = move_to_cuda(batch)
            net_output = model(**batch["net_input"])

            class_logits = net_output['class_encoder_out'].float()
            reg_logits = net_output['regression_out'].float()

            class_probs = torch.sigmoid(class_logits.float())

            class_mask = class_probs > 0.5

            class_preds = class_mask.float()

            class_labels = batch['sed_labels'].float()
            reg_targets = batch['doa_labels'].float()

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(
                0.0).to(class_labels.device)

            class_mask_extended = torch.cat([class_mask]*doa_size, dim=-1)

            reg_logits[~class_mask_extended] = torch.tensor(
                0.0).to(reg_targets)
            reg_logits = reg_logits.cpu().numpy()

            class_preds = class_preds.cpu().numpy()
            class_labels = class_labels.cpu().numpy()

            B, T, N = class_labels.shape
            reg_logits = reg_logits.reshape((B, T, N, doa_size))
            reg_targets = reg_targets.reshape(
                reg_logits.shape).cpu().numpy()

            if cartesian_coordinates:
                reg_logits_rad = cart2sph_array(reg_logits)
                reg_targets_rad = cart2sph_array(reg_targets)
            else:
                reg_targets_rad = reg_targets*np.pi / 180
                reg_logits_rad = reg_logits*np.pi / 180

        y_true.append(reshape_3Dto2D(class_labels))
        y_pred.append(reshape_3Dto2D(class_preds))

        y_true_reg.append(reshape_3Dto2D(reg_targets_rad))
        y_pred_reg.append(reshape_3Dto2D(reg_logits_rad))

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
    parser.add_argument('--cartesian_coordinates',
                        default=False,
                        type=bool,
                        help='Whether it is using cartesian coordinates.')

    args = parser.parse_args()
    evaluate(args.ckpt_path, args.set, args.batch_size, args.output_file_path,
             args.user_dir, args.doa_size, args.cartesian_coordinates)
