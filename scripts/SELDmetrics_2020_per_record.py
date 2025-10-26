import logging
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, Tuple

import fairseq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.utils import move_to_cuda
from omegaconf import OmegaConf, open_dict

from seld_wav2vec2.criterions.evaluation_metrics import early_stopping_metric

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate-2020-per-record")

SAMPLE_RATE = 16000

colors = [
    (0.09676227253559819, 0.9830171624399461, 0.19914743050554115),
    (0.9710781619311514, 0.10693913388932161, 0.6707916233534635),
    (0.10627541532810426, 0.49860059032999904, 0.9326878821840091),
    (0.9168536757041643, 0.6915370678292033, 0.12257009256916077),
    (0.34764963939161575, 0.4164907042605828, 0.30121309804609503),
    (0.35910796424290825, 0.9882577269516203, 0.8759364673876884),
    (0.7756745481061432, 0.5229912973607651, 0.9371556929402267),
    (0.4285670623044025, 0.10038747142379564, 0.8759213870581116),
    (0.7209734302469649, 0.11569182086855051, 0.09880788173118137),
    (0.6410793809231611, 0.9996882476840667, 0.34303337929314137),
    (0.4851780074346647, 0.679209674324328, 0.6415736720853036),
    (0.770343247834787, 0.43240524707107864, 0.4632441573523907),
    (0.1267216854101511, 0.0923725802444786, 0.4745091734780951),
    (0.9576577859571034, 0.75527990621142, 0.6448705006855648),
]

unique_classes = [
    "alarm",
    "baby",
    "crash",
    "dog",
    "engine",
    "female_scream",
    "female_speech",
    "fire",
    "footsteps",
    "knock",
    "male_scream",
    "male_speech",
    "phone",
    "piano",
]


def get_seld_metrics(
    fnames_ref_preds: Dict,
    fnames: Dict,
    task: fairseq.tasks.FairseqTask,
    num_classes: int,
    linewidth: float = 3.0,
    figsize: Tuple[int, int] = (10, 8),
    save_fig: bool = False,
    model_name: str = None,
    label_hop_len_s: float = 0.02,
) -> Dict:
    """
    Calculate per-record SELD metrics and optionally plot SED and DOA predictions vs references.
    Args:
        fnames_ref_preds: Dictionary with predictions and references for each file.
        fnames: file names list.
        task: Fairseq task object.
        num_classes: Number of classes.
        linewidth: Line width for plots.
        figsize: Figure size for plots.
        save_fig: Whether to save the plots as PDF files.
        model_name: Name of the model being evaluated.
        label_hop_len_s: Hop length of labels in seconds.
    Returns:
        seld_metrics: Dictionary with SELD metrics for each record.
    """
    seld_metrics = {}

    for key in fnames:
        preds = fnames_ref_preds[key]["pred"]
        refs = fnames_ref_preds[key]["ref"]

        doa_preds_x = fnames_ref_preds[key]["doa_preds"]["x"]
        doa_preds_y = fnames_ref_preds[key]["doa_preds"]["y"]
        doa_preds_z = fnames_ref_preds[key]["doa_preds"]["z"]

        doa_refs_x = fnames_ref_preds[key]["doa_labels"]["x"]
        doa_refs_y = fnames_ref_preds[key]["doa_labels"]["y"]
        doa_refs_z = fnames_ref_preds[key]["doa_labels"]["z"]

        doa_preds = np.vstack((doa_preds_x, doa_preds_y, doa_preds_z)).T
        doa_refs = np.vstack((doa_refs_x, doa_refs_y, doa_refs_z)).T

        y_pred_class = preds.T
        y_true_class = refs.T

        task.eval_seld_score_2020(doa_preds, doa_refs, y_pred_class, y_true_class)
        er, f, de, de_f = task.cls_new_metric.compute_seld_scores()
        seld_score = early_stopping_metric([er, f], [de, de_f])

        task.cls_new_metric.reset_states()

        ov_type = y_true_class.sum(-1).max()

        if ov_type <= 1:
            ov_type = 1

        seld_metrics[key] = {
            "F1": f,
            "ER": er,
            "DOA-error": de,
            "FR": de_f,
            "SELD_score": seld_score,
            "frame_events": y_true_class.sum(),
            "ov_type": ov_type,
        }

        mask_pred = preds == 0
        mask_ref = refs == 0

        preds[mask_pred] = np.nan
        refs[mask_ref] = np.nan

        doa_preds_x[mask_pred] = np.nan
        doa_refs_x[mask_ref] = np.nan
        doa_preds_y[mask_pred] = np.nan
        doa_refs_y[mask_ref] = np.nan
        doa_preds_z[mask_pred] = np.nan
        doa_refs_z[mask_ref] = np.nan

        if save_fig:
            x_range = np.arange(len(preds[0])) * label_hop_len_s

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
                4, 2, figsize=figsize
            )
            ax1.set_title("SED predicted")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Class-id")
            for i in range(num_classes):
                ax1.plot(
                    x_range,
                    preds[i] * i,
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            ax2.set_title("SED reference")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Class-id")
            for i in range(num_classes):
                ax2.plot(
                    x_range,
                    refs[i] * i,
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            # X
            ax3.set_title("X-axis DOA predicted")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Position")
            for i in range(num_classes):
                ax3.plot(
                    x_range,
                    doa_preds_x[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            ax4.set_title("X-axis DOA reference")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Position")
            for i in range(num_classes):
                ax4.plot(
                    x_range,
                    doa_refs_x[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            # Y
            ax5.set_title("Y-axis DOA predicted")
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Position")
            for i in range(num_classes):
                ax5.plot(
                    x_range,
                    doa_preds_y[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            ax6.set_title("Y-axis DOA reference")
            ax6.set_xlabel("Time (s)")
            ax6.set_ylabel("Position")
            for i in range(num_classes):
                ax6.plot(
                    x_range,
                    doa_refs_y[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            # Z
            ax7.set_title("Z-axis DOA predicted")
            ax7.set_xlabel("Time (s)")
            ax7.set_ylabel("Position")
            for i in range(num_classes):
                ax7.plot(
                    x_range,
                    doa_preds_z[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            ax8.set_title("Z-axis DOA reference")
            ax8.set_xlabel("Time (s)")
            ax8.set_ylabel("Position")
            for i in range(num_classes):
                ax8.plot(
                    x_range,
                    doa_refs_z[i],
                    label=f"{unique_classes[i]}",
                    linewidth=linewidth,
                    color=colors[i],
                )

            handles, labels = ax8.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.01),
                fancybox=True,
                shadow=True,
                ncol=5,
            )

            for ax in fig.get_axes():
                ax.label_outer()

            fig.tight_layout()

            fig.savefig(
                f"{key}_{model_name}.pdf", format="pdf", dpi=300, bbox_inches="tight"
            )

    return seld_metrics


def get_xyz(
    accdoa_in: torch.Tensor, nb_classes: int = 11
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the ACCDOA format into x, y, z components.
    Args:
        accdoa_in: ACCDOA formatted tensor of shape (B, T, 3 * nb_classes)
        nb_classes: number of classes
    Returns:
        x, y, z components each of shape (B, T, nb_classes)
    """
    x, y, z = (
        accdoa_in[:, :, :nb_classes],
        accdoa_in[:, :, nb_classes: 2 * nb_classes],
        accdoa_in[:, :, 2 * nb_classes:],
    )
    return x, y, z


def evaluate(
    ckpt_path: str,
    model_name: str,
    subset: str,
    output_file_path: str,
    user_dir: str,
    doa_size: int = 3,
    data: str = None,
    label_hop_len_s: float = None,
    labels: str = None,
    segment_in_seconds: bool = True,
    segment_window: int = None,
    segment_stride: int = None,
    use_best_threshold: bool = True,
    evaluate_at_100ms: bool = False,
    save_fig: bool = False,
):
    """
    Evaluate a trained model on a given subset and save per-record SELD metrics to a CSV file.
    Args:
        ckpt_path: Path to the model checkpoint file.
        model_name: Name of the model (for logging purposes).
        subset: Subset to evaluate (e.g., "test").
        output_file_path: Path to save the output metrics CSV file.
        user_dir: Path to a python module containing custom extensions.
        doa_size: Size of DOA (default is 3 for x, y, z).
        data: Path to the data (optional).
        label_hop_len_s: Hop length of labels in seconds.
        labels: Path to the labels (optional).
        segment_in_seconds: Whether to use segment window and stride in seconds.
        segment_window: Segment window size (in seconds or number of samples).
        segment_stride: Segment stride size (in seconds or number of samples).
        use_best_threshold: Whether to use the best threshold for classification.
        evaluate_at_100ms: Whether to adjust model for evaluation at 100ms hop length.
        save_fig: Whether to save SED and DOA prediction plots as PDF files.
    Returns:
        None
    """
    Arg = namedtuple("Arg", ["user_dir"])
    arg = Arg(user_dir.__str__())
    fairseq.utils.import_user_module(arg)

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

    model = models[0].eval()
    if torch.cuda.is_available():
        model = model.cuda()

    logger.info(model)

    model.w2v_encoder.apply_mask = False
    task.cfg.audio_augm = False
    task.cfg.random_crop = False
    task.cfg.enable_padding = False
    task.cfg.max_sample_size = None

    if data is not None:
        task.cfg.data = data

    if labels is not None:
        task.cfg.labels = labels

    if label_hop_len_s is not None:
        task.cfg.label_hop_len_s = label_hop_len_s

    if evaluate_at_100ms:
        logger.info("Evaluating at 100ms hop length.")
        task.cfg.align_outputs_frames = True
        model.w2v_encoder.align_100ms = True
        model.w2v_encoder.pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)
        model.w2v_encoder.mask_pool = nn.AvgPool1d(kernel_size=5, stride=5, ceil_mode=True)

    if segment_in_seconds:
        logger.info("Using segment window and stride in seconds.")
    else:
        logger.info("Using segment window and stride in number of labels.")
        OmegaConf.set_struct(task.cfg, True)
        with open_dict(task.cfg):
            task.cfg.segment_in_seconds = False

        task.cfg.segment_window = segment_window
        task.cfg.segment_stride = segment_stride

    logger.info(f"Evaluating {subset} subset")

    logger.info(task.cfg)

    task.load_dataset(subset)
    if subset != "test":
        task.load_dataset("test")

    if use_best_threshold:
        thr_best = task.best_threshold
    else:
        thr_best = 0.5

    logger.info(f"Using threshold: {thr_best}")

    assert task.dataset(subset).dataset.pad is False, (
        "Dataset padding must be disabled for per-record evaluation."
    )

    batch_itr = torch.utils.data.DataLoader(
        task.dataset(subset),
        batch_size=1,
        shuffle=False,
        collate_fn=task.dataset(subset).collater,
    )

    fnames = [i.as_py().decode("utf-8") for i in task.dataset(subset).dataset.fnames]

    logger.info(f"fnames size {len(fnames)}")

    num_classes = len(unique_classes)

    # create dict to store predictions and references
    fnames_ref_preds = {}
    for key in fnames:
        fnames_ref_preds[key] = []

    # iterate over data
    for batch in batch_itr:
        with torch.no_grad():
            idx = batch["id"].cpu().numpy()[0]

            if torch.cuda.is_available():
                batch = move_to_cuda(batch)

            net_output, batch = task.inference_step(batch, model)

            class_logits = net_output["class_encoder_out"].float()
            reg_logits = net_output["regression_out"].float()

            class_probs = torch.sigmoid(class_logits.float())

            class_mask = class_probs > thr_best

            class_preds = class_mask.float()

            class_labels = batch["sed_labels"].float()
            reg_targets = batch["doa_labels"].float()

            class_pad_mask = class_labels < 0
            class_labels[class_pad_mask] = torch.tensor(0.0).to(class_labels)

            B, T, N = class_labels.shape

            x, y, z = get_xyz(reg_targets.cpu(), num_classes)
            xp, yp, zp = get_xyz(reg_logits.cpu(), num_classes)

            y_true = class_labels.squeeze(0).cpu().numpy().T
            y_pred = class_preds.squeeze(0).cpu().numpy().T

            x = x.squeeze(0).cpu().numpy().T
            y = y.squeeze(0).cpu().numpy().T
            z = z.squeeze(0).cpu().numpy().T
            xp = xp.squeeze(0).cpu().numpy().T
            yp = yp.squeeze(0).cpu().numpy().T
            zp = zp.squeeze(0).cpu().numpy().T

            fnames_ref_preds[fnames[idx]] = {
                    "pred": y_pred,
                    "ref": y_true,
                    "doa_labels": {"x": x, "y": y, "z": z},
                    "doa_preds": {"x": xp, "y": yp, "z": zp},
                }

    seld_metrics = get_seld_metrics(
        fnames_ref_preds=fnames_ref_preds,
        fnames=fnames,
        task=task,
        num_classes=num_classes,
        save_fig=save_fig,
        model_name=model_name,
        label_hop_len_s=task.cfg.label_hop_len_s,
    )

    # Creates Output folder if it doesn't exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    df = pd.DataFrame(seld_metrics).T

    df.index.name = "record"

    df["F1"] = df["F1"] * 100
    df["FR"] = df["FR"] * 100

    df.rename(columns={"F1": "F1%", "FR": "FR%"}, inplace=True)

    df["model"] = model_name

    logger.info(df)

    logger.info(df.describe())

    df.sort_values(by=["SELD_score"])

    # save to file
    df.to_csv(output_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path", required=True, help="path to model checkpoint file."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help=("path and name of output metrics file."),
    )
    parser.add_argument("--subset", default="test", help="subset to evaluate.")
    parser.add_argument(
        "--output_file_path",
        default="reports/metrics.csv",
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
    parser.add_argument("--data", type=str, help=("path to the data."))
    parser.add_argument("--label_hop_len_s", type=float, default=None, help="label_hop_len_s")
    parser.add_argument("--labels", type=str, default=None, help="labels path")
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
    parser.add_argument("--use_best_threshold", action="store_true")
    parser.add_argument("--evaluate_at_100ms", action="store_true")
    parser.add_argument("--save_fig", action="store_true")

    args = parser.parse_args()
    evaluate(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        subset=args.subset,
        output_file_path=args.output_file_path,
        user_dir=args.user_dir,
        doa_size=args.doa_size,
        data=args.data,
        label_hop_len_s=args.label_hop_len_s,
        labels=args.labels,
        segment_in_seconds=args.segment_in_seconds,
        use_best_threshold=args.use_best_threshold,
        evaluate_at_100ms=args.evaluate_at_100ms,
        segment_window=args.segment_window,
        segment_stride=args.segment_stride,
        save_fig=args.save_fig,
    )
