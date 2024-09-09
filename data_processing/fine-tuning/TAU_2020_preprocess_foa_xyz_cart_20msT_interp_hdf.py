import copy
import glob
import itertools
import logging
import os
import shutil
from collections import Counter

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig
from torchaudio.transforms import Spectrogram
from tqdm import tqdm
from utils import (
    _next_greater_power_of_2,
    cart2sph,
    cart2sph_array,
    gen_tsv_manifest,
    get_feat_extract_output_lengths,
    get_feat_extract_output_lengths_spec,
    linear_interp,
    remove_overlap_same_class,
    sph2cart,
)

logger = logging.getLogger("preprocessing-ft-tau2020-interp")

ROOT_DIR = os.path.abspath(os.curdir)
DEF_SAMPLE_RATE = 24000
FS_TARGET = 16000
MIN_LENGTH = 400
DOA_SIZE = 3
TEMPORAL_RESOLUTION = 100 * 1e-3  # 100ms
VALID_SPLIT = "fold6"
CONV_FEATURE_LAYERS = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
CONV_FEATURE_LAYERS_SPEC = "[(512,2,2)] + [(512,2,2)]"
MASK_ID = -1000
INTERP_FUN = "linear"


unique_classes_dict = {
    "alarm": 0,
    "baby": 1,
    "crash": 2,
    "dog": 3,
    "engine": 4,
    "female_scream": 5,
    "female_speech": 6,
    "fire": 7,
    "footsteps": 8,
    "knock": 9,
    "male_scream": 10,
    "male_speech": 11,
    "phone": 12,
    "piano": 13,
}


def get_original_DOA_100ms(df, num_classes, max_size=None):
    if max_size is None:
        max_size = max(df["frame_number"])

    ele = np.zeros((num_classes, max_size))
    azi = np.zeros((num_classes, max_size))
    ele[:] = np.nan
    azi[:] = np.nan

    for index, row in df.iterrows():
        class_id = int(row["sound_event_recording"])
        ele[class_id, int(row["frame_number"]) - 1] = row["ele"]
        azi[class_id, int(row["frame_number"]) - 1] = row["azi"]

    return ele, azi


def get_array_class(array_ele, array_azi):
    ele_class = array_ele.copy()
    azi_class = array_azi.copy()

    mask_ele_class = np.isnan(ele_class)
    mask_azi_class = np.isnan(azi_class)

    mask_class = mask_ele_class | mask_azi_class

    ele_class[~mask_ele_class] = 1
    ele_class[mask_ele_class] = 0

    azi_class[~mask_azi_class] = 1
    azi_class[mask_azi_class] = 0

    array_class = (ele_class.astype(bool) | azi_class.astype(bool)).astype(float)

    array_class[mask_class] = np.nan

    return array_class


def preprocess_waves_metadata(
    metadata_dir,
    wav_dir,
    num_classes,
    unique_classes,
    seldnet_window,
    save_folder,
    X_train=[],
    X_valid=[],
    train=False,
    vizualize_figs=False,
    spec_transform=None,
):
    dict_files = {}
    wav_names = []

    csv_files = glob.glob(f"{metadata_dir}/**/*.csv", recursive=True)

    for csv_filename in tqdm(csv_files):
        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df["start_time"] = df[0] * TEMPORAL_RESOLUTION
        df["end_time"] = df[0] * TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: "sound_event_recording"}, inplace=True)
        df.rename(columns={0: "frame_number"}, inplace=True)
        df.rename(columns={3: "azi"}, inplace=True)
        df.rename(columns={4: "ele"}, inplace=True)

        # remove duplicate same class on same time-step
        df = remove_overlap_same_class(df)

        filename = os.path.splitext(os.path.basename(csv_filename))[0]

        wav_filename = f"{wav_dir}/{filename}.wav"

        wav, curr_sample_rate = torchaudio.load(wav_filename)

        assert curr_sample_rate == DEF_SAMPLE_RATE, (
            f"{curr_sample_rate}!={DEF_SAMPLE_RATE}"
        )

        # wav_info = torchaudio.info(wav_filename)

        resampler = T.Resample(curr_sample_rate, FS_TARGET)

        wav_resampled = resampler(wav)

        set_filename = f"{os.path.basename(wav_dir)}_{filename}"

        dict_events = {}
        for class_i in unique_classes:
            dict_events[class_i] = {}

        for class_i in unique_classes:
            df_class = df[df["sound_event_recording"] == class_i]

            df_class_index = df_class.index

            events = []
            ele_list = []
            azi_list = []
            start_list = []
            for i in range(len(df_class_index)):
                start_time = df_class["start_time"][df_class_index[i]]
                end_time = df_class["end_time"][df_class_index[i]]
                ele = df_class["ele"][df_class_index[i]]
                azi = df_class["azi"][df_class_index[i]]

                start = int(np.floor(start_time * FS_TARGET))
                end = int(np.floor(end_time * FS_TARGET))

                ele_array = [np.nan] * (end - start)
                ele_array[0] = ele

                azi_array = [np.nan] * (end - start)
                azi_array[0] = azi

                ele_list.append(ele)
                azi_list.append(azi)
                start_list.append(start)
                try:
                    fm = df_class["frame_number"][df_class_index[i]]
                    fm_next = df_class["frame_number"][df_class_index[i + 1]]

                    if fm - fm_next != -1:
                        ele_array[-1] = ele
                        azi_array[-1] = azi
                except IndexError:
                    ele_array[-1] = linear_interp(end, start_list[-2:], ele_list[-2:])
                    azi_array[-1] = linear_interp(end, start_list[-2:], azi_list[-2:])
                events.append(
                    {"range": np.arange(start, end), "ele": ele_array, "azi": azi_array}
                )

            dict_events[class_i] = events

        classes_list = list(dict_events.keys())

        assert len(classes_list) == len(unique_classes), (
            f"{classes_list}!={unique_classes}"
        )

        array_ele = []
        array_azi = []
        for class_i in unique_classes:
            array_empty_ele = np.zeros(wav_resampled.shape[1])
            array_empty_ele[:] = MASK_ID

            array_empty_azi = np.zeros(wav_resampled.shape[1])
            array_empty_azi[:] = MASK_ID

            for event in dict_events[class_i]:
                array_empty_ele[event["range"]] = event["ele"]
                array_empty_azi[event["range"]] = event["azi"]

            array_ele.append(array_empty_ele)
            array_azi.append(array_empty_azi)

        array_ele = np.vstack(array_ele)
        array_azi = np.vstack(array_azi)

        mask_ele = array_ele == MASK_ID
        mask_azi = array_azi == MASK_ID

        array_ele = pd.DataFrame(array_ele).T
        array_azi = pd.DataFrame(array_azi).T

        for i in range(len(unique_classes)):
            if np.isnan(array_ele[i].values).any():
                sel_array_ele = array_ele[i][~mask_ele[i]].interpolate(
                    method=INTERP_FUN, limit_area="inside"
                )
                array_ele[i][~mask_ele[i]] = sel_array_ele

            assert not np.isnan(array_ele[i].values).any()

            if np.isnan(array_azi[i].values).any():
                sel_array_azi = array_azi[i][~mask_azi[i]].interpolate(
                    method=INTERP_FUN, limit_area="inside"
                )
                array_azi[i][~mask_azi[i]] = sel_array_azi

            assert not np.isnan(array_azi[i].values).any()

        array_ele = array_ele.T.values
        array_azi = array_azi.T.values

        assert not np.isnan(array_ele).all()
        assert not np.isnan(array_azi).all()

        # ele and azi at FS_TARGET
        array_ele[mask_ele] = np.nan
        array_azi[mask_azi] = np.nan

        input_length = torch.tensor(array_ele.shape[1])

        orig_ele, orig_azi = get_original_DOA_100ms(
            df, num_classes=num_classes, max_size=int(input_length / (FS_TARGET * 0.1))
        )
        orig_class = get_array_class(array_ele=orig_ele, array_azi=orig_azi)
        array_class = get_array_class(array_ele=array_ele, array_azi=array_azi)

        class_dict = []

        # make sure that the wave lenghth is equal to x, z length
        assert wav_resampled.shape[1] == input_length, (
            f"wav_resampled: {wav_resampled.shape}, input_length: {input_length}"
        )

        if spec_transform is not None:
            output_length = get_feat_extract_output_lengths_spec(
                CONV_FEATURE_LAYERS_SPEC, spec_transform, input_length
            ).tolist()
        else:
            output_length = get_feat_extract_output_lengths(
                CONV_FEATURE_LAYERS, input_length
            ).tolist()

        pooler = nn.AdaptiveMaxPool1d(output_length)

        # convert ele, azi -> x,y,z
        x, y, z = sph2cart(array_azi * np.pi / 180, array_ele * np.pi / 180, r=1)

        azimuth, elevation, r = cart2sph(x, y, z)

        azimuth = azimuth * 180 / np.pi
        elevation = elevation * 180 / np.pi

        r_sel = r[~np.isnan(r)]
        azi_sel = array_azi.copy()
        azi_sel[np.isnan(azi_sel)] = 0.0
        azimuth_sel = azimuth.copy()
        azimuth_sel[np.isnan(azimuth_sel)] = 0.0

        ele_sel = array_ele.copy()
        ele_sel[np.isnan(ele_sel)] = 0.0
        elevation_sel = elevation.copy()
        elevation_sel[np.isnan(elevation_sel)] = 0.0

        # rotate azimuth < -180 to positive
        if np.min(azi_sel) < -180.0:
            azi_sel[azi_sel < -180.0] = azi_sel[azi_sel < -180.0] + 360
        elif np.max(azi_sel) > 180.0:
            # rotate azimuth > 180 to positive
            azi_sel[azi_sel > 180.0] = azi_sel[azi_sel > 180.0] - 360

        assert np.allclose(r_sel, [1.0] * len(r_sel), atol=1e-05)
        assert np.allclose(azimuth_sel, azi_sel, atol=1e-05)
        assert np.allclose(elevation_sel, ele_sel, atol=1e-05)

        output_ele = pooler(torch.from_numpy(array_ele)).cpu().numpy()
        output_azi = pooler(torch.from_numpy(array_azi)).cpu().numpy()

        output_ele_class = output_ele.copy()
        output_azi_class = output_azi.copy()

        mask_output_ele_class = np.isnan(output_ele_class)
        mask_output_azi_class = np.isnan(output_azi_class)

        mask_class = mask_output_ele_class | mask_output_azi_class

        output_ele_class[~mask_output_ele_class] = 1
        output_ele_class[mask_output_ele_class] = 0

        output_azi_class[~mask_output_azi_class] = 1
        output_azi_class[mask_output_azi_class] = 0

        # get classes binary matrix
        output_class = (
            output_ele_class.astype(bool) | output_azi_class.astype(bool)
        ).astype(int)

        output_xx, output_yy, output_zz = sph2cart(
            output_azi * np.pi / 180, output_ele * np.pi / 180, r=1
        )

        # normalize x values
        x_norm = output_xx.copy()
        x_norm[np.isnan(x_norm)] = 0

        # normalize y values
        y_norm = output_yy.copy()
        y_norm[np.isnan(y_norm)] = 0

        # normalize z values
        z_norm = output_zz.copy()
        z_norm[np.isnan(z_norm)] = 0

        assert output_class.shape[0] == num_classes
        assert x_norm.shape[0] == num_classes
        assert y_norm.shape[0] == num_classes
        assert z_norm.shape[0] == num_classes

        output_xyz = np.expand_dims(
            np.stack((x_norm, y_norm, z_norm), axis=0).T, axis=0
        )

        B = output_xyz.shape[0]
        Ts = output_xyz.shape[1]

        output_sph = np.expand_dims(
            np.stack((output_ele, output_azi), axis=0).T, axis=0
        ).reshape(B, Ts, -1)
        output_sph[np.isnan(output_sph)] = 0

        # rotate azimuth < -180 to positive
        if np.min(output_sph) < -180.0:
            output_sph[output_sph < -180.0] = output_sph[output_sph < -180.0] + 360
        elif np.max(output_sph) > 180.0:
            output_sph[output_sph > 180.0] = output_sph[output_sph > 180.0] - 360

        output_sph_array = cart2sph_array(output_xyz)
        output_sph_array = output_sph_array * 180 / np.pi

        assert np.allclose(output_sph, output_sph_array, atol=1e-05)

        sed_labels = output_class.T
        x_norm = x_norm.T
        y_norm = y_norm.T
        z_norm = z_norm.T

        orig_x, orig_y, orig_z = sph2cart(
            orig_azi * np.pi / 180, orig_ele * np.pi / 180, r=1
        )

        assert orig_x.shape[0] == num_classes
        assert orig_y.shape[0] == num_classes
        assert orig_z.shape[0] == num_classes

        doa_labels = np.concatenate((x_norm, y_norm, z_norm), axis=-1)

        assert doa_labels.shape[1] == 3 * num_classes
        assert sed_labels.shape[1] == num_classes

        doa_labels_100ms = np.concatenate((orig_x, orig_y, orig_z), axis=0).T
        sed_labels_100ms = orig_class.T

        doa_labels_100ms[np.isnan(doa_labels_100ms)] = 0
        sed_labels_100ms[np.isnan(sed_labels_100ms)] = 0

        assert doa_labels_100ms.shape[1] == 3 * num_classes
        assert sed_labels_100ms.shape[1] == num_classes

        class_dict.append({"sed_labels": sed_labels, "doa_labels": doa_labels})

        # save for vizualization
        if vizualize_figs:
            output_class = output_class.astype(np.float32)
            output_class[mask_class] = np.nan

        dict_files[f"{set_filename}"] = class_dict

        if vizualize_figs:
            fig, axs = plt.subplots(3, 3, figsize=(20, 12))

            # ELE
            axs[0, 0].set_title("orig-ele 100ms")
            for i in range(num_classes):
                axs[0, 0].plot(orig_ele[i], label=f"orig-{i}", linewidth=2.5)
            axs[1, 0].set_title("interp-ele pool 20ms")
            for i in range(num_classes):
                axs[1, 0].plot(output_ele[i].T, label=f"interp-pool-{i}", linewidth=2.5)
            axs[2, 0].set_title(f"interp-ele 1/{FS_TARGET}s")
            for i in range(num_classes):
                axs[2, 0].plot(array_ele[i], label=f"interp-{i}", linewidth=2.5)

            # AZI
            axs[0, 1].set_title("orig-azi 100ms")
            for i in range(num_classes):
                axs[0, 1].plot(orig_azi[i], label=f"orig-{i}", linewidth=2.5)
            axs[1, 1].set_title("interp-azi pool 20ms")
            for i in range(num_classes):
                axs[1, 1].plot(output_azi[i].T, label=f"interp-pool-{i}", linewidth=2.5)
            axs[2, 1].set_title(f"interp-azi 1/{FS_TARGET}s")
            for i in range(num_classes):
                axs[2, 1].plot(array_azi[i], label=f"interp-{i}", linewidth=2.5)

            # CLASS
            axs[0, 2].set_title("orig-class 100ms")
            for i in range(num_classes):
                axs[0, 2].plot(i * orig_class[i].T, label=f"orig-{i}", linewidth=2.5)
            axs[1, 2].set_title("interp-class pool 20ms")
            for i in range(num_classes):
                axs[1, 2].plot(
                    i * output_class[i].T, label=f"interp-pool-{i}", linewidth=2.5
                )
            axs[2, 2].set_title(f"interp-class 1/{FS_TARGET}s")
            for i in range(num_classes):
                axs[2, 2].plot(i * array_class[i].T, label=f"interp-{i}", linewidth=2.5)

            fig.tight_layout()

            if set_filename in " ".join(X_train) and train:
                fig_filename = f"{save_folder}/train_figs/{set_filename}.png"
            elif set_filename in " ".join(X_valid):
                fig_filename = f"{save_folder}/valid_figs/{set_filename}.png"
            else:
                fig_filename = f"{save_folder}/test_figs/{set_filename}.png"

            plt.savefig(fig_filename, bbox_inches="tight")
            plt.close()

        # save waves to folder
        if set_filename in " ".join(X_train) and train:
            save_filename = f"{save_folder}/train/{set_filename}.hdf"
        elif set_filename in " ".join(X_valid):
            save_filename = f"{save_folder}/valid/{set_filename}.hdf"
        else:
            save_filename = f"{save_folder}/test/{set_filename}.hdf"

        wav_names.append(save_filename)

        with h5py.File(save_filename, "w") as f:
            wav_set = f.create_dataset("wav", data=wav_resampled)
            wav_set.attrs["sample_rate"] = FS_TARGET

            f.create_dataset("sed_labels", data=sed_labels)
            f.create_dataset("doa_labels", data=doa_labels)
            f.create_dataset("sed_labels_100ms", data=sed_labels_100ms)
            f.create_dataset("doa_labels_100ms", data=doa_labels_100ms)

    return dict_files, wav_names


@hydra.main(
    version_base=None, config_path=f"{ROOT_DIR}/conf", config_name="config_ft_tau2020"
)
def finetuning_preprocess_data_tau2020(cfg: DictConfig) -> None:
    params = cfg["ft_dataset_tau2020"]

    seldnet_window = int(FS_TARGET * params["window_in_s"])
    logger.info(f"SELDnet window: {seldnet_window}")

    seldnet_window_t = seldnet_window / FS_TARGET

    logger.info(f"SELDnet window (s): {seldnet_window_t}")
    logger.info(
        f"SELDnet window {DEF_SAMPLE_RATE} Hz: {int((seldnet_window / FS_TARGET) * DEF_SAMPLE_RATE)}"
    )

    input_lengths = torch.tensor(seldnet_window)

    if "spectrogram" in params:
        hop_len_s = params["spectrogram"]["hop_len_s"]  # 5ms
        hop_len = int(hop_len_s * FS_TARGET)
        win_len = 2 * hop_len
        n_fft = _next_greater_power_of_2(win_len)

        spec_transform = Spectrogram(
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            power=None,
            pad_mode="constant",
        )
        output_lengths = get_feat_extract_output_lengths_spec(
            CONV_FEATURE_LAYERS_SPEC, spec_transform, input_lengths
        ).tolist()
    else:
        output_lengths = get_feat_extract_output_lengths(
            CONV_FEATURE_LAYERS, input_lengths
        ).tolist()
        spec_transform = None

    logger.info(f"output_lengths: {output_lengths}")
    logger.info(f"s: {seldnet_window_t / output_lengths}")
    logger.info(f"ms: {seldnet_window_t / output_lengths * 1000}")

    if spec_transform is not None:
        save_folder = (
            f"{params['save_folder']}/tau_nigens_2020_foa_ft_"
            f"val_split_xyz_cart_20mst_interp_spec_hdf"
        )
        manifest_folder = (
            f"{params['manifest_folder']}/tau_nigens_2020_foa_ft_"
            f"val_split_xyz_cart_20mst_interp_spec_hdf"
        )
    else:
        save_folder = f"{params['save_folder']}/tau_nigens_2020_foa_ft_val_split_xyz_cart_20mst_interp_hdf"
        manifest_folder = f"{params['manifest_folder']}/tau_nigens_2020_foa_ft_val_split_xyz_cart_20mst_interp_hdf"

    logger.info(f"save_folder: {save_folder}")
    logger.info(f"manifest_folder: {manifest_folder}")

    if os.path.isdir(f"{save_folder}"):
        shutil.rmtree(f"{save_folder}")

    if os.path.isdir(f"{manifest_folder}"):
        shutil.rmtree(f"{manifest_folder}")

    os.makedirs(f"{save_folder}/train")
    os.makedirs(f"{save_folder}/valid")
    os.makedirs(f"{save_folder}/test")

    if params.get("vizualize_figs", False):
        os.makedirs(f"{save_folder}/train_figs")
        os.makedirs(f"{save_folder}/valid_figs")
        os.makedirs(f"{save_folder}/test_figs")

    min_list_ele = []
    min_list_azi = []
    max_list_ele = []
    max_list_azi = []

    csv_files = glob.glob(f"{params['metadata_dev_path']}/**/*.csv", recursive=True)

    assert len(csv_files) > 0

    for csv_filename in csv_files:
        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df["start_time"] = df[0] * TEMPORAL_RESOLUTION
        df["end_time"] = df[0] * TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: "sound_event_recording"}, inplace=True)
        df.rename(columns={0: "frame_number"}, inplace=True)
        df.rename(columns={3: "azi"}, inplace=True)
        df.rename(columns={4: "ele"}, inplace=True)

        min_list_ele.append(df["ele"].min())
        min_list_azi.append(df["azi"].min())

        max_list_ele.append(df["ele"].max())
        max_list_azi.append(df["azi"].max())

    min_value_ele = min(min_list_ele)
    min_value_azi = min(min_list_azi)

    max_value_ele = max(max_list_ele)
    max_value_azi = max(max_list_azi)

    logger.info(f"min_value ele: {min_value_ele}")
    logger.info(f"min_value azi: {min_value_azi}")

    logger.info(f"max_value ele: {max_value_ele}")
    logger.info(f"max_value azi: {max_value_azi}")

    min_value = min(min(min_list_ele), min(min_list_azi))
    max_value = max(max(max_list_ele), max(max_list_azi))

    logger.info(f"min_value: {min_value}")
    logger.info(f"max_value: {max_value}")

    foa_wav_files = glob.glob(f"{params['foa_dev_path']}/**/*.wav", recursive=True)

    assert len(foa_wav_files) > 0

    logger.info(f"csv_files size: {len(foa_wav_files)}")
    logger.info(f"wav foa info: {torchaudio.info(foa_wav_files[0])}")

    X_list = []
    unique_classes = []

    valid_splits = [VALID_SPLIT]

    for csv_filename in csv_files:
        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df["start_time"] = df[0] * TEMPORAL_RESOLUTION
        df["end_time"] = df[0] * TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: "sound_event_recording"}, inplace=True)
        df.rename(columns={0: "frame_number"}, inplace=True)
        df.rename(columns={3: "azi"}, inplace=True)
        df.rename(columns={4: "ele"}, inplace=True)

        classes = df["sound_event_recording"].tolist()

        unique_classes.append(classes)

        filename = os.path.splitext(os.path.basename(csv_filename))[0]

        X_list.append(f"{filename}")

    logger.info(f"X_list size: {len(X_list)}")

    X_list = ["foa_dev_" + i for i in X_list]

    logger.info(f"X_list size: {len(X_list)}")
    logger.info(f"X_list: {X_list[0:5]}")

    X_train = []
    X_valid = []
    for xi in X_list:
        if xi.split("_")[2] in valid_splits:
            X_valid.append(xi)
        else:
            X_train.append(xi)

    logger.info(f"X_train size: {len(X_train)}")
    logger.info(f"X_train: {X_train[0:5]}")

    logger.info(f"X_valid size: {len(X_valid)}")
    logger.info(f"X_valid: {X_valid[0:5]}")

    unique_classes = np.unique(np.concatenate(unique_classes))

    logger.info(f"unique_classes: {unique_classes}")

    num_classes = len(unique_classes)

    assert num_classes == len(unique_classes_dict)
    assert Counter(list(unique_classes_dict.values())) == Counter(unique_classes)

    logger.info(f"num_classes: {num_classes}")

    dict_files_dev, wav_names_dev = preprocess_waves_metadata(
        f"{params['metadata_dev_path']}",
        f"{params['foa_dev_path']}",
        num_classes=num_classes,
        unique_classes=unique_classes,
        seldnet_window=seldnet_window,
        save_folder=save_folder,
        X_train=X_train,
        X_valid=X_valid,
        train=True,
        vizualize_figs=params.get("vizualize_figs", False),
        spec_transform=spec_transform,
    )

    dict_files_eval, wav_names_eval = preprocess_waves_metadata(
        f"{params['metadata_eval_path']}",
        f"{params['foa_eval_path']}",
        num_classes=num_classes,
        unique_classes=unique_classes,
        seldnet_window=seldnet_window,
        save_folder=save_folder,
        train=False,
        vizualize_figs=params.get("vizualize_figs", False),
        spec_transform=spec_transform,
    )

    for key in dict_files_dev:
        target_dict = dict_files_dev[key]

        for i in range(len(target_dict)):
            assert target_dict[i]["sed_labels"].shape[1] == num_classes, (
                f"key={key}, i={i}, {target_dict[i]['sed_labels'].shape}"
            )
            assert target_dict[i]["doa_labels"].shape[1] == 3 * num_classes

            doa_labels = target_dict[i]["doa_labels"]

            ts = doa_labels.shape[0]

            sed_labels = target_dict[i]["sed_labels"]

            doa_labels = np.transpose(
                doa_labels.reshape((ts, DOA_SIZE, num_classes)), (0, 2, 1)
            )

            x = doa_labels[:, :, 0]
            y = doa_labels[:, :, 1]
            z = doa_labels[:, :, 2]

            r = np.sqrt(x**2 + y**2 + z**2)

            r_sel = r[r != 0]

            assert sed_labels.shape[1] == num_classes
            assert doa_labels.shape[1] == num_classes
            assert doa_labels.shape[2] == DOA_SIZE

            assert np.allclose(r_sel, [1.0] * len(r_sel), atol=1e-05), r_sel

    for key in dict_files_eval:
        target_dict = dict_files_eval[key]

        for i in range(len(target_dict)):
            assert target_dict[i]["sed_labels"].shape[1] == num_classes, (
                f"key={key}, i={i}, {target_dict[i]['sed_labels'].shape}"
            )
            assert target_dict[i]["doa_labels"].shape[1] == 3 * num_classes

            doa_labels = target_dict[i]["doa_labels"]

            ts = doa_labels.shape[0]

            sed_labels = target_dict[i]["sed_labels"]

            doa_labels = np.transpose(
                doa_labels.reshape((ts, DOA_SIZE, num_classes)), (0, 2, 1)
            )

            x = doa_labels[:, :, 0]
            y = doa_labels[:, :, 1]
            z = doa_labels[:, :, 2]

            r = np.sqrt(x**2 + y**2 + z**2)

            r_sel = r[r != 0]

            assert sed_labels.shape[1] == num_classes
            assert doa_labels.shape[1] == num_classes
            assert doa_labels.shape[2] == DOA_SIZE

            assert np.allclose(r_sel, [1.0] * len(r_sel), atol=1e-05), r_sel

    dict_targets_files_dev = list(itertools.chain(*dict_files_dev.values()))
    dict_targets_files_eval = list(itertools.chain(*dict_files_eval.values()))

    assert len(dict_targets_files_dev) == len(wav_names_dev)
    assert len(dict_targets_files_eval) == len(wav_names_eval)

    logger.info(f"dict_targets_files_dev size: {len(dict_targets_files_dev)}")
    logger.info(f"wav_names_dev size: {len(wav_names_dev)}")

    logger.info(f"dict_targets_files_eval size: {len(dict_targets_files_eval)}")
    logger.info(f"wav_names_eval size: {len(wav_names_eval)}")

    logger.info(f"dict_targets_files_dev: {dict_targets_files_dev[0:5]}")
    logger.info(f"wav_names_dev: {wav_names_dev[0:5]}")

    dict_targets_dev = []

    for i, i_dict in enumerate(dict_targets_files_dev):
        new_dict = copy.deepcopy(i_dict)

        assert new_dict["sed_labels"].shape[1] == num_classes
        assert new_dict["doa_labels"].shape[1] == 3 * num_classes

        new_dict["sed_labels"] = new_dict["sed_labels"].tolist()
        new_dict["doa_labels"] = new_dict["doa_labels"].tolist()

        dict_targets_dev.append(new_dict)

    dict_targets_eval = []

    for i, i_dict in enumerate(dict_targets_files_eval):
        new_dict = copy.deepcopy(i_dict)

        assert new_dict["sed_labels"].shape[1] == num_classes
        assert new_dict["doa_labels"].shape[1] == 3 * num_classes

        new_dict["sed_labels"] = new_dict["sed_labels"].tolist()
        new_dict["doa_labels"] = new_dict["doa_labels"].tolist()

        dict_targets_eval.append(new_dict)

    logger.info(f"wav_names_dev size: {len(wav_names_dev)}")
    logger.info(f"dict_targets_dev size: {len(dict_targets_dev)}")

    dict_target_names = {}
    for i in range(len(dict_targets_dev)):
        dict_target_names[os.path.basename(wav_names_dev[i])] = dict_targets_dev[i]

    logger.info(f"dict_target_names size: {len(dict_target_names)}")

    dict_target_names_test = {}
    for i in range(len(dict_targets_eval)):
        dict_target_names_test[os.path.basename(wav_names_eval[i])] = dict_targets_eval[
            i
        ]

    logger.info(f"dict_target_names_test size: {len(dict_target_names_test)}")

    logger.info(f"dict_targets_files_dev size: {len(dict_targets_files_dev)}")
    logger.info(f"dict_targets_dev size: {len(dict_targets_dev)}")

    saved_wav_files = glob.glob(f"{save_folder}/**/*.hdf", recursive=True)

    logger.info(f"saved_wav_files size: {len(saved_wav_files)}")
    logger.info(f"saved_wav_files: {saved_wav_files[0:5]}")

    saved_wav_files_train = glob.glob(f"{save_folder}/train/**/*.hdf", recursive=True)

    logger.info(f"saved_wav_files_train size: {len(saved_wav_files_train)}")

    logger.info(f"saved_wav_files_train: {saved_wav_files_train[0:5]}")

    saved_wav_files_valid = glob.glob(f"{save_folder}/valid/**/*.hdf", recursive=True)

    logger.info(f"saved_wav_files_valid size: {len(saved_wav_files_valid)}")
    logger.info(f"saved_wav_files_valid: {saved_wav_files_valid[0:5]}")

    saved_wav_files_test = glob.glob(f"{save_folder}/test/**/*.hdf", recursive=True)

    logger.info(f"saved_wav_files_test size: {len(saved_wav_files_test)}")

    logger.info(f"dict_targets_eval size: {len(dict_targets_eval)}")

    logger.info(f"saved_wav_files_test: {saved_wav_files_test[0:5]}")

    gen_tsv_manifest(save_folder, manifest_folder, dset="train", ext="hdf")
    gen_tsv_manifest(save_folder, manifest_folder, dset="valid", ext="hdf")

    with open(f"{manifest_folder}/train.tsv", "r") as tsv:
        train_size = len([line.strip().split("\t") for line in tsv]) - 1

    with open(f"{manifest_folder}/valid.tsv", "r") as tsv:
        valid_size = len([line.strip().split("\t") for line in tsv]) - 1

    assert train_size > 0 and valid_size > 0

    f = open(f"{manifest_folder}/train.tsv", "r")
    file_contents = f.read()
    logger.info(file_contents[0:590])
    f.close()

    f = open(f"{manifest_folder}/valid.tsv", "r")
    file_contents = f.read()
    logger.info(file_contents[0:534])
    f.close()

    train_tsv = []
    with open(f"{manifest_folder}/train.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                train_tsv.append(items[0])

    valid_tsv = []
    with open(f"{manifest_folder}/valid.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                valid_tsv.append(items[0])

    assert train_size == len(train_tsv) and valid_size == len(valid_tsv)

    logger.info(f"train size: {train_size}")
    logger.info(f"train_tsv size: {len(train_tsv)}")
    logger.info(f"valid size: {valid_size}")
    logger.info(f"valid_tsv size: {len(valid_tsv)}")

    logger.info(f"train_tsv: {train_tsv[0:5]}")
    logger.info(f"valid_tsv: {valid_tsv[0:5]}")

    dict_targets_train = []
    for tsv_file in train_tsv:
        dict_targets_train.append(dict_target_names[tsv_file])

    dict_targets_valid = []
    for tsv_file in valid_tsv:
        dict_targets_valid.append(dict_target_names[tsv_file])

    assert train_size == len(dict_targets_train)
    assert valid_size == len(dict_targets_valid)

    logger.info(f"train size: {train_size}")
    logger.info(f"dict_targets_train size: {len(dict_targets_train)}")

    logger.info(f"valid size: {valid_size}")
    logger.info(f"dict_targets_valid size: {len(dict_targets_valid)}")

    sizes_train = []
    with open(f"{manifest_folder}/train.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                sizes_train.append(int(items[1]))

    logger.info(f"train - min sample size: {min(sizes_train)}")
    logger.info(f"train - max sample size: {max(sizes_train)}")

    sizes_valid = []
    with open(f"{manifest_folder}/valid.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                sizes_valid.append(int(items[1]))

    logger.info(f"valid - min sample size: {min(sizes_valid)}")
    logger.info(f"valid - max sample size: {max(sizes_valid)}")

    gen_tsv_manifest(save_folder, manifest_folder, dset="test", ext="hdf")

    with open(f"{manifest_folder}/test.tsv", "r") as tsv:
        test_size = len([line.strip().split("\t") for line in tsv]) - 1

    assert test_size > 0

    f = open(f"{manifest_folder}/test.tsv", "r")
    file_contents = f.read()
    logger.info(file_contents[0:2000])
    f.close()

    test_tsv = []
    with open(f"{manifest_folder}/test.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                test_tsv.append(items[0])

    sizes_test = []
    with open(f"{manifest_folder}/test.tsv", "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                sizes_test.append(int(items[1]))

    logger.info(f"test - min sample size: {min(sizes_test)}")
    logger.info(f"test - max sample size: {max(sizes_test)}")

    assert test_size == len(test_tsv)

    logger.info(f"test size: {test_size}")
    logger.info(f"test_tsv size: {len(test_tsv)}")
    logger.info(f"test_tsv: {test_tsv[0:5]}")

    dict_targets_test = []
    for tsv_file in test_tsv:
        dict_targets_test.append(dict_target_names_test[tsv_file])

    logger.info(f"test size: {test_size}")
    logger.info(f"dict_targets_test size: {len(dict_targets_test)}")

    assert test_size == len(dict_targets_test)


if __name__ == "__main__":
    finetuning_preprocess_data_tau2020()
