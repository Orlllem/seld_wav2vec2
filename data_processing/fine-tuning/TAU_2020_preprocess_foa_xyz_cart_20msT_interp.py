

import copy
import gc
import glob
import itertools
import json
import logging
import os
import shutil

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig
from tqdm import tqdm
from utils import (cart2sph, cart2sph_array, gen_tsv_manifest,
                   get_feat_extract_output_lengths, linear_interp,
                   remove_overlap_same_class, sph2cart)

logger = logging.getLogger("preprocessing-ft-tau2020-interp")

ROOT_DIR = os.path.abspath(os.curdir)
DEF_SAMPLE_RATE = 24000
FS_TARGET = 16000
MIN_LENGTH = 400
DOA_SIZE = 3
TEMPORAL_RESOLUTION = 100*1e-3
VALID_SPLIT = 'fold6'
CONV_FEATURE_LAYERS = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
MASK_ID = -1000
INTERP_FUN = "linear"


def preprocess_waves_metadata(metadata_dir, wav_dir, num_classes, unique_classes,
                              seldnet_window, stride, save_folder,
                              X_train=[], X_valid=[], train=True,
                              vizualize_figs=False):

    dict_files = {}
    wav_names = []

    csv_files = glob.glob(f'{metadata_dir}/**/*.csv',
                          recursive=True)

    for csv_filename in tqdm(csv_files):

        # logger.info(f"csv_filename: {csv_filename}")

        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df['start_time'] = df[0]*TEMPORAL_RESOLUTION
        df['end_time'] = df[0]*TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: 'sound_event_recording'}, inplace=True)
        df.rename(columns={0: 'frame_number'}, inplace=True)
        df.rename(columns={3: 'azi'}, inplace=True)
        df.rename(columns={4: 'ele'}, inplace=True)

        # classes = df["sound_event_recording"].tolist()

        # remove duplicate same class on same time-step
        df = remove_overlap_same_class(df)

        filename = os.path.splitext(os.path.basename(csv_filename))[0]

        wav_filename = f'{wav_dir}/{filename}.wav'

        wav, curr_sample_rate = torchaudio.load(wav_filename)

        assert curr_sample_rate == DEF_SAMPLE_RATE, f"{curr_sample_rate}!={DEF_SAMPLE_RATE}"

        wav_info = torchaudio.info(wav_filename)

        resampler = T.Resample(curr_sample_rate, FS_TARGET)

        wav_resampled = resampler(wav)

        set_filename = f"{os.path.basename(wav_dir)}_{filename}"

        if set_filename in ' '.join(X_train) and train:
            wav_split_array = []
            wavT = wav_resampled.T
            for i in range(0, len(wavT), stride):
                wav_split_array.append(wavT[i: i+seldnet_window].T)
        else:
            wav_split_array = []
            wavT = wav_resampled.T
            for i in range(0, len(wavT), seldnet_window):
                wav_split_array.append(wavT[i: i+seldnet_window].T)

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

                start = int(np.floor(start_time*FS_TARGET))
                end = int(np.floor(end_time*FS_TARGET))

                ele_array = [np.nan]*(end - start)
                ele_array[0] = ele

                azi_array = [np.nan]*(end - start)
                azi_array[0] = azi

                ele_list.append(ele)
                azi_list.append(azi)
                start_list.append(start)
                try:
                    fm = df_class["frame_number"][df_class_index[i]]
                    fm_next = df_class["frame_number"][df_class_index[i+1]]

                    if fm - fm_next != -1:
                        ele_array[-1] = linear_interp(end,
                                                      start_list[-2:], ele_list[-2:])
                        azi_array[-1] = linear_interp(end,
                                                      start_list[-2:], azi_list[-2:])
                except IndexError:
                    ele_array[-1] = linear_interp(end,
                                                  start_list[-2:], ele_list[-2:])
                    azi_array[-1] = linear_interp(end,
                                                  start_list[-2:], azi_list[-2:])
                events.append({"range": np.arange(start, end),
                               "ele": ele_array,
                               "azi": azi_array})

            dict_events[class_i] = events

        classes_list = list(dict_events.keys())

        assert len(classes_list) == len(
            unique_classes), f"{classes_list}!={unique_classes}"

        df_array_ele = []
        df_array_azi = []
        for class_i in unique_classes:

            array_empty_ele = np.zeros(wav_resampled.shape[1])
            array_empty_ele[:] = MASK_ID

            array_empty_azi = np.zeros(wav_resampled.shape[1])
            array_empty_azi[:] = MASK_ID

            for event in dict_events[class_i]:
                array_empty_ele[event["range"]] = event["ele"]
                array_empty_azi[event["range"]] = event["azi"]

            df_array_ele.append(array_empty_ele)
            df_array_azi.append(array_empty_azi)

        df_array_ele = np.vstack(df_array_ele)
        df_array_azi = np.vstack(df_array_azi)

        mask_ele = df_array_ele == MASK_ID
        mask_azi = df_array_azi == MASK_ID

        df_array_ele = pd.DataFrame(df_array_ele).T
        df_array_azi = pd.DataFrame(df_array_azi).T

        for i in range(len(unique_classes)):

            if np.isnan(df_array_ele[i].values).any():
                sel_df_array_ele = df_array_ele[i][~mask_ele[i]].interpolate(method=INTERP_FUN,
                                                                             limit_area="inside")
                df_array_ele[i][~mask_ele[i]] = sel_df_array_ele

            assert not np.isnan(df_array_ele[i].values).any()

            if np.isnan(df_array_azi[i].values).any():
                sel_df_array_azi = df_array_azi[i][~mask_azi[i]].interpolate(method=INTERP_FUN,
                                                                             limit_area="inside")
                df_array_azi[i][~mask_azi[i]] = sel_df_array_azi

            assert not np.isnan(df_array_azi[i].values).any()

        df_array_ele = df_array_ele.T.values
        df_array_azi = df_array_azi.T.values

        assert not np.isnan(df_array_ele).all()
        assert not np.isnan(df_array_azi).all()

        df_array_ele[mask_ele] = np.nan
        df_array_azi[mask_azi] = np.nan

        if set_filename in ' '.join(X_train) and train:
            df_split_array_ele = []
            df_ele = df_array_ele.T
            for i in range(0, len(df_ele), stride):
                df_split_array_ele.append(df_ele[i: i+seldnet_window].T)

            df_split_array_azi = []
            df_azi = df_array_azi.T
            for i in range(0, len(df_azi), stride):
                df_split_array_azi.append(df_azi[i: i+seldnet_window].T)
        else:
            df_split_array_ele = []
            df_ele = df_array_ele.T
            for i in range(0, len(df_ele), seldnet_window):
                df_split_array_ele.append(df_ele[i: i+seldnet_window].T)

            df_split_array_azi = []
            df_azi = df_array_azi.T
            for i in range(0, len(df_azi), seldnet_window):
                df_split_array_azi.append(df_azi[i: i+seldnet_window].T)

        class_dict = []
        x_array_list = []
        y_array_list = []
        z_array_list = []
        class_array_list = []

        for j in range(len(df_split_array_ele)):
            ele = df_split_array_ele[j]
            azi = df_split_array_azi[j]

            input_length = (torch.tensor(ele.shape[1]))

            # make sure that the wave lenghth is equal to x, z length
            assert wav_split_array[j].shape[
                1] == input_length, f"wav_split_array[j]: {wav_split_array[j].shape}, input_length: {input_length}"

            if (input_length >= MIN_LENGTH) or (not train):

                if input_length < MIN_LENGTH and (not train):

                    logger.info(
                        f"padding ele, azi {set_filename} index {j}, ele: {ele.shape} , azi: {azi.shape}")

                    ele = np.ascontiguousarray(ele, dtype=np.float32)
                    ele.resize(num_classes, MIN_LENGTH)

                    azi = np.ascontiguousarray(azi, dtype=np.float32)
                    azi.resize(num_classes, MIN_LENGTH)

                    logger.info(
                        f"padded ele, azi {set_filename} index {j}, ele: {ele.shape} , azi: {azi.shape}")

                    input_length = (torch.tensor(ele.shape[1]))

                output_length = get_feat_extract_output_lengths(CONV_FEATURE_LAYERS,
                                                                input_length).tolist()

                pooler = nn.AdaptiveMaxPool1d(output_length)

                # convert ele, azi -> x,y,z
                x, y, z = sph2cart(azi*np.pi/180, ele*np.pi/180, r=1)

                azimuth, elevation, r = cart2sph(x, y, z)

                azimuth = azimuth*180/np.pi
                elevation = elevation*180/np.pi

                r_sel = r[~np.isnan(r)]
                azi_sel = azi.copy()
                azi_sel[np.isnan(azi_sel)] = 0.0
                azimuth_sel = azimuth.copy()
                azimuth_sel[np.isnan(azimuth_sel)] = 0.0

                ele_sel = ele.copy()
                ele_sel[np.isnan(ele_sel)] = 0.0
                elevation_sel = elevation.copy()
                elevation_sel[np.isnan(elevation_sel)] = 0.0

                # rotate azimuth < -180 to positive
                if np.min(azi_sel) < -180.0:
                    azi_sel[azi_sel < -180.0] = azi_sel[azi_sel < -180.0] + 360
                elif np.max(azi_sel) > 180.0:
                    # rotate azimuth > 180 to positive
                    azi_sel[azi_sel > 180.0] = azi_sel[azi_sel > 180.0] - 360

                assert np.allclose(r_sel, [1.0]*len(r_sel), atol=1e-05)
                assert np.allclose(azimuth_sel, azi_sel, atol=1e-05)
                assert np.allclose(elevation_sel, ele_sel, atol=1e-05)

                output_ele = pooler(torch.from_numpy(ele)).cpu().numpy()
                output_azi = pooler(torch.from_numpy(azi)).cpu().numpy()

                output_ele_class = output_ele.copy()
                output_azi_class = output_azi.copy()

                output_ele_class[~np.isnan(output_ele_class)] = 1
                output_ele_class[np.isnan(output_ele_class)] = 0

                output_azi_class[~np.isnan(output_azi_class)] = 1
                output_azi_class[np.isnan(output_azi_class)] = 0

                # get classes binary matrix
                output_class = (output_ele_class.astype(bool) |
                                output_azi_class.astype(bool)).astype(int)

                output_xx, output_yy, output_zz = sph2cart(
                    output_azi*np.pi/180, output_ele*np.pi/180, r=1)

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
                    np.stack((x_norm, y_norm, z_norm), axis=0).T, axis=0)

                B = output_xyz.shape[0]
                Ts = output_xyz.shape[1]

                output_sph = np.expand_dims(
                    np.stack((output_ele, output_azi), axis=0).T, axis=0).reshape(B, Ts, -1)
                output_sph[np.isnan(output_sph)] = 0

                # rotate azimuth < -180 to positive
                if np.min(output_sph) < -180.0:
                    output_sph[output_sph < -
                               180.0] = output_sph[output_sph < -180.0] + 360
                elif np.max(output_sph) > 180.0:
                    output_sph[output_sph >
                               180.0] = output_sph[output_sph > 180.0] - 360

                output_sph_array = cart2sph_array(output_xyz)
                output_sph_array = output_sph_array*180/np.pi

                assert np.allclose(output_sph, output_sph_array, atol=1e-05)

                sed_labels = output_class.T
                x_norm = x_norm.T
                y_norm = y_norm.T
                z_norm = z_norm.T

                doa_labels = np.concatenate((x_norm, y_norm, z_norm), axis=-1)

                assert doa_labels.shape[1] == 3*num_classes
                assert sed_labels.shape[1] == num_classes

                class_dict.append(
                    {"sed_labels": sed_labels, "doa_labels": doa_labels})

                x_array_list.append(x_norm)
                y_array_list.append(y_norm)
                z_array_list.append(z_norm)
                class_array_list.append(sed_labels)

                if vizualize_figs:
                    fig, axs = plt.subplots(2, 2)

                    axs[0, 0].set_title("orig-ele")
                    for i in range(num_classes):
                        axs[0, 0].plot(
                            ele[i], label=f"orig-ele-{i}", linewidth=2.5)
                    axs[1, 0].set_title("interp-ele")
                    for i in range(num_classes):
                        axs[1, 0].plot(output_ele[i].T,
                                       label=f"max-pool-{i}", linewidth=2.5)
                    axs[0, 1].set_title("orig-azi")
                    for i in range(num_classes):
                        axs[0, 1].plot(
                            azi[i], label=f"orig-azi-{i}", linewidth=2.5)
                    axs[1, 1].set_title("interp-azi")
                    for i in range(num_classes):
                        axs[1, 1].plot(output_azi[i].T,
                                       label=f"max-pool-{i}", linewidth=2.5)
                    fig.tight_layout()

                    if set_filename in ' '.join(X_train) and train:
                        fig_filename = f"{save_folder}/train_figs/{set_filename}_index_{j}.png"
                    elif set_filename in ' '.join(X_valid):
                        fig_filename = f"{save_folder}/valid_figs/{set_filename}_index_{j}.png"
                    else:
                        fig_filename = f"{save_folder}/test_figs/{set_filename}_index_{j}.png"

                    plt.savefig(fig_filename, bbox_inches="tight")
                    plt.close()

                # clear memory
                del ele, azi, azimuth, elevation, output_ele, output_azi, output_sph_array, output_sph
                gc.collect()

            else:
                logger.info(
                    f"discarting target {set_filename} index {j}, ele: {ele.shape}, azi: {azi.shape}")

        dict_files[f"{set_filename}"] = class_dict

        # save waves to folder
        for k in range(len(wav_split_array)):
            if set_filename in ' '.join(X_train) and train:
                save_filename = f"{save_folder}/train/{set_filename}_index_{k}.wav"
            elif set_filename in ' '.join(X_valid):
                save_filename = f"{save_folder}/valid/{set_filename}_index_{k}.wav"
            else:
                save_filename = f"{save_folder}/test/{set_filename}_index_{k}.wav"

            wav_arr = wav_split_array[k]

            if (wav_arr.shape[1] >= MIN_LENGTH) or (not train):

                if (wav_arr.shape[1] < MIN_LENGTH) and (not train):

                    logger.info(
                        f"padding wav {set_filename} index {k}, length: {wav_arr.shape}")
                    wav_arr = np.ascontiguousarray(
                        wav_arr.cpu().numpy(), dtype=np.float32)
                    wav_arr.resize(4, MIN_LENGTH)

                    wav_arr = torch.from_numpy(wav_arr)

                    logger.info(
                        f"padded wav {set_filename} index {k}, length: {wav_arr.shape}")

                    logger.info(
                        "-----------------------------------------------------------------")

                input_length = (torch.tensor(wav_arr.shape[1]))
                output_length = get_feat_extract_output_lengths(CONV_FEATURE_LAYERS,
                                                                input_length).tolist()

                pooler = nn.AdaptiveMaxPool1d(output_length)

                pooler_wav_arr = pooler(wav_arr)

                pooler_x = pooler(torch.from_numpy(x_array_list[k]))
                pooler_y = pooler(torch.from_numpy(y_array_list[k]))
                pooler_z = pooler(torch.from_numpy(z_array_list[k]))
                pooler_class = pooler(torch.from_numpy(
                    class_array_list[k]).float())

                assert pooler_wav_arr.shape[1] == pooler_x.shape[
                    1], f"wav: {pooler_wav_arr.shape}, class: {pooler_x.shape}"
                assert pooler_wav_arr.shape[1] == pooler_y.shape[
                    1], f"wav: {pooler_wav_arr.shape}, class: {pooler_y.shape}"
                assert pooler_wav_arr.shape[1] == pooler_z.shape[
                    1], f"wav: {pooler_wav_arr.shape}, class: {pooler_z.shape}"
                assert pooler_wav_arr.shape[1] == pooler_class.shape[
                    1], f"wav: {pooler_wav_arr.shape}, class: {pooler_class.shape}"

                wav_names.append(f"{set_filename}_index_{k}.wav")
                torchaudio.save(save_filename, wav_arr, FS_TARGET,
                                bits_per_sample=wav_info.bits_per_sample)
            else:
                logger.info(
                    f"discarting wav {set_filename} index {k}, length: {wav_arr.shape}")
                logger.info(
                    "-----------------------------------------------------------------")

    # logger.info(f"finished csv_filename: {csv_filename}")
    # logger.info("----------------------------------")

    return dict_files, wav_names


@hydra.main(version_base=None, config_path=f"{ROOT_DIR}/conf",
            config_name="config")
def finetuning_preprocess_data_tau2020(cfg: DictConfig) -> None:

    params = cfg["ft_dataset_tau2020"]

    seldnet_window = int(FS_TARGET*params["window_in_s"])
    stride = int(FS_TARGET*params["stride_in_s"])

    logger.info(f"stride: {stride}")
    logger.info(f"stride (s): {stride/FS_TARGET}")
    logger.info(f"SELDnet window: {seldnet_window}")

    seldnet_window_t = seldnet_window/FS_TARGET

    logger.info(f"SELDnet window (s): {seldnet_window_t}")
    logger.info(
        f"SELDnet window 24000 Hz: {int((seldnet_window/FS_TARGET)*DEF_SAMPLE_RATE)}")

    input_lengths = (torch.tensor(seldnet_window))

    output_lengths = get_feat_extract_output_lengths(
        CONV_FEATURE_LAYERS, input_lengths).tolist()

    logger.info(f"output_lengths: {output_lengths}")
    logger.info(f"s: {seldnet_window_t/output_lengths}")
    logger.info(f"ms: {seldnet_window_t/output_lengths*1000}")

    save_folder = (f"{params['save_folder']}/tau_nigens_2020_foa_ft_chunk{seldnet_window}"
                   f"_stride{stride}_val_split_xyz_cart_20mst_interp")
    manifest_folder = (f"{params['manifest_folder']}/tau_nigens_2020_foa_ft_chunk{seldnet_window}"
                       f"_stride{stride}_val_split_xyz_cart_20mst_interp")

    logger.info(f"save_folder: {save_folder}")
    logger.info(f"manifest_folder: {manifest_folder}")

    if os.path.isdir(f'{save_folder}'):
        shutil.rmtree(f'{save_folder}')

    if os.path.isdir(f'{manifest_folder}'):
        shutil.rmtree(f'{manifest_folder}')

    os.makedirs(f'{save_folder}/train')
    os.makedirs(f'{save_folder}/valid')
    os.makedirs(f'{save_folder}/test')

    if params.get("vizualize_figs", False):
        os.makedirs(f'{save_folder}/train_figs')
        os.makedirs(f'{save_folder}/valid_figs')
        os.makedirs(f'{save_folder}/test_figs')

    min_list_ele = []
    min_list_azi = []
    max_list_ele = []
    max_list_azi = []

    csv_files = glob.glob(f"{params['metadata_dev_path']}/**/*.csv",
                          recursive=True)

    assert len(csv_files) > 0

    for csv_filename in csv_files:
        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df['start_time'] = df[0]*TEMPORAL_RESOLUTION
        df['end_time'] = df[0]*TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: 'sound_event_recording'}, inplace=True)
        df.rename(columns={0: 'frame_number'}, inplace=True)
        df.rename(columns={3: 'azi'}, inplace=True)
        df.rename(columns={4: 'ele'}, inplace=True)

        min_list_ele.append(df['ele'].min())
        min_list_azi.append(df['azi'].min())

        max_list_ele.append(df['ele'].max())
        max_list_azi.append(df['azi'].max())

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

    foa_wav_files = glob.glob(f"{params['foa_dev_path']}/**/*.wav",
                              recursive=True)

    assert len(foa_wav_files) > 0

    logger.info(f"csv_files size: {len(foa_wav_files)}")
    logger.info(f"wav foa info: {torchaudio.info(foa_wav_files[0])}")

    X_list = []
    unique_classes = []

    valid_splits = [VALID_SPLIT]

    for csv_filename in csv_files:

        df = pd.read_csv(csv_filename, index_col=False, header=None)

        df['start_time'] = df[0]*TEMPORAL_RESOLUTION
        df['end_time'] = df[0]*TEMPORAL_RESOLUTION + TEMPORAL_RESOLUTION

        df.rename(columns={1: 'sound_event_recording'}, inplace=True)
        df.rename(columns={0: 'frame_number'}, inplace=True)
        df.rename(columns={3: 'azi'}, inplace=True)
        df.rename(columns={4: 'ele'}, inplace=True)

        classes = df["sound_event_recording"].tolist()

        unique_classes.append(classes)

        filename = os.path.splitext(os.path.basename(csv_filename))[0]

        X_list.append(f"{filename}")

    logger.info(f"X_list size: {len(X_list)}")

    X_list = ["foa_dev_"+i for i in X_list]

    logger.info(f"X_list size: {len(X_list)}")
    logger.info(f"X_list: {X_list[0:5]}")

    X_train = []
    X_valid = []
    for xi in X_list:
        # logger.info(xi.split("_")[2])
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

    logger.info(f"num_classes: {num_classes}")

    dict_files_dev, wav_names_dev = preprocess_waves_metadata(f"{params['metadata_dev_path']}",
                                                              f"{params['foa_dev_path']}",
                                                              num_classes=num_classes,
                                                              unique_classes=unique_classes,
                                                              seldnet_window=seldnet_window, stride=stride,
                                                              save_folder=save_folder,
                                                              X_train=X_train, X_valid=X_valid,
                                                              train=True,
                                                              vizualize_figs=params.get("vizualize_figs", False))

    dict_files_eval, wav_names_eval = preprocess_waves_metadata(f"{params['metadata_eval_path']}",
                                                                f"{params['foa_eval_path']}",
                                                                num_classes=num_classes,
                                                                unique_classes=unique_classes,
                                                                seldnet_window=seldnet_window, stride=stride,
                                                                save_folder=save_folder,
                                                                train=False,
                                                                vizualize_figs=params.get("vizualize_figs", False))

    for key in dict_files_dev:
        target_dict = dict_files_dev[key]

        for i in range(len(target_dict)):
            assert target_dict[i]['sed_labels'].shape[
                1] == num_classes, f"key={key}, i={i}, {target_dict[i]['sed_labels'].shape}"
            assert target_dict[i]['doa_labels'].shape[1] == 3*num_classes

            doa_labels = target_dict[i]['doa_labels']

            ts = doa_labels.shape[0]

            sed_labels = target_dict[i]['sed_labels']

            doa_labels = np.transpose(doa_labels.reshape(
                (ts, DOA_SIZE, num_classes)), (0, 2, 1))

            x = doa_labels[:, :, 0]
            y = doa_labels[:, :, 1]
            z = doa_labels[:, :, 2]

            r = np.sqrt(x**2 + y**2 + z**2)

            r_sel = r[r != 0]

            assert sed_labels.shape[1] == num_classes
            assert doa_labels.shape[1] == num_classes
            assert doa_labels.shape[2] == DOA_SIZE

            assert np.allclose(
                r_sel, [1.0]*len(r_sel), atol=1e-05), r_sel

    for key in dict_files_eval:
        target_dict = dict_files_eval[key]

        for i in range(len(target_dict)):
            assert target_dict[i]['sed_labels'].shape[
                1] == num_classes, f"key={key}, i={i}, {target_dict[i]['sed_labels'].shape}"
            assert target_dict[i]['doa_labels'].shape[1] == 3*num_classes

            doa_labels = target_dict[i]['doa_labels']

            ts = doa_labels.shape[0]

            sed_labels = target_dict[i]['sed_labels']

            doa_labels = np.transpose(doa_labels.reshape(
                (ts, DOA_SIZE, num_classes)), (0, 2, 1))

            x = doa_labels[:, :, 0]
            y = doa_labels[:, :, 1]
            z = doa_labels[:, :, 2]

            r = np.sqrt(x**2 + y**2 + z**2)

            r_sel = r[r != 0]

            assert sed_labels.shape[1] == num_classes
            assert doa_labels.shape[1] == num_classes
            assert doa_labels.shape[2] == DOA_SIZE

            assert np.allclose(
                r_sel, [1.0]*len(r_sel), atol=1e-05), r_sel

    dict_targets_files_dev = list(itertools.chain(*dict_files_dev.values()))
    dict_targets_files_eval = list(itertools.chain(*dict_files_eval.values()))

    assert len(dict_targets_files_dev) == len(wav_names_dev)
    assert len(dict_targets_files_eval) == len(wav_names_eval)

    logger.info(f"dict_targets_files_dev size: {len(dict_targets_files_dev)}")
    logger.info(f"wav_names_dev size: {len(wav_names_dev)}")

    logger.info(
        f"dict_targets_files_eval size: {len(dict_targets_files_eval)}")
    logger.info(f"wav_names_eval size: {len(wav_names_eval)}")

    logger.info(f"dict_targets_files_dev: {dict_targets_files_dev[0:5]}")
    logger.info(f"wav_names_dev: {wav_names_dev[0:5]}")

    dict_targets_dev = []

    for i, i_dict in enumerate(dict_targets_files_dev):
        # filename = wav_names_dev[i]
        new_dict = copy.deepcopy(i_dict)

        assert new_dict['sed_labels'].shape[1] == num_classes
        assert new_dict['doa_labels'].shape[1] == 3*num_classes

        new_dict['sed_labels'] = new_dict['sed_labels'].tolist()
        new_dict['doa_labels'] = new_dict['doa_labels'].tolist()

        dict_targets_dev.append(new_dict)

    dict_targets_eval = []

    for i, i_dict in enumerate(dict_targets_files_eval):
        # filename = wav_names_eval[i]
        new_dict = copy.deepcopy(i_dict)

        assert new_dict['sed_labels'].shape[1] == num_classes
        assert new_dict['doa_labels'].shape[1] == 3*num_classes

        new_dict['sed_labels'] = new_dict['sed_labels'].tolist()
        new_dict['doa_labels'] = new_dict['doa_labels'].tolist()

        dict_targets_eval.append(new_dict)

    logger.info(f"wav_names_dev size: {len(wav_names_dev)}")
    logger.info(f"dict_targets_dev size: {len(dict_targets_dev)}")

    dict_target_names = {}
    for i in range(len(dict_targets_dev)):
        dict_target_names[wav_names_dev[i]] = dict_targets_dev[i]

    logger.info(f"dict_target_names size: {len(dict_target_names)}")

    dict_target_names_test = {}
    for i in range(len(dict_targets_eval)):
        dict_target_names_test[wav_names_eval[i]] = dict_targets_eval[i]

    logger.info(f"dict_target_names_test size: {len(dict_target_names_test)}")

    logger.info(f"dict_targets_files_dev size: {len(dict_targets_files_dev)}")
    logger.info(f"dict_targets_dev size: {len(dict_targets_dev)}")

    saved_wav_files = glob.glob(f'{save_folder}/**/*.wav',
                                recursive=True)

    logger.info(f"saved_wav_files size: {len(saved_wav_files)}")
    logger.info(f"saved_wav_files: {saved_wav_files[0:5]}")

    saved_wav_files_train = glob.glob(f'{save_folder}/train/**/*.wav',
                                      recursive=True)

    logger.info(f"saved_wav_files_train size: {len(saved_wav_files_train)}")

    logger.info(f"saved_wav_files_train: {saved_wav_files_train[0:5]}")

    saved_wav_files_valid = glob.glob(f'{save_folder}/valid/**/*.wav',
                                      recursive=True)

    logger.info(f"saved_wav_files_valid size: {len(saved_wav_files_valid)}")
    logger.info(f"saved_wav_files_valid: {saved_wav_files_valid[0:5]}")

    saved_wav_files_test = glob.glob(f'{save_folder}/test/**/*.wav',
                                     recursive=True)

    logger.info(f"saved_wav_files_test size: {len(saved_wav_files_test)}")

    logger.info(f"dict_targets_eval size: {len(dict_targets_eval)}")

    logger.info(f"saved_wav_files_test: {saved_wav_files_test[0:5]}")

    gen_tsv_manifest(save_folder, manifest_folder, dset="train", ext="wav")
    gen_tsv_manifest(save_folder, manifest_folder, dset="valid", ext="wav")

    with open(f'{manifest_folder}/train.tsv', 'r') as tsv:
        train_size = len([line.strip().split('\t') for line in tsv]) - 1
        # logger.info(train_size)

    with open(f'{manifest_folder}/valid.tsv', 'r') as tsv:
        valid_size = len([line.strip().split('\t') for line in tsv]) - 1
        # logger.info(valid_size)

    assert train_size > 0 and valid_size > 0

    f = open(f'{manifest_folder}/train.tsv', 'r')
    file_contents = f.read()
    logger.info(file_contents[0:590])
    f.close()

    f = open(f'{manifest_folder}/valid.tsv', 'r')
    file_contents = f.read()
    logger.info(file_contents[0:534])
    f.close()

    train_tsv = []
    with open(f'{manifest_folder}/train.tsv', "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                train_tsv.append(items[0])

    valid_tsv = []
    with open(f'{manifest_folder}/valid.tsv', "r") as tsv_file:
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

    with open(f'{manifest_folder}/train.json', 'w') as output_file:
        json.dump(dict_targets_train, output_file)

    with open(f'{manifest_folder}/valid.json', 'w') as output_file:
        json.dump(dict_targets_valid, output_file)

    sizes_train = []
    with open(f'{manifest_folder}/train.tsv', "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                # logger.info(items[1])
                sizes_train.append(int(items[1]))

    logger.info(f"train - min sample size: {min(sizes_train)}")
    logger.info(f"train - max sample size: {max(sizes_train)}")

    # logger.info(Counter(sizes_train))

    sizes_valid = []
    with open(f'{manifest_folder}/valid.tsv', "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                # logger.info(items[1])
                sizes_valid.append(int(items[1]))

    logger.info(f"valid - min sample size: {min(sizes_valid)}")
    logger.info(f"valid - max sample size: {max(sizes_valid)}")

    gen_tsv_manifest(save_folder, manifest_folder, dset="test", ext="wav")

    with open(f'{manifest_folder}/test.tsv', 'r') as tsv:
        test_size = len([line.strip().split('\t') for line in tsv]) - 1
        # logger.info(test_size)

    assert test_size > 0

    f = open(f'{manifest_folder}/test.tsv', 'r')
    file_contents = f.read()
    logger.info(file_contents[0:2000])
    f.close()

    test_tsv = []
    with open(f'{manifest_folder}/test.tsv', "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                test_tsv.append(items[0])

    sizes_test = []
    with open(f'{manifest_folder}/test.tsv', "r") as tsv_file:
        for line in tsv_file:
            items = line.strip().split("\t")
            if len(items) > 1:
                # logger.info(items[1])
                sizes_test.append(int(items[1]))

    logger.info(f"test - min sample size: {min(sizes_test)}")
    logger.info(f"test - max sample size: {max(sizes_test)}")

    # logger.info(Counter(sizes_test))

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

    with open(f'{manifest_folder}/test.json', 'w') as output_file:
        json.dump(dict_targets_test, output_file)


if __name__ == "__main__":
    finetuning_preprocess_data_tau2020()
