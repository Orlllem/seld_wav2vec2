# Inspired by wav2vec_manifest.py
# https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py
"""
Data pre-processing: build train, valid, and test data.
"""
import glob
import logging
import os

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("preprocessing-ft-manifest")


def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def gen_tsv_manifest(save_folder, manifest_folder, dset, ext):
    dir_path = f"{save_folder}/{dset}/"
    dest_path = f"{manifest_folder}"

    os.makedirs(dest_path, exist_ok=True)

    search_path = os.path.join(dir_path, "**/*." + f"{ext}")

    with open(os.path.join(dest_path, f"{dset}.tsv"), "w") as set_f:
        print(dir_path, file=set_f)

        files = glob.glob(search_path, recursive=True)

        files.sort()
        files.sort(key=len)

        for fname in files:
            file_path = os.path.realpath(fname)

            frames = sf.info(fname).frames
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=set_f
            )


def get_feat_extract_output_lengths(conv_feature_layers, input_lengths):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        return torch.floor((input_length - kernel_size) / stride + 1)

    conv_cfg_list = eval(conv_feature_layers)

    for i in range(len(conv_cfg_list)):
        input_lengths = _conv_out_length(
            input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
        )

    return input_lengths.to(torch.long)


def cart2sph_array(array):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    assert array.shape[-1] == 3

    x = array[:, :, :, 0]
    y = array[:, :, :, 1]
    z = array[:, :, :, 2]

    B = array.shape[0]
    T = array.shape[1]

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    r_sel = r[r != 0]

    assert np.allclose(
        r_sel, [1.0]*len(r_sel), atol=1e-05), r_sel

    return np.stack((elevation, azimuth), axis=-1).reshape(B, T, -1)


def remove_overlap_same_class(df):

    index_to_remove = []

    if any(df.duplicated(subset=['frame_number'])):

        df_frames = df[df.duplicated(subset=['frame_number'], keep=False)]

        for i in range(len(df_frames)):
            frame = df_frames.iloc[i]["frame_number"]

            df_frames_sub = df_frames[df_frames["frame_number"] == int(frame)]

            if len(df_frames_sub["sound_event_recording"].unique()) == 1:

                if len(df_frames_sub) == 2:
                    assert len(df_frames_sub) == 2
                    index_to_remove.append(df_frames_sub.index.tolist()[-1])
                else:
                    assert len(df_frames_sub) == 3
                    index_to_remove.append(df_frames_sub.index.tolist()[-1])
                    index_to_remove.append(df_frames_sub.index.tolist()[-2])

    index_to_remove = list(set(index_to_remove))

    if len(index_to_remove) > 0:

        df = df.drop(index=index_to_remove)

        return df
    else:
        return df


def linear_interp(xp, x, y):
    x1, x2 = x
    y1, y2 = y
    yp = ((y2-y1)/(x2 - x1))*(xp - x1) + y1
    return yp
