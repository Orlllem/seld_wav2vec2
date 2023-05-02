

import glob
import logging
import os
import sys

from utils import save_wav_for_dataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def preprocess_audio_waves(ds_tag,
                           data_path,
                           save_path,
                           fs_target,
                           Ts,
                           stride):

    def_window = int(Ts*fs_target)
    stride_window = int(stride*fs_target)

    if ds_tag == "tut_2018":
        def_sample_rate = 44100
        logger = logging.getLogger("preprocessing-pretraining-tut2018")
        save_folder = f"{save_path}/TUT_2018/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/TUT_2018/ANSYN": "ANSYN",
                     f"{data_path}/TUT_2018/REAL": "REAL"}

        os.makedirs(f'{save_folder}/ANSYN')
        os.makedirs(f'{save_folder}/REAL')

    elif ds_tag == "tau_2019":
        def_sample_rate = 48000
        logger = logging.getLogger("preprocessing-pretraining-tau2019")
        save_folder = f"{save_path}/TAU_2019/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/TAU_2019/foa_dev": "foa_dev",
                     f"{data_path}/TAU_2019/proj/asignal/DCASE2019/dataset"
                     "/foa_eval": "foa_eval",
                     f"{data_path}/TAU_2019/mic_dev": "mic_dev",
                     f"{data_path}/TAU_2019/proj/asignal/DCASE2019/dataset"
                     "/mic_eval": "mic_eval"}

        os.makedirs(f'{save_folder}/foa_dev')
        os.makedirs(f'{save_folder}/foa_eval')
        os.makedirs(f'{save_folder}/mic_dev')
        os.makedirs(f'{save_folder}/mic_eval')

    elif ds_tag == "tau_2020":
        def_sample_rate = 24000
        logger = logging.getLogger("preprocessing-pretraining-tau2020")
        save_folder = f"{save_path}/TAU_2020/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/TAU_2020/foa_dev": "foa_dev",
                     f"{data_path}/TAU_2020/foa_eval": "foa_eval",
                     f"{data_path}/TAU_2020/mic_dev": "mic_dev",
                     f"{data_path}/TAU_2020/mic_eval": "mic_eval"}

        os.makedirs(f'{save_folder}/foa_dev')
        os.makedirs(f'{save_folder}/foa_eval')
        os.makedirs(f'{save_folder}/mic_dev')
        os.makedirs(f'{save_folder}/mic_eval')

    elif ds_tag == "tau_2021":
        def_sample_rate = 24000
        logger = logging.getLogger("preprocessing-pretraining-tau2021")
        save_folder = f"{save_path}/TAU_2021/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/TAU_2021/foa_dev": "foa_dev",
                     f"{data_path}/TAU_2021/foa_eval": "foa_eval",
                     f"{data_path}/TAU_2021/mic_dev": "mic_dev",
                     f"{data_path}/TAU_2021/mic_eval": "mic_eval"}

        os.makedirs(f'{save_folder}/foa_dev')
        os.makedirs(f'{save_folder}/foa_eval')
        os.makedirs(f'{save_folder}/mic_dev')
        os.makedirs(f'{save_folder}/mic_eval')

    elif ds_tag == "L3DAS21":
        def_sample_rate = 16000
        logger = logging.getLogger("preprocessing-pretraining-l3das21")
        save_folder = f"{save_path}/L3DAS21/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/L3DAS21/L3DAS_Task1_dev":
                     "L3DAS_Task1_dev",
                     f"{data_path}/L3DAS21/L3DAS_Task1_test":
                     "L3DAS_Task1_test",
                     f"{data_path}/L3DAS21/L3DAS_Task1_train100":
                     "L3DAS_Task1_train100",
                     f"{data_path}/L3DAS21/L3DAS_Task1_train360":
                     "L3DAS_Task1_train360",
                     f"{data_path}/L3DAS21/L3DAS_Task2_dev":
                     "L3DAS_Task2_dev",
                     f"{data_path}/L3DAS21/L3DAS_Task2_test":
                     "L3DAS_Task2_test",
                     f"{data_path}/L3DAS21/L3DAS_Task2_train":
                     "L3DAS_Task2_train"}

        os.makedirs(f'{save_folder}/L3DAS_Task1_dev')
        os.makedirs(f'{save_folder}/L3DAS_Task1_test')
        os.makedirs(f'{save_folder}/L3DAS_Task1_train100')
        os.makedirs(f'{save_folder}/L3DAS_Task1_train360')
        os.makedirs(f'{save_folder}/L3DAS_Task2_dev')
        os.makedirs(f'{save_folder}/L3DAS_Task2_test')
        os.makedirs(f'{save_folder}/L3DAS_Task2_train')

    elif ds_tag == "L3DAS22":
        def_sample_rate = 16000
        logger = logging.getLogger("preprocessing-pretraining-l3das22")
        save_folder = f"{save_path}/L3DAS22/chunk{def_window}_ts_{Ts}_str_{stride}"

        data_dict = {f"{data_path}/L3DAS22/L3DAS22_Task1_dev":
                     "L3DAS22_Task1_dev",
                     f"{data_path}/L3DAS22/L3DAS22_Task1_test":
                     "L3DAS22_Task1_test",
                     f"{data_path}/L3DAS22/L3DAS22_Task1_train100":
                     "L3DAS22_Task1_train100",
                     f"{data_path}/L3DAS22/L3DAS22_Task1_train360_1":
                     "L3DAS22_Task1_train360_1",
                     f"{data_path}/L3DAS22/L3DAS22_Task1_train360_2":
                     "L3DAS22_Task1_train360_2",
                     f"{data_path}/L3DAS22/L3DAS22_Task2_dev":
                     "L3DAS22_Task2_dev",
                     f"{data_path}/L3DAS22/L3DAS22_Task2_test":
                     "L3DAS22_Task2_test",
                     f"{data_path}/L3DAS22/L3DAS22_Task2_train":
                     "L3DAS22_Task2_train"}

        os.makedirs(f'{save_folder}/L3DAS22_Task1_dev')
        os.makedirs(f'{save_folder}/L3DAS22_Task1_test')
        os.makedirs(f'{save_folder}/L3DAS22_Task1_train100')
        os.makedirs(f'{save_folder}/L3DAS22_Task1_train360_1')
        os.makedirs(f'{save_folder}/L3DAS22_Task1_train360_2')
        os.makedirs(f'{save_folder}/L3DAS22_Task2_dev')
        os.makedirs(f'{save_folder}/L3DAS22_Task2_test')
        os.makedirs(f'{save_folder}/L3DAS22_Task2_train')

    logger.info(f"window: {def_window}")

    window_t = def_window/fs_target

    logger.info(f"window (s): {window_t}")

    for ds_path, ds_name in data_dict.items():

        wav_files = glob.glob(f'{ds_path}/**/*.wav',
                              recursive=True)

        assert len(wav_files) > 0, f'{ds_path}'

        logger.info(f"pre-processing {len(wav_files)} files")

        save_wav_for_dataset(ds_wav_files=wav_files,
                             save_path=save_folder,
                             ds_name=ds_name,
                             slide_win=def_window,
                             stride_win=stride_window,
                             sample_rate=fs_target,
                             default_sample_rate=def_sample_rate)

        saved_foa_wav_files = glob.glob(f'{save_folder}/**/*.wav',
                                        recursive=True)

        logger.info(f"saved {len(saved_foa_wav_files)} .wav files")
