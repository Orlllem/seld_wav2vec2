import os
import shutil

import hydra
from omegaconf import DictConfig
from preprocess_20XX_stride import preprocess_audio_waves

ROOT_DIR = os.path.abspath(os.curdir)


@hydra.main(version_base=None, config_path=f"{ROOT_DIR}/conf", config_name="config")
def pretraining_preprocess_data(cfg: DictConfig) -> None:
    params = cfg["pret_dataset"]

    if os.path.isdir(params["save_path"]):
        shutil.rmtree(params["save_path"])
    os.makedirs(params["save_path"])

    preprocess_audio_waves(
        ds_tag="tut_2018",
        data_path=params["data_work_path"],
        save_path=params["save_path"],
        fs_target=params["fs_target"],
        Ts=params["window_in_s"],
        stride=params["stride_in_s"],
        exclude_mic=params.get("exclude_mic", False),
    )

    preprocess_audio_waves(
        ds_tag="tau_2019",
        data_path=params["data_work_path"],
        save_path=params["save_path"],
        fs_target=params["fs_target"],
        Ts=params["window_in_s"],
        stride=params["stride_in_s"],
        exclude_mic=params.get("exclude_mic", False),
    )

    preprocess_audio_waves(
        ds_tag="tau_2020",
        data_path=params["data_work_path"],
        save_path=params["save_path"],
        fs_target=params["fs_target"],
        Ts=params["window_in_s"],
        stride=params["stride_in_s"],
        exclude_mic=params.get("exclude_mic", False),
    )

    preprocess_audio_waves(
        ds_tag="tau_2021",
        data_path=params["data_work_path"],
        save_path=params["save_path"],
        fs_target=params["fs_target"],
        Ts=params["window_in_s"],
        stride=params["stride_in_s"],
        exclude_mic=params.get("exclude_mic", False),
    )

    if params["l3das"] == 21:
        preprocess_audio_waves(
            ds_tag="L3DAS21",
            data_path=params["data_work_path"],
            save_path=params["save_path"],
            fs_target=params["fs_target"],
            Ts=params["window_in_s"],
            stride=params["stride_in_s"],
        )
    else:
        preprocess_audio_waves(
            ds_tag="L3DAS22",
            data_path=params["data_work_path"],
            save_path=params["save_path"],
            fs_target=params["fs_target"],
            Ts=params["window_in_s"],
            stride=params["stride_in_s"],
        )


if __name__ == "__main__":
    pretraining_preprocess_data()
