# Sound event detection and localization (SELD) with wav2vec2


# Pre-trained models

Model Name | Dataset | Model
|---|---|---
w2v-SELD BASE | L3DAS21-SELD | [download](https://drive.google.com/file/d/1UxB7gUxQQY1r4DLKMz1dEY12BUUE37A_/view?usp=drive_link)
w2v-SELD LARGE | L3DAS22-SELD | [download](https://drive.google.com/file/d/178olxS5N6efvmXdM8F6YioWVUYK-UmW-/view?usp=drive_link)



## Data Processing (DVC)


The data processing steps are implemented in Data Version Control (DVC) to easy the reproduction process. The steps are presented in ```dvc.yaml``` . 
The .zip data files are expected to be in ```data/pre-training/raw_3d_audio```. Thus, download all zip files and put inside the folder

```
cd data/pre-training/raw_3d_audio

tree
.
├── L3DAS21
│   ├── L3DAS_Task1_dev.zip
│   ├── L3DAS_Task1_test.zip
│   ├── L3DAS_Task1_train100.zip
│   ├── L3DAS_Task1_train360.zip
│   ├── L3DAS_Task2_dev.zip
│   ├── L3DAS_Task2_test.zip
│   └── L3DAS_Task2_train.zip
├── L3DAS22
│   └── l3das22.zip
├── TAU_2019
│   ├── foa_dev_full.zip
│   ├── foa_eval.zip
│   ├── metadata_dev.zip
│   ├── metadata_eval.zip
│   ├── mic_dev_full.zip
│   └── mic_eval.zip
├── TAU_2020
│   ├── foa_dev_full.zip
│   ├── foa_eval.zip
│   ├── metadata_dev.zip
│   ├── metadata_eval.zip
│   ├── mic_dev_full.zip
│   └── mic_eval.zip
├── TAU_2021
│   ├── foa_dev_full.zip
│   ├── foa_eval.zip
│   ├── metadata_dev.zip
│   ├── metadata_eval.zip
│   ├── mic_dev_full.zip
│   └── mic_eval.zip
└── TUT_2018
    ├── ANSYN
    │   ├── ov1_split1.zip
    │   ├── ov1_split2.zip
    │   ├── ov1_split3.zip
    │   ├── ov2_split1.zip
    │   ├── ov2_split2.zip
    │   ├── ov2_split3.zip
    │   ├── ov3_split1.zip
    │   ├── ov3_split2.zip
    │   └── ov3_split3.zip
    └── REAL
        ├── ov1_split1.zip
        ├── ov1_split8.zip
        ├── ov1_split9.zip
        ├── ov2_split1.zip
        ├── ov2_split8.zip
        ├── ov2_split9.zip
        ├── ov3_split1.zip
        ├── ov3_split8.zip
        └── ov3_split9.zip
```

### Setup
```
$ conda create -n seld_w2v_dvc python=3.8 -y
$ conda activate seld_w2v_dvc
(seld_w2v_dvc) $ cd seld_wav2vec2/
```

Install data processing requirements

```
(seld_w2v_dvc) $ pip install -r dvc-requirements.txt
(seld_w2v_dvc) $ pip install -e . --no-deps
```

### steps

The DVC stages are 

1. ```unzip_pret_data``` $\rightarrow$ Outputs: data/pre-training/raw_3d_audio_data

2. ```prepare_pret_dataset``` $\rightarrow$ Outputs: data/pre-training/pret_l3das22_raw_ts4_str4

3. ```ft_dataset_tau2019```  $\rightarrow$ Outputs: data/fine-tuning/manifest/finetuning

4. ```ft_dataset_tau2020``` $\rightarrow$ Outputs: data/fine-tuning/manifest/finetuning


To execute all stages 

```
(seld_w2v) $ dvc repro 
```

and to execute a single step ```stage```

```
(seld_w2v) $ dvc repro -s stage
```


## Install and development (DEV)

```
$ conda create -n seld_w2v python=3.8 cudatoolkit=11.3 cuDNN=8.2 -y
$ conda activate seld_w2v
(seld_w2v) $ cd seld_wav2vec2/
```

Install libsndfile library

```
(seld_w2v) $ conda install -c conda-forge libsndfile -y
```

if you using GPU, I recommend to use CUDA 11.3 (or the next one)

```
(seld_w2v) $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 cuDNN=8.2 -c pytorch -y
```

Then install the setup.py

```
(seld_w2v) $ pip install -r dev-requirements.txt
(seld_w2v) $ pip install -e . --no-deps
```

Also, for faster training install NVIDIA's apex library

```
(seld_w2v) $ git clone https://github.com/NVIDIA/apex
(seld_w2v) $ cd apex

(seld_w2v) $ export CUDA_HOME=/usr/local/cuda-11.3

(seld_w2v) $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Pre-training

To pre-train the wav2vec2-4ch it is necessary to segment the 3d audio files using the data processing steps above implemented in DVC. 


To convert the wav2vec 2.0 to a 4 channel wav2vec2-4ch model use the [script](scripts/wav2vec_change_feature_extraction.py) that convert the original pre-trained models of wav2vec 2.0 to have 4 channels. 


### Train wav2vec2-4ch BASE model

```
(seld_w2v) $ fairseq-hydra-train --config-dir conf/pretraining --config-name w2v_base_pret_audio_seld_l3das21_400K_ts4_str4_v1 task.data=data/pre-training/manifest/pretraining/l3das21_ts4_str4 model.pre_w2v_path=/path/to/models/w2v_audio_base_4ch_unorm.pt common.user_dir=/path/to/src/seld_wav2vec2
```


### Train wav2vec2-4ch LARGE model

```
(seld_w2v) $ fairseq-hydra-train --config-dir conf/pretraining --config-name w2v_large_pret_audio_seld_l3das22_600K_ts4_str4_v1 task.data=data/pre-training/manifest/pretraining/l3das21_ts4_str4 model.pre_w2v_path=/path/to/models/w2v_audio_large_4ch_norm.pt common.user_dir=/path/to/seld_wav2vec2/src/seld_wav2vec2
```


## Fine-tuning 

Fine-tuning the pre-trained wav2vec2-4ch models on labeled datasets 

```
fairseq-hydra-train --config-dir /path/to/conf/finetuning --config-name exp.yaml task.data=/path/to/manifest model.w2v_path=/path/to/model.pt common.user_dir=/path/to/seld_wav2vec2/src/seld_wav2vec2

```

The ```config-name``` represent the .yaml experiments that are in ```conf/finetuning```.


# License

seld_wav2vec2(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{olima2023w2v2seld,
  title = {w2v-SELD: A Sound Events Localization and Detection (SELD) Framework for Self-Supervised Spatial Audio Pre-Training},
  author = {Orlem Lima dos Santos, Karen Rosero and Roberto de Alencar Lotufo},
  booktitle = {arxiv},
  year = {2023},
}
```