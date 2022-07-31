# Sound event detection and localization (SELD) with wav2vec2


## Install and development

```
$ conda create -n seld_w2v python=3.7 jupyter ipython
$ conda activate seld_w2v
(seld_w2v) $ cd seld_wav2vec2/
```

if you using GPU, I recommend to use CUDA 11.1 (or the next one)

```
(seld_w2v) $ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Then install the setup.py
```
(seld_w2v) $ pip install -e .
```

Also, for faster training install NVIDIA's apex library

```
(seld_w2v) $ git clone https://github.com/NVIDIA/apex
(seld_w2v) $ cd apex

(seld_w2v) $ export CUDA_HOME=/usr/local/cuda-11.1

(seld_w2v) $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```



