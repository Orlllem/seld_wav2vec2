import os
from distutils.core import setup

from setuptools import find_packages

pkg_dir = os.path.dirname(__name__)


requirements = [
    "autopep8",
    "pandas==1.5.3",
    "soundfile==0.11.0",
    "librosa==0.9.2",
    "scikit-learn==0.24.2",
    "matplotlib==3.5.3",
    # "torch==1.12.1",
    # "torchvision==0.13.1",
    # "torchaudio==0.12.1",
]

dvc_requirements = [
    "torch==1.12.1",
    "torchvision==0.13.1",
    "torchaudio==0.12.1",
    "dvclive==2.7.0",
    "dvc-gdrive==2.19.2"
]

dev_requirements = [
    "filelock==3.12.0",
    "wandb==0.15.3",
    "tensorboardX==2.6.0",
    "torch_audiomentations==0.11.0",
    "fairseq @ git+https://github.com/pytorch/fairseq.git@"
    "4fe8583396191c22011350248119db98ec1b5cb8"
]

setup(
    name='seld_wav2vec2',
    version='0.1.0',
    author='Orlem',
    long_description=open('README.md').read(),
    packages=find_packages('src', exclude=[
        'data*',
        'models*',
        'notebooks*',
        'scripts*',
    ]),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dvc": dvc_requirements,
        "dev": dev_requirements,
    },
)
