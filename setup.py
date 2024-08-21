import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup

extensions = [
    "src/cseld_ambisonics.pyx",
    "src/seld_wav2vec2/criterions/ccls_feature_class.pyx",
    "src/seld_wav2vec2/criterions/cSELD_evaluation_metrics.pyx",
]

setup(
    name="seld_wav2vec2",
    author="Orlem",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
