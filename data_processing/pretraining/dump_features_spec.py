import glob
import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torchaudio
from mpire import WorkerPool
from spafe.fbanks import gammatone_fbanks
from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump-features-spectrogram-par")

eps = torch.finfo(torch.float32).eps


def _next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


def init(data_path, save_path, sample_rate, spec_extractor, worker_state):
    worker_state["data_path"] = data_path
    worker_state["save_path"] = save_path
    worker_state["sample_rate"] = sample_rate
    worker_state["spec_extractor"] = spec_extractor


class ExtractLogMelSpectrogram(object):
    def __init__(
        self,
        sample_rate,
        spectrogram_1d=False,
        n_mels=64,
        hop_len_s=0.005,  # 5ms
        gammatone_filter_banks=False,
    ):
        self.sample_rate = sample_rate
        self.hop_len = int(hop_len_s * self.sample_rate)
        self.win_len = 2 * self.hop_len
        self.n_fft = _next_greater_power_of_2(self.win_len)

        self.spec_transform = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            power=None,
            pad_mode="constant",
        )

        self.melscale_transform = MelScale(
            sample_rate=self.sample_rate,
            mel_scale="slaney",
            n_mels=n_mels,
            norm=None,
            n_stft=self.n_fft // 2 + 1,
        )

        self.amp2db_transform = AmplitudeToDB()
        self.spectrogram_1d = spectrogram_1d

        if gammatone_filter_banks:
            fb = gammatone_fbanks.gammatone_filter_banks(
                nfilts=n_mels, nfft=self.n_fft, fs=self.sample_rate, scale="descendant"
            )[0].T
            self.melscale_transform.fb = torch.from_numpy(fb).float()

        self.mel_wts = self.melscale_transform.fb

    def length_spectrogram(self, wav):
        return int(np.floor(wav / (self.win_len - self.hop_len)) + 1)

    def _get_foa_intensity_vectors(self, linear_spectra):
        IVx = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = torch.real(torch.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])

        normal = torch.sqrt(IVx**2 + IVy**2 + IVz**2) + eps
        IVx = torch.mm(IVx / normal, self.mel_wts)
        IVy = torch.mm(IVy / normal, self.mel_wts)
        IVz = torch.mm(IVz / normal, self.mel_wts)

        # we are doing the following instead of simply concatenating to keep
        # the processing similar to mel_spec and gcc
        foa_iv = torch.stack((IVx, IVy, IVz), dim=-1)
        if self.spectrogram_1d:
            foa_iv = foa_iv.reshape(foa_iv.shape[0], -1)  # (192, T)
        return foa_iv.permute(2, 1, 0)

    def extract_feat_spectrogram(self, source):
        # get spectrogram (4, 513, T)
        spec = self.spec_transform(source)

        # get intensity vector (3, 64, T) or (192, T)
        foa_iv = self._get_foa_intensity_vectors(spec.permute(2, 1, 0))

        # get mel-spectrogram (4, 64, T)
        melscale_spect = self.melscale_transform(spec.abs().pow(2))

        # get logmel-spectrogram (4, 64, T)
        logmel_spec = self.amp2db_transform(melscale_spect)

        if self.spectrogram_1d:
            # convert (4, 64, T) to (256, T)
            C, N, T = logmel_spec.shape
            logmel_spec = logmel_spec.reshape(C * N, T)

        # concatenate logmel-spectrogram with foa iv (7, 64, T) or (448, T)
        feat = torch.cat((logmel_spec, foa_iv), dim=0)

        if not self.spectrogram_1d:
            feat = feat.transpose(1, 2)  # (7, 64, T) -> (7, T, 64)

        return feat


def save_file(worker_state, filename):
    with open(filename, "rb") as file:
        wav, curr_sample_rate = torchaudio.load(file)  # wav -> (C,T)

    assert curr_sample_rate == worker_state["sample_rate"]

    # spec -> (7, T, 64)
    spec = worker_state["spec_extractor"].extract_feat_spectrogram(wav)
    save_name = f"{worker_state['save_path']}/{filename.replace(worker_state['data_path'], '').split('.')[0]}.npy"

    local_path = os.path.dirname(save_name)
    os.makedirs(local_path, exist_ok=True)

    np.save(file=save_name, arr=spec)


def dump_features_spectrogram(
    data_path,
    save_path,
    sample_rate=16000,
    spectrogram_1d=False,
    n_mels=64,
    hop_len_s=0.005,
    gammatone_filter_banks=False,
    n_jobs=1,
):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    spec_extractor = ExtractLogMelSpectrogram(
        sample_rate=sample_rate,
        spectrogram_1d=spectrogram_1d,
        n_mels=n_mels,
        hop_len_s=hop_len_s,
        gammatone_filter_banks=gammatone_filter_banks,
    )

    dataset = glob.glob(f"{data_path}/**/*.wav", recursive=True)

    logger.info(f"dataset size: {len(dataset)}")

    worker_init = partial(init, data_path, save_path, sample_rate, spec_extractor)

    if n_jobs == 1:
        for fn in dataset:
            worker_state = {}
            worker_state["data_path"] = data_path
            worker_state["save_path"] = save_path
            worker_state["sample_rate"] = sample_rate
            worker_state["spec_extractor"] = spec_extractor
            save_file(worker_state, fn)
    else:
        with WorkerPool(
            n_jobs=n_jobs, use_worker_state=True, start_method="spawn"
        ) as pool:
            _ = pool.map(save_file, dataset, worker_init=worker_init, progress_bar=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True, help="path to train file.")
    parser.add_argument(
        "--save_path", required=True, help="path to save the dump features."
    )
    parser.add_argument(
        "--sample_rate", default=16000, type=int, help="sample-rate used."
    )
    parser.add_argument(
        "--spectrogram_1d", default=False, type=bool, help="sample-rate used."
    )
    parser.add_argument("--n_mels", default=64, type=int, help="n_mels used.")
    parser.add_argument(
        "--hop_len_s", default=0.005, type=float, help="hop_len_s of fft."
    )
    parser.add_argument(
        "--gammatone_filter_banks",
        default=False,
        type=bool,
        help="whether to use gammatone_filter.",
    )
    parser.add_argument("--n_jobs", default=1, type=int, help="n_jobs used.")

    args = parser.parse_args()
    dump_features_spectrogram(
        args.data_path,
        args.save_path,
        args.sample_rate,
        spectrogram_1d=args.spectrogram_1d,
        n_mels=args.n_mels,
        hop_len_s=args.hop_len_s,
        gammatone_filter_banks=args.gammatone_filter_banks,
        n_jobs=args.n_jobs,
    )
