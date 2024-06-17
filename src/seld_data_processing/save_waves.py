import gc
import logging
import os
import sys

import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


logger = logging.getLogger("preprocessing-pretraining-utils")


def save_wav_for_dataset(
    ds_wav_files,
    save_path,
    ds_name,
    slide_win,
    stride_win,
    sample_rate,
    default_sample_rate,
):
    logger.info(
        f"resampling default_sample_rate={default_sample_rate}"
        f" to sample_rate={sample_rate}"
    )

    for wav_filename in tqdm(ds_wav_files):
        wav, curr_sample_rate = torchaudio.load(wav_filename)

        if curr_sample_rate != default_sample_rate:
            logger.info(
                "curr_sample_rate!=default_sample_rate: "
                f"{curr_sample_rate}!={default_sample_rate}, "
                f"resampling file {wav_filename} "
                f"curr_sample_rate {curr_sample_rate} "
                f"to sample_rate {sample_rate}"
            )

        if curr_sample_rate != sample_rate:
            resampler = T.Resample(curr_sample_rate, sample_rate)
            wav_proc = resampler(wav)
        else:
            wav_proc = wav.clone()

        if wav_proc.shape[0] > 1 and wav_proc.shape[0] < 4:
            wav_proc = wav_proc.mean(dim=0)
            logger.info(f"making file {wav_filename} monochannel -> shape {wav.shape}")

            assert len(wav_proc.shape) == 1

            wav_proc = wav_proc.unsqueeze(0)  # (1, L)

        assert len(wav_proc.shape) > 1

        filename = os.path.splitext(os.path.basename(wav_filename))[0]

        for k in range(0, wav_proc.shape[1], stride_win):
            wav_arr = wav_proc[:, k: k + slide_win]

            assert wav_arr.shape[0] == 4 or wav_arr.shape[0] == 1, f"{wav_arr.shape}"

            save_filename = f"{save_path}/{ds_name}/{filename}_index_{k}.wav"

            torchaudio.save(
                save_filename,
                wav_arr,
                sample_rate=sample_rate,
                bits_per_sample=16,
                encoding="PCM_S",
            )

        # save memory
        del wav_arr, wav_proc, wav
        gc.collect()
