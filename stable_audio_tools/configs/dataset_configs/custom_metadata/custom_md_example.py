import random

import torch
from torchaudio.functional import resample

from tokenizer.bestrq.AudioTokenizer import AudioTokenizer

best_rq_ckpt = "..."
vq_ckpt = "..."

audio_tokenizer = AudioTokenizer(best_rq_ckpt=best_rq_ckpt, vq_ckpt=vq_ckpt)


def get_custom_metadata(info, audio):
    # audio -> c, t -> 2, sr*seconds_total
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)  # 1, t

    # resample data to the audio tokenizer's input sr
    audio = resample(
        audio, orig_freq=44100, new_freq=audio_tokenizer.sr
    )  # audio -> 1, t

    # the output may be stored in a json and tensor isn't json serializable
    # so consider storing as a list

    codes = audio_tokenizer.encode(audio)  # 1, t_
    codes = codes[0].tolist()  # t_

    return {"sem_tokens": codes}
