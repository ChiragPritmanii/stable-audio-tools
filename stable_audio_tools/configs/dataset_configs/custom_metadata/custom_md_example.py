import random

import torch
from einops import rearrange
from torchaudio.functional import resample

from .tokenizer.best_rq_vq.AudioTokenizer import AudioTokenizer

# checkpoints on current vm
best_rq_ckpt = "/home/chirag/models/tokenizer/bestrq.196000.pt"
vq_ckpt = "/home/chirag/models/tokenizer/centroids.npy"

audio_tokenizer = AudioTokenizer(best_rq_ckpt=best_rq_ckpt, vq_ckpt=vq_ckpt)


def get_custom_metadata(info, audio):

    seconds_total = info["seconds_total"]
    seconds_start = info["seconds_start"]
    seconds_end = seconds_start + 32
    ranges = [(i, i + 32) for i in range(0, seconds_total, 32)]
    start_range = [ran for ran in ranges if seconds_start in range(ran[0], ran[1])]
    end_range = [ran for ran in ranges if seconds_end in range(ran[0], ran[1])]
    token_range = start_range + end_range

    # audio -> c, t -> 2, sr*seconds_total
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)  # 1, t

    # resample data to the audio tokenizer's input sr
    audio = resample(
        audio, orig_freq=44100, new_freq=audio_tokenizer.sr
    )  # audio -> 1, t

    # we only get the codes of unpadded part
    waves = torch.cat([audio[:, ran[0] : ran[1]] for ran in token_range], dim=0)  # b, t
    # the output may be stored in a json and tensor isn't json serializable
    # so consider storing as a list

    start_code = (seconds_start - start_range[0][0]) * 50
    codes = audio_tokenizer.encode(waves)  # b, t_ (i.e. t/480)
    codes = rearrange(codes, "b t -> (b t)")  # 1, t_
    codes = codes[start_code : start_code + 1600]
    codes = codes.tolist()

    return {"tokens": codes}
