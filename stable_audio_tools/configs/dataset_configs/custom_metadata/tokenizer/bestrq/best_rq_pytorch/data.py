from pathlib import Path
from functools import wraps

import pandas as pd

from beartype import beartype
from beartype.typing import Optional, Tuple
from beartype.door import is_bearable

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.functional import resample

# from torchaudio.transforms import Resample

from einops import rearrange


def exists(val):
    return val is not None


class AudioDataset(Dataset):
    def __init__(
        self,
        data: Optional[str] = None,
        folder: Optional[str] = None,
        max_length_in_seconds: Optional[
            int
        ] = 32,  # longer segment size works well with music data
        pad_to_max_length=True,
    ):
        super().__init__()
        if folder != None:
            path = Path(folder)
            assert path.exists(), "folder does not exist"
            files = list(path.glob("**/*.wav"))
            assert len(files) > 0, "no files found"
        elif data != None:
            self.files = []
            data = pd.read_csv(data)
            files = list(data["path"])
            assert len(files) > 0, "no files found"
        else:
            assert (
                folder != None or data != None
            ), "one of data/folder parameter needs to be provided"

        self.files = files
        self.target_sr = 24000
        self.max_length_in_seconds = max_length_in_seconds
        self.max_length = (
            (max_length_in_seconds * self.target_sr)
            if exists(max_length_in_seconds)
            else None
        )
        self.pad_to_max_length = pad_to_max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            # when setting normalize=True; divides each sample value in the tensor by 2**bit_depth//2
            # bit_depth determines the range of values each sample can take
            # e.g. int16 meaning 16-bit depth -> 2**16//2 = 32768 -> range: [-32768, 32767]
            wav, sr = torchaudio.load(file, normalize=True, backend="ffmpeg")
        except Exception as e:
            print(f"Error loading item {self.files[idx]}: {e}")
            return None

        # mean and resample operations on full audios are time expensive
        # directly select the n seconds clip, and then perform the mean and resample ops
        wav_len = wav.size(1)  # 44100*n
        seg_dur = self.max_length_in_seconds
        seg_size = int(sr * seg_dur)
        # get random segment from the audio
        if wav_len > seg_size:
            max_start = wav_len - seg_size
            start = torch.randint(0, max_start, (1,))
            wav = wav[:, start : start + seg_size]

        elif self.pad_to_max_length:
            wav = F.pad(wav, (0, seg_size - wav_len), "constant", value=0)

        # convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # 1, t

        # resample data to the target_sr
        wav = resample(
            wav, orig_freq=sr, new_freq=self.target_sr
        )  # this offers more speed
        # OR
        # transform = Resample(orig_freq=sr, new_freq=self.target_sr)
        # wav = transform(wav)

        wav = rearrange(wav, "1 n -> n")  # 1, t -> t

        return wav


# data loader utilities


def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    # only keep the audios that were able to load
    valid_items = [item for item in data if item is not None]
    return pad_sequence(valid_items, batch_first=True)


def get_dataloader(ds, pad_to_longest=True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(
        ds, collate_fn=collate_fn, num_workers=2, prefetch_factor=2, **kwargs
    )
