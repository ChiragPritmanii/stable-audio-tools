from pathlib import Path
from functools import wraps

import pandas as pd

from einops import rearrange

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

from best_rq_pytorch.best_rq import BestRQ
from best_rq_pytorch.conformer import ConformerWrapper


def exists(val):
    return val is not None


accelerator = "cuda"
pretrained_checkpoint = (
    "/home/chirag//audio_tokenizer/best_rq/runs/8/results/bestrq.196000.pt"
)

pre_transform = BestRQ(
    codebook_size=1024,
    codebook_dim=16,
    sample_rate=24_000,
    n_mels=80,
    win_length=480 * 4,
    hop_length=480,
    conformer=ConformerWrapper(
        num_tokens=1024,
        conformer=dict(
            dim=1024,
            depth=24,
            heads=16,
            conv_kernel_size=5,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1,
            attn_flash=False,
        ),
    ),
).to(accelerator)

pkg = pre_transform.load(pretrained_checkpoint)
print("BEST-RQ is loaded on GPU!")
pre_transform.eval()
print("BEST-RQ is set to eval mode!")


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

        return wav


# data loader utilities


@torch.no_grad()
def apply_transform(waves, output_layer=14, pre_transform=pre_transform):
    """Applies pre-transform on the GPU."""
    activation = pre_transform(
        (waves).to(accelerator),
        return_layer_output=output_layer,
    )
    activation = activation.detach().cpu()
    return activation


def get_activations(data):
    # only keep the audios that were able to load
    waves = [item for item in data if item is not None]
    waves = torch.cat(waves, dim=0)
    activations = apply_transform(waves)
    activations = rearrange(activations, "b n d -> (b n) d")
    return activations


def get_dataloader(ds, **kwargs):
    collate_fn = get_activations
    return DataLoader(
        ds, collate_fn=collate_fn, num_workers=0, pin_memory=True, **kwargs
    )
