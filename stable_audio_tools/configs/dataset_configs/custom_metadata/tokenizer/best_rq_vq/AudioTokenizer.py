import torch
import numpy as np

from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.abs_tokenizer import (
    AbsTokenizer,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.best_rq_vq.best_rq_pytorch.vq import (
    VQ,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.best_rq_vq.best_rq_pytorch.best_rq import (
    BestRQ,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.best_rq_vq.best_rq_pytorch.conformer import (
    ConformerWrapper,
)

best_rq_params = dict(
    codebook_size=1024,
    codebook_dim=16,
    sample_rate=24_000,
    n_mels=80,
    win_length=480 * 4,
    hop_length=480,
    conformer_params=dict(
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
)


def quant_mem_efficient(
    representation, centroids, last_token_removed=False, feature_dim=1024
):
    assert representation.size(-1) % feature_dim == 0
    # Removing the first token and keeping the shape as [batch_size, seq_length - 1, 768] for clarity

    if last_token_removed:
        representation = representation[
            :, :-1, :
        ]  # Shape: [batch_size, seq_length - 1, 768]

    # Compute squared norms of each row in representation
    norm_rep = representation.pow(2).sum(
        dim=2, keepdim=True
    )  # Shape: [batch_size, seq_length - 1, 1]

    # Compute squared norms of centroids
    norm_cent = centroids.pow(2).sum(dim=1, keepdim=True)  # Shape: [2048, 1]

    # Compute dot products
    # Reshape representation for batch matrix multiplication: [batch_size * (seq_length - 1), 768]
    rep_flat = representation.reshape(-1, feature_dim)
    # Dot product, need to transpose centroids: [batch_size * (seq_length - 1), 2048]
    dot_product = torch.mm(rep_flat, centroids.t())
    dot_product = dot_product.reshape(
        representation.shape[0], representation.shape[1], -1
    )  # Reshape back

    # Compute L2 distance using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    distances = norm_rep + norm_cent.t() - 2 * dot_product  # Correct broadcasting

    # Find the index of the closest centroid for each vector
    _, tokens = torch.min(distances, dim=2)  # Shape: [batch_size, seq_length - 1]

    return tokens


class AudioTokenizer(AbsTokenizer):
    def __init__(
        self,
        best_rq_ckpt,
        vq_ckpt,
        best_rq_params=best_rq_params,
        device="cuda",
    ):
        super().__init__()

        self.sr = 24_000
        self.output_layer = 14
        self.device = device

        self.best_rq = BestRQ(
            codebook_size=best_rq_params["codebook_size"],
            codebook_dim=best_rq_params["codebook_dim"],
            sample_rate=best_rq_params["sample_rate"],
            n_mels=best_rq_params["n_mels"],
            win_length=best_rq_params["win_length"],
            hop_length=best_rq_params["hop_length"],
            conformer=ConformerWrapper(
                num_tokens=best_rq_params["conformer_params"]["num_tokens"],
                conformer=best_rq_params["conformer_params"]["conformer"],
            ),
        ).to(device)

        best_rq_pkg = self.best_rq.load(best_rq_ckpt)
        print("BEST-RQ is loaded on GPU!")
        self.best_rq.eval()
        print("BEST-RQ is set to eval mode!")

        self.vq = torch.tensor(np.load(vq_ckpt)).to(device)

    @torch.no_grad()
    def encode(self, waves):
        # waves -> b, t
        brq_embed = self.best_rq(
            (waves).to(self.device), return_layer_output=self.output_layer
        )
        brq_embed = brq_embed.detach().cpu()  # b, n, d

        codes = quant_mem_efficient(brq_embed, self.vq, True, 1024)  # b, n
        codes = codes.cpu()

        return codes  # b, n
