from pathlib import Path

import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize


class VQ(nn.Module):
    def __init__(
        self,
        heads=1,
        dim=1024,
        codebook_dim=32,
        codebook_size=16384,
        decay=0.8,
        commitment_weight=2.0,
        codebook_diversity_loss_weight=0.5,
    ):
        super().__init__()

        self.vq = VectorQuantize(
            heads=heads,  # num heads
            dim=dim,  # dimension of the input
            codebook_dim=codebook_dim,  # dimension of the codebook
            codebook_size=codebook_size,  # codebook size
            decay=decay,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=commitment_weight,  # the weight on the commitment loss
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,  #  the weight on the codebook diversity loss
        )
        # if we want to add a projection network then we can define under the self.vq.project_in/project_out

    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")
        try:
            self.load_state_dict(pkg["model"], strict=strict)
        except Exception:
            self.load_state_dict(pkg["encoder"], strict=strict)
        return pkg

    # if projection network is used, add use_proj_network=False param
    def forward(self, x, return_loss_breakdown=True):
        if return_loss_breakdown:
            quantized, indices, loss, loss_breakdown = self.vq(
                x, return_loss_breakdown=return_loss_breakdown
            )
            return (quantized, indices, loss, loss_breakdown)

        quantized, indices, loss = self.vq(
            x, return_loss_breakdown=return_loss_breakdown
        )

        return (quantized, indices, loss)
