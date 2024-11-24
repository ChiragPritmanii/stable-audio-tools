import torch

from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.abs_tokenizer import (
    AbsTokenizer,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.bestrq.best_rq_pytorch.vq import (
    VQ,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.bestrq.best_rq_pytorch.best_rq import (
    BestRQ,
)
from stable_audio_tools.configs.dataset_configs.custom_metadata.tokenizer.bestrq.best_rq_pytorch.conformer import (
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


vq_params = dict(
    heads=1,
    dim=1024,
    codebook_dim=32,
    codebook_size=16384,
    decay=0.8,
    commitment_weight=2.0,
    codebook_diversity_loss_weight=0.2,
)


class AudioTokenizer(AbsTokenizer):
    def __init__(
        self,
        best_rq_ckpt,
        vq_ckpt,
        best_rq_params=best_rq_params,
        vq_params=vq_params,
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

        self.vq = VQ(
            heads=vq_params["heads"],
            dim=vq_params["dim"],
            codebook_dim=vq_params["codebook_dim"],
            codebook_size=vq_params["codebook_size"],
            decay=vq_params["decay"],
            commitment_weight=vq_params["commitment_weight"],
            codebook_diversity_loss_weight=vq_params["codebook_diversity_loss_weight"],
        )

        vq_pkg = self.vq.load(vq_ckpt)
        print("VQ is loaded on GPU!")
        self.vq.eval()
        print("VQ is set to eval mode!")


    @torch.no_grad()
    def encode(self, waves):
        # waves -> b, t
        brq_embed = self.best_rq(
            (waves).to(self.device), return_layer_output=self.output_layer
        )
        brq_embed = brq_embed.detach().cpu()  # b, n, d
        quantized, codes, loss = self.vq(x=brq_embed, return_loss_breakdown=False)
        return codes  # b, n
