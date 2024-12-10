from best_rq_pytorch.vq import VQ
from best_rq_pytorch.trainer_vq import VQPretrainer
from best_rq_pytorch.data_vq import AudioDataset

# load the dataset

# dataset_folder = "..."
csv_path = "/home/chirag/datasets/audio_data.csv"

ds = AudioDataset(data=csv_path, max_length_in_seconds=32)

# set up the model

vq = VQ(
    heads=1,
    dim=1024,
    codebook_dim=32,
    codebook_size=16384,
    decay=0.8,
    commitment_weight=2.0,
    codebook_diversity_loss_weight=0.5,
)

print(
    f"trainable parameters: {sum(p.numel() for p in vq.parameters() if p.requires_grad)}"
)

# train the model

trainer = VQPretrainer(
    run=1,
    model=vq,
    dataset=ds,
    num_train_steps=150000,
    lr=3e-4,
    num_warmup_steps=10000,
    initial_lr=1e-6,
    batch_size=32,
    grad_accum_every=1,
)

trainer.train()
