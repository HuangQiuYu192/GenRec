from .base import DatasetBundle, dataset_dir, from_sequences, load_dataset, read_sequences, save_dataset, write_sequences
from .amazon import prepare_beauty, load_item_texts


def make_synthetic_dataset(seed=42, users=24, items=48, length=8):
    import torch
    generator = torch.Generator().manual_seed(seed)
    raw = {str(user): torch.randint(0, items, (length,), generator=generator).tolist() for user in range(users)}
    return from_sequences(raw, "synthetic")
