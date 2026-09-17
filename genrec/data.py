from dataclasses import dataclass
from pathlib import Path
import csv
import torch

from .utils import stable_hash


@dataclass
class DatasetBundle:
    train_sequences: list
    valid_examples: list
    test_examples: list
    num_users: int
    num_items: int
    user_mapping: dict
    item_mapping: dict
    split_hash: str
    name: str = "synthetic"


def make_synthetic_dataset(seed=42, users=24, items=48, length=8) -> DatasetBundle:
    generator = torch.Generator().manual_seed(seed)
    raw = {str(user): torch.randint(0, items, (length,), generator=generator).tolist() for user in range(users)}
    return _from_sequences(raw, "synthetic")


def load_csv(path: str, name: str) -> DatasetBundle:
    interactions = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            missing = {"user_id", "item_id", "timestamp"} - set(row)
            if missing:
                raise ValueError(f"CSV misses required columns: {sorted(missing)}")
            interactions.setdefault(row["user_id"], []).append((float(row["timestamp"]), row["item_id"]))
    sequences = {u: [item for _, item in sorted(rows)] for u, rows in interactions.items() if len(rows) >= 3}
    if not sequences:
        raise ValueError("Need at least one user with three interactions.")
    return _from_sequences(sequences, name)


def _from_sequences(raw_sequences, name):
    user_mapping = {user: idx for idx, user in enumerate(sorted(raw_sequences))}
    item_values = sorted({item for sequence in raw_sequences.values() for item in sequence})
    item_mapping = {item: idx + 1 for idx, item in enumerate(item_values)}  # 0 is padding
    train, valid, test, split_material = [], [], [], []
    for user, sequence in raw_sequences.items():
        values = [item_mapping[item] for item in sequence]
        train.append(values[:-2])
        valid.append((values[:-2], values[-2]))
        test.append((values[:-1], values[-1]))
        split_material.append((user, values))
    return DatasetBundle(train, valid, test, len(user_mapping), len(item_mapping) + 1,
                         user_mapping, item_mapping, stable_hash(split_material), name)


def save_dataset(bundle: DatasetBundle, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset.pt"
    torch.save(bundle, path)
    return path


def load_dataset(root: Path) -> DatasetBundle:
    return torch.load(root / "dataset.pt", map_location="cpu")

