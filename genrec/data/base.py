"""Dataset bundle and portable processed-sequence format."""
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from genrec.utils import stable_hash


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
    metadata: dict = None


def dataset_dir(name: str) -> Path:
    """Canonical on-disk location; accepts legacy `amazon_beauty` as an alias."""
    aliases = {"amazon_beauty": "Beauty", "beauty": "Beauty"}
    return Path("data") / aliases.get(name.lower(), name)


def from_sequences(raw_sequences, name, metadata=None):
    user_mapping = {user: idx for idx, user in enumerate(sorted(raw_sequences))}
    item_values = sorted({item for sequence in raw_sequences.values() for item in sequence})
    item_mapping = {item: idx + 1 for idx, item in enumerate(item_values)}  # 0 is padding
    train, valid, test, split_material = [], [], [], []
    for user, sequence in raw_sequences.items():
        values = [item_mapping[item] for item in sequence]
        if len(values) < 3: continue
        train.append(values[:-2]); valid.append((values[:-2], values[-2])); test.append((values[:-1], values[-1]))
        split_material.append((user, values))
    if not train: raise ValueError("Processed sequences need at least one user with three interactions.")
    return DatasetBundle(train, valid, test, len(user_mapping), len(item_mapping) + 1, user_mapping, item_mapping,
                         stable_hash(split_material), name, metadata or {})


def write_sequences(path: Path, sequences: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for user in sorted(sequences): handle.write(" ".join([user, *sequences[user]]) + "\n")


def read_sequences(path: Path) -> dict:
    sequences = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split(" ")
            if len(fields) < 4: raise ValueError(f"{path}:{line_number} needs a user ID and at least three items.")
            sequences[fields[0]] = fields[1:]
    return sequences


def save_dataset(bundle: DatasetBundle, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True); path = root / "dataset.pt"; torch.save(bundle, path); return path


def load_dataset(root: Path) -> DatasetBundle:
    return torch.load(root / "dataset.pt", map_location="cpu", weights_only=False)


def write_manifest(root: Path, bundle: DatasetBundle) -> None:
    (root / "manifest.json").write_text(json.dumps({"dataset": bundle.name, "users": bundle.num_users, "items": bundle.num_items,
        "split_hash": bundle.split_hash, **bundle.metadata}, indent=2, sort_keys=True), encoding="utf-8")
