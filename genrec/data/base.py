"""Dataset bundle and portable processed-sequence format."""
from dataclasses import dataclass
from pathlib import Path
import json
from collections import Counter
from statistics import median
from typing import Optional
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
    # Parallel to ``train_sequences``.  Optional so datasets written by older
    # benchmark versions can still be loaded for non-personalized models.
    train_user_ids: Optional[list] = None


def dataset_dir(name: str) -> Path:
    """Canonical on-disk location; accepts legacy `amazon_beauty` as an alias."""
    aliases = {"amazon_beauty": "Beauty", "beauty": "Beauty"}
    return Path("data") / aliases.get(name.lower(), name)


def from_sequences(raw_sequences, name, metadata=None):
    user_mapping = {user: idx for idx, user in enumerate(sorted(raw_sequences))}
    item_values = sorted({item for sequence in raw_sequences.values() for item in sequence})
    item_mapping = {item: idx + 1 for idx, item in enumerate(item_values)}  # 0 is padding
    train, valid, test, split_material = [], [], [], []
    train_users = []
    # Preserve the file's interaction ordering: it is part of the existing
    # split fingerprint used by representation and index artifacts.
    for user, sequence in raw_sequences.items():
        values = [item_mapping[item] for item in sequence]
        if len(values) < 3: continue
        user_id = user_mapping[user]
        train.append(values[:-2]); train_users.append(user_id)
        valid.append((values[:-2], values[-2], user_id)); test.append((values[:-1], values[-1], user_id))
        split_material.append((user, values))
    if not train: raise ValueError("Processed sequences need at least one user with three interactions.")
    return DatasetBundle(train, valid, test, len(user_mapping), len(item_mapping) + 1, user_mapping, item_mapping,
                         stable_hash(split_material), name, metadata or {}, train_users)


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


def write_stats(root: Path, sequences: dict, bundle: DatasetBundle) -> None:
    """Dataset audit statistics derived from the portable interactions file."""
    lengths = [len(sequence) for sequence in sequences.values()]
    popularity = Counter(item for sequence in sequences.values() for item in sequence)
    counts = list(popularity.values()); threshold = median(counts)
    long_tail = {item for item, count in popularity.items() if count <= threshold}
    long_tail_interactions = sum(count for item, count in popularity.items() if item in long_tail)
    result = {
        "interaction_count": sum(lengths), "user_count": bundle.num_users, "item_count": len(popularity),
        "matrix_density": sum(lengths) / (bundle.num_users * max(len(popularity), 1)),
        "sequence_length": {"min": min(lengths), "median": median(lengths), "mean": sum(lengths) / len(lengths), "max": max(lengths)},
        "item_frequency": {"min": min(counts), "median": threshold, "mean": sum(counts) / len(counts), "max": max(counts)},
        "long_tail": {"definition": "item interaction count <= median item frequency", "threshold": threshold,
                      "item_count": len(long_tail), "item_fraction": len(long_tail) / len(popularity),
                      "interaction_fraction": long_tail_interactions / sum(lengths)},
        "split": {"policy": "leave-two-out after timestamp sorting", "validation": "second-to-last", "test": "last"},
        "internal_mapping": {"user_ids": "lexicographically sorted raw user IDs", "item_ids": "lexicographically sorted raw item IDs", "padding_id": 0},
    }
    (root / "stats.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
