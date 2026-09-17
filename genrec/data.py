from dataclasses import dataclass
from pathlib import Path
import csv
import ast
import gzip
import urllib.request
from collections import Counter
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
    metadata: dict = None


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


AMAZON_CATEGORY_URLS = {
    # Official UCSD Amazon page links to both complete and precomputed k-core files.
    "beauty": {
        "full": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz",
        "5core": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    },
}


def download_amazon_category(category: str, destination: Path, source="5core") -> Path:
    """Download an official Amazon review file without silently overwriting it."""
    category = category.lower()
    if category not in AMAZON_CATEGORY_URLS:
        raise ValueError(f"Unsupported Amazon category {category!r}; available: {sorted(AMAZON_CATEGORY_URLS)}")
    if source not in AMAZON_CATEGORY_URLS[category]:
        raise ValueError(f"Unsupported source {source!r}; available: {sorted(AMAZON_CATEGORY_URLS[category])}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(AMAZON_CATEGORY_URLS[category][source], temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_amazon_reviews(path: str, name="amazon_beauty", min_user_interactions=5, min_item_interactions=5) -> DatasetBundle:
    """Read Amazon's Python-literal gzip reviews, apply iterative k-core, then split by time."""
    interactions = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                review = ast.literal_eval(line)
                interactions.append((str(review["reviewerID"]), str(review["asin"]), int(review["unixReviewTime"]), line_number))
            except (SyntaxError, ValueError, KeyError) as error:
                raise ValueError(f"Invalid Amazon review at line {line_number}: {error}") from error
    filtered, rounds = iterative_k_core(interactions, min_user_interactions, min_item_interactions)
    sequences = {}
    for user, item, timestamp, order in filtered:
        sequences.setdefault(user, []).append((timestamp, order, item))
    ordered = {user: [item for _, _, item in sorted(rows)] for user, rows in sequences.items()}
    bundle = _from_sequences(ordered, name)
    bundle.metadata = {"source": "UCSD Amazon product data", "raw_file": str(path), "k_core": {"user": min_user_interactions, "item": min_item_interactions},
                       "k_core_rounds": rounds, "raw_interactions": len(interactions), "filtered_interactions": len(filtered)}
    return bundle


def iterative_k_core(interactions, min_user_interactions=5, min_item_interactions=5):
    """Repeatedly drop low-frequency users/items until both degree constraints hold."""
    active = list(interactions); rounds = 0
    while True:
        user_counts = Counter(row[0] for row in active)
        item_counts = Counter(row[1] for row in active)
        reduced = [row for row in active if user_counts[row[0]] >= min_user_interactions and item_counts[row[1]] >= min_item_interactions]
        rounds += 1
        if len(reduced) == len(active):
            return reduced, rounds
        active = reduced


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
                         user_mapping, item_mapping, stable_hash(split_material), name, {})


def save_dataset(bundle: DatasetBundle, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset.pt"
    torch.save(bundle, path)
    return path


def load_dataset(root: Path) -> DatasetBundle:
    return torch.load(root / "dataset.pt", map_location="cpu")
