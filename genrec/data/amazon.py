"""Official Amazon 2014 5-core download and portable Beauty preprocessing."""
import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path
import urllib.request
from .base import from_sequences, save_dataset, write_manifest, write_sequences, write_stats

BEAUTY_5CORE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
BEAUTY_META_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try: urllib.request.urlretrieve(url, temporary); temporary.replace(destination)
    except Exception: temporary.unlink(missing_ok=True); raise
    return destination


def prepare_beauty(root=Path("data/Beauty"), download=True):
    root = Path(root); raw = root / "raw"
    reviews, metadata = raw / "reviews_Beauty_5.json.gz", raw / "meta_Beauty.json.gz"
    if download: _download(BEAUTY_5CORE_URL, reviews); _download(BEAUTY_META_URL, metadata)
    if not reviews.exists() or not metadata.exists(): raise FileNotFoundError("Beauty requires official 5-core reviews and metadata; use --download.")
    histories, active_items = defaultdict(list), set()
    with gzip.open(reviews, "rt", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            review = ast.literal_eval(line); user, item = str(review["reviewerID"]), str(review["asin"])
            histories[user].append((int(review["unixReviewTime"]), order, item)); active_items.add(item)
    sequences = {user: [item for _, _, item in sorted(rows)] for user, rows in histories.items() if len(rows) >= 3}
    write_sequences(root / "interactions.txt", sequences)
    with gzip.open(metadata, "rt", encoding="utf-8") as source, (root / "items.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            item = ast.literal_eval(line); asin = str(item.get("asin", ""))
            if asin in active_items:
                target.write(json.dumps({"item_id": asin, "title": item.get("title") or "", "categories": item.get("categories") or [],
                    "brand": item.get("brand") or ""}, ensure_ascii=False) + "\n")
    bundle = from_sequences(sequences, "Beauty", {"source": "UCSD Amazon 2014 official Beauty 5-core reviews", "fit_scope": "official_5core",
        "reviews": sum(map(len, sequences.values())), "item_auxiliary_file": "items.jsonl", "split_policy": "timestamp order; second-to-last validation; last test",
        "internal_mapping_policy": "lexicographically sorted raw IDs; padding item ID 0"})
    save_dataset(bundle, root); write_manifest(root, bundle); write_stats(root, sequences, bundle); return bundle


def load_item_texts(items_path: Path, item_mapping: dict) -> list:
    texts = [""] * (len(item_mapping) + 1)
    with Path(items_path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line); item_id = item["item_id"]
            if item_id not in item_mapping: continue
            categories = " ".join(" ".join(path) for path in item.get("categories", []))
            texts[item_mapping[item_id]] = (item.get("title", "") + " " + item.get("brand", "") + " " + categories).strip() or item_id
    return texts
