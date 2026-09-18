"""Official Amazon 2014 5-core download and portable Beauty preprocessing."""
from __future__ import annotations
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
                description = item.get("description") or ""
                if isinstance(description, list): description = " ".join(map(str, description))
                target.write(json.dumps({"item_id": asin, "title": item.get("title") or "", "categories": item.get("categories") or [],
                    "brand": item.get("brand") or "", "price": item.get("price") or "", "description": description}, ensure_ascii=False) + "\n")
    bundle = from_sequences(sequences, "Beauty", {"source": "UCSD Amazon 2014 official Beauty 5-core reviews", "fit_scope": "official_5core",
        "reviews": sum(map(len, sequences.values())), "item_auxiliary_file": "items.jsonl", "split_policy": "timestamp order; second-to-last validation; last test",
        "internal_mapping_policy": "lexicographically sorted raw IDs; padding item ID 0"})
    save_dataset(bundle, root); write_manifest(root, bundle); write_stats(root, sequences, bundle); return bundle


def _category_text(categories) -> str:
    return " | ".join(" > ".join(map(str, path)) for path in categories if path)


def _item_text(item: dict, fields: tuple[str, ...], template: str, custom_template: str | None) -> str:
    values = {"title": str(item.get("title") or "").strip(), "brand": str(item.get("brand") or "").strip(),
              "categories": _category_text(item.get("categories") or []), "price": str(item.get("price") or "").strip(),
              "description": str(item.get("description") or "").strip()}
    selected = {key: values.get(key, "") for key in fields}
    if custom_template:
        return custom_template.format(**values).strip() or str(item["item_id"])
    if template == "tiger":
        return " ".join(f"{key.title()}: {selected[key]}." for key in ("title", "brand", "categories", "price") if key in selected and selected[key]) or str(item["item_id"])
    if template == "labeled":
        return " ".join(f"{key.title()}: {value}." for key, value in selected.items() if value) or str(item["item_id"])
    if template == "plain": return " ".join(value for value in selected.values() if value) or str(item["item_id"])
    raise ValueError(f"Unknown item text template: {template}")


def load_item_texts(items_path: Path, item_mapping: dict, fields=("title", "brand", "categories", "price"), template="tiger", custom_template=None) -> list:
    fields = tuple(fields)
    invalid = set(fields) - {"title", "brand", "categories", "price", "description"}
    if invalid: raise ValueError(f"Unknown item-text fields: {sorted(invalid)}")
    texts = [""] * (len(item_mapping) + 1)
    with Path(items_path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line); item_id = item["item_id"]
            if item_id not in item_mapping: continue
            texts[item_mapping[item_id]] = _item_text(item, fields, template, custom_template)
    return texts
