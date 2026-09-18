"""CLI: `python -m genrec.data.prepare --dataset Beauty --download`."""
import argparse
from .amazon import prepare_beauty
from .base import dataset_dir

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="Beauty", choices=["Beauty", "beauty", "amazon_beauty"]); p.add_argument("--download", action="store_true")
    a = p.parse_args(); bundle = prepare_beauty(dataset_dir(a.dataset), download=a.download)
    print(f"Saved data/{bundle.name}/interactions.txt, items.jsonl, manifest.json, and stats.json; users={bundle.num_users}, items={bundle.num_items}, split_hash={bundle.split_hash}")


if __name__ == "__main__": main()
