import argparse
from pathlib import Path
from genrec.data import load_csv, save_dataset

p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--dataset", required=True); args = p.parse_args()
bundle = load_csv(args.input, args.dataset); path = save_dataset(bundle, Path("cache/datasets") / args.dataset)
print(f"Saved {path}; users={bundle.num_users}, items={bundle.num_items}, split_hash={bundle.split_hash}")

