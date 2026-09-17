import argparse
from pathlib import Path
from genrec.data import download_amazon_category, load_amazon_reviews, load_csv, save_dataset

p = argparse.ArgumentParser()
p.add_argument("--input", help="CSV with user_id,item_id,timestamp")
p.add_argument("--dataset", default="amazon_beauty")
p.add_argument("--amazon", choices=["beauty"], help="Prepare an official Amazon category")
p.add_argument("--download", action="store_true", help="Download the official raw file when using --amazon")
p.add_argument("--min-user-interactions", type=int, default=5)
p.add_argument("--min-item-interactions", type=int, default=5)
args = p.parse_args()
if args.amazon:
    raw = Path("data/raw") / f"reviews_{args.amazon.title()}.json.gz"
    if args.download: download_amazon_category(args.amazon, raw)
    if not raw.exists(): p.error(f"Missing {raw}; re-run with --download.")
    bundle = load_amazon_reviews(raw, args.dataset, args.min_user_interactions, args.min_item_interactions)
elif args.input:
    bundle = load_csv(args.input, args.dataset)
else:
    p.error("Provide --input CSV or --amazon beauty --download.")
path = save_dataset(bundle, Path("cache/datasets") / args.dataset)
print(f"Saved {path}; users={bundle.num_users}, items={bundle.num_items}, split_hash={bundle.split_hash}; metadata={bundle.metadata}")
