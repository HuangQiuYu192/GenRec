import argparse
import sys
from pathlib import Path

# This file is normally executed as `python scripts/build_representation.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genrec.data import dataset_dir, load_dataset, load_item_texts
from genrec.representations import HashedRepresentationBuilder, CollaborativeRepresentationBuilder, SentenceT5RepresentationBuilder


p = argparse.ArgumentParser(description="Build versioned item representations.")
p.add_argument("--dataset", required=True); p.add_argument("--representation", choices=["hashed", "collaborative", "sentence_t5"], default="hashed")
p.add_argument("--artifact-name", default=None, help="Subdirectory under artifacts/representations; defaults to --representation.")
p.add_argument("--dim", type=int, default=32); p.add_argument("--seed", type=int, default=42)
p.add_argument("--model-name", default="sentence-transformers/sentence-t5-base"); p.add_argument("--revision", default=None)
p.add_argument("--batch-size", type=int, default=64); p.add_argument("--device", default=None); p.add_argument("--max-length", type=int, default=128)
p.add_argument("--pooling", choices=["mean", "first"], default="mean"); p.add_argument("--normalize", choices=["none", "l2"], default="none")
p.add_argument("--standardize", choices=["none", "per_dimension"], default="none"); p.add_argument("--precision", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
p.add_argument("--local-files-only", action="store_true"); p.add_argument("--log-every", type=int, default=100)
p.add_argument("--text-fields", default="title,brand,categories,price", help="Comma-separated fields: title,brand,categories,price,description")
p.add_argument("--text-template", choices=["tiger", "labeled", "plain"], default="tiger")
p.add_argument("--custom-template", default=None, help="Python format template, e.g. 'Title: {title}. Description: {description}.'")
a = p.parse_args()
root = dataset_dir(a.dataset); d = load_dataset(root)
if a.representation == "sentence_t5":
    items = root / "items.jsonl"
    if not items.exists(): p.error(f"Missing {items}; run: python data/prepare.py --dataset Beauty --download")
    fields = tuple(part.strip() for part in a.text_fields.split(",") if part.strip())
    texts = load_item_texts(items, d.item_mapping, fields=fields, template=a.text_template, custom_template=a.custom_template)
    artifact = SentenceT5RepresentationBuilder(a.model_name, a.batch_size, a.device, revision=a.revision, max_length=a.max_length,
        pooling=a.pooling, normalize=a.normalize, standardize=a.standardize, precision=a.precision, local_files_only=a.local_files_only, log_every=a.log_every).build(d, texts)
    artifact.metadata.update({"text_fields": list(fields), "text_template": a.text_template, "custom_template": a.custom_template})
else:
    artifact = {"hashed": HashedRepresentationBuilder, "collaborative": CollaborativeRepresentationBuilder}[a.representation](a.dim, a.seed).build(d)
name = a.artifact_name or a.representation
path = root / "artifacts" / "representations" / name / "artifact.pt"; artifact.save(path)
print(f"Saved {path}")
