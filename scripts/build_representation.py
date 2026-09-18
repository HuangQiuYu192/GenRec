import argparse
from pathlib import Path
from genrec.data import dataset_dir, load_dataset, load_item_texts
from genrec.representations import HashedRepresentationBuilder, CollaborativeRepresentationBuilder, SentenceT5RepresentationBuilder

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--representation",choices=["hashed","collaborative","sentence_t5"],default="hashed"); p.add_argument("--dim",type=int,default=32); p.add_argument("--seed",type=int,default=42); p.add_argument("--model-name",default="sentence-transformers/sentence-t5-base"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--device",default=None); a=p.parse_args()
root=dataset_dir(a.dataset); d=load_dataset(root)
if a.representation == "sentence_t5":
    items = root / "items.jsonl"
    if not items.exists(): p.error(f"Missing {items}; run: python -m genrec.data.prepare --dataset Beauty --download")
    artifact = SentenceT5RepresentationBuilder(a.model_name,a.batch_size,a.device).build(d, load_item_texts(items,d.item_mapping))
else:
    artifact = {"hashed":HashedRepresentationBuilder,"collaborative":CollaborativeRepresentationBuilder}[a.representation](a.dim,a.seed).build(d)
path=root/"artifacts"/"representations"/a.representation/"artifact.pt"; artifact.save(path); print(f"Saved {path}")
