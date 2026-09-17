import argparse
from pathlib import Path
from genrec.data import download_amazon_metadata, load_amazon_item_texts, load_dataset
from genrec.representations import HashedRepresentationBuilder, CollaborativeRepresentationBuilder, SentenceT5RepresentationBuilder

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--representation",choices=["hashed","collaborative","sentence_t5"],default="hashed"); p.add_argument("--dim",type=int,default=32); p.add_argument("--seed",type=int,default=42); p.add_argument("--metadata",default=None); p.add_argument("--download-metadata",action="store_true"); p.add_argument("--model-name",default="sentence-transformers/sentence-t5-base"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--device",default=None); a=p.parse_args()
d=load_dataset(Path("cache/datasets")/a.dataset)
if a.representation == "sentence_t5":
    metadata = Path(a.metadata) if a.metadata else Path("data/raw/meta_Beauty.json.gz")
    if a.download_metadata: download_amazon_metadata("beauty", metadata)
    if not metadata.exists(): p.error(f"Missing {metadata}; pass --download-metadata or --metadata.")
    artifact = SentenceT5RepresentationBuilder(a.model_name,a.batch_size,a.device).build(d, load_amazon_item_texts(metadata,d.item_mapping))
else:
    artifact = {"hashed":HashedRepresentationBuilder,"collaborative":CollaborativeRepresentationBuilder}[a.representation](a.dim,a.seed).build(d)
path=Path("cache/representations")/a.dataset/a.representation/"artifact.pt"; artifact.save(path); print(f"Saved {path}")
