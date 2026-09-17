import argparse
from pathlib import Path
from genrec.data import load_dataset
from genrec.representations import HashedRepresentationBuilder, CollaborativeRepresentationBuilder

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--representation",choices=["hashed","collaborative"],default="hashed"); p.add_argument("--dim",type=int,default=32); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
d=load_dataset(Path("cache/datasets")/a.dataset); builder={"hashed":HashedRepresentationBuilder,"collaborative":CollaborativeRepresentationBuilder}[a.representation](a.dim,a.seed)
path=Path("cache/representations")/a.dataset/a.representation/"artifact.pt"; builder.build(d).save(path); print(f"Saved {path}")

