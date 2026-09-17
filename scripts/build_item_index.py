import argparse
from pathlib import Path
from genrec.data import load_dataset
from genrec.artifacts import RepresentationArtifact
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--representation",default="hashed"); p.add_argument("--item-index",choices=["rqkmeans","rqvae"],default="rqkmeans"); p.add_argument("--code-length",type=int,default=3); p.add_argument("--codebook-size",type=int,default=16); a=p.parse_args()
d=load_dataset(Path("cache/datasets")/a.dataset); r=RepresentationArtifact.load(Path("cache/representations")/a.dataset/a.representation/"artifact.pt",d.split_hash); builder={"rqkmeans":RQKMeansBuilder,"rqvae":RQVAEBuilder}[a.item_index](a.code_length,a.codebook_size)
artifact=builder.build(d,r); path=Path("cache/item_indexes")/a.dataset/a.representation/a.item_index/"artifact.pt"; artifact.save(path); print(f"Saved {path}: {artifact.diagnostics()}")

