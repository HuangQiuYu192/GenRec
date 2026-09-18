import argparse
from pathlib import Path
from genrec.data import dataset_dir, load_dataset
from genrec.artifacts import RepresentationArtifact
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--representation",default="hashed"); p.add_argument("--item-index",choices=["rqkmeans","rqvae"],default="rqkmeans"); p.add_argument("--code-length",type=int,default=3); p.add_argument("--codebook-size",type=int,default=256); p.add_argument("--epochs",type=int,default=30); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--device",default=None); a=p.parse_args()
root=dataset_dir(a.dataset); d=load_dataset(root); r=RepresentationArtifact.load(root/"artifacts"/"representations"/a.representation/"artifact.pt",d.split_hash)
builder = RQKMeansBuilder(a.code_length,a.codebook_size) if a.item_index == "rqkmeans" else RQVAEBuilder(a.code_length,a.codebook_size,a.hidden_dim,a.epochs,device=a.device)
artifact=builder.build(d,r); path=root/"artifacts"/"item_indexes"/a.representation/a.item_index/"artifact.pt"; artifact.save(path); print(f"Saved {path}: {artifact.diagnostics()}")
