import argparse
from pathlib import Path
import torch
from genrec.utils import set_seed, write_json
from genrec.data import make_synthetic_dataset, load_dataset
from genrec.representations import HashedRepresentationBuilder
from genrec.item_indexes import RQKMeansBuilder
from genrec.artifacts import ItemIndexArtifact
from genrec.models import TigerModel
from genrec.training import Trainer

p=argparse.ArgumentParser(); p.add_argument("--dataset",default="synthetic"); p.add_argument("--representation",default="hashed"); p.add_argument("--item-index",default="rqkmeans"); p.add_argument("--epochs",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--debug",action="store_true"); a=p.parse_args(); set_seed(a.seed)
dataset=make_synthetic_dataset(a.seed) if a.debug or a.dataset=="synthetic" else load_dataset(Path("cache/datasets")/a.dataset)
index_path=Path("cache/item_indexes")/a.dataset/a.representation/a.item_index/"artifact.pt"
if index_path.exists() and not a.debug: index=ItemIndexArtifact.load(index_path,dataset.split_hash)
else: index=RQKMeansBuilder(seed=a.seed).build(dataset,HashedRepresentationBuilder(seed=a.seed).build(dataset))
device="cuda" if torch.cuda.is_available() else "cpu"; trainer=Trainer(epochs=a.epochs, device=device); model=TigerModel(dataset.num_items,index)
resource=trainer.fit(model,dataset); metrics=trainer.evaluate(model,dataset.test_examples); output={"dataset":dataset.name,"protocol":"controlled","device":device,"metrics":metrics,"index":index.diagnostics(),"resource":resource}
path=Path("outputs")/f"{dataset.name}_seed{a.seed}.json"; write_json(path,output); print(output); print(f"Saved {path}")

