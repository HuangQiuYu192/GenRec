import argparse
from pathlib import Path
import torch
from genrec.utils import set_seed, write_json
from genrec.data import dataset_dir, make_synthetic_dataset, load_dataset
from genrec.representations import HashedRepresentationBuilder
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder
from genrec.artifacts import ItemIndexArtifact
from genrec.models import TigerModel
from genrec.training import Trainer

p = argparse.ArgumentParser(description="Train/evaluate TIGER on a prepared GenRec dataset.")
p.add_argument("--dataset", default="synthetic"); p.add_argument("--representation", default="hashed"); p.add_argument("--item-index", choices=["rqkmeans", "rqvae"], default="rqvae")
p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden-dim", type=int, default=128); p.add_argument("--heads", type=int, default=4); p.add_argument("--layers", type=int, default=2); p.add_argument("--max-history", type=int, default=50)
p.add_argument("--seed", type=int, default=42); p.add_argument("--debug", action="store_true"); p.add_argument("--protocol", choices=["faithful", "controlled"], default="controlled"); a = p.parse_args(); set_seed(a.seed)
root = dataset_dir(a.dataset)
dataset = make_synthetic_dataset(a.seed) if a.debug or a.dataset == "synthetic" else load_dataset(root)
index_path = root / "artifacts" / "item_indexes" / a.representation / a.item_index / "artifact.pt"
if index_path.exists() and not a.debug: index = ItemIndexArtifact.load(index_path, dataset.split_hash)
else:
    representation = HashedRepresentationBuilder(seed=a.seed).build(dataset)
    index = (RQVAEBuilder(epochs=3, codebook_size=16, hidden_dim=32, seed=a.seed) if a.item_index == "rqvae" else RQKMeansBuilder(codebook_size=16, seed=a.seed)).build(dataset, representation)
device = "cuda" if torch.cuda.is_available() else "cpu"
path = (Path("outputs") if a.debug or a.dataset == "synthetic" else root / "outputs") / f"tiger_{dataset.name}_seed{a.seed}.json"
trainer = Trainer(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=device, max_history=a.max_history, log_path=path.with_suffix(".jsonl"))
model = TigerModel(dataset.num_items, index, a.hidden_dim, a.heads, a.layers, max_history=a.max_history)
resource = trainer.fit(model, dataset); valid = trainer.evaluate(model, dataset.valid_examples); test = trainer.evaluate(model, dataset.test_examples)
output = {"dataset":dataset.name,"protocol":a.protocol,"device":device,"valid_metrics":valid,"test_metrics":test,"index":index.diagnostics(),"resource":resource}
write_json(path, output); torch.save({"model":model.state_dict(),"output":output}, path.with_suffix(".pt"))
print(output); print(f"Saved {path}")
