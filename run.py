import argparse
from pathlib import Path

import torch

from genrec.artifacts import ItemIndexArtifact
from genrec.data import dataset_dir, make_synthetic_dataset, load_dataset
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder
from genrec.models import TigerModel
from genrec.representations import HashedRepresentationBuilder
from genrec.training import Trainer
from genrec.utils import set_seed, write_json


p = argparse.ArgumentParser(description="Train/evaluate a paper-configured TIGER generator.")
p.add_argument("--dataset", default="synthetic"); p.add_argument("--representation", default="hashed"); p.add_argument("--item-index", default="rqvae")
p.add_argument("--steps", type=int, default=None, help="Training updates; Beauty faithful default is 200000."); p.add_argument("--epochs", type=int, default=None, help="Testing-only alternative to --steps.")
p.add_argument("--batch-size", type=int, default=256); p.add_argument("--lr", type=float, default=.01); p.add_argument("--warmup-steps", type=int, default=10_000)
p.add_argument("--hidden-dim", type=int, default=128); p.add_argument("--heads", type=int, default=6); p.add_argument("--layers", type=int, default=4)
p.add_argument("--d-ff", type=int, default=1024); p.add_argument("--d-kv", type=int, default=64); p.add_argument("--dropout", type=float, default=.1)
p.add_argument("--max-history", type=int, default=20); p.add_argument("--beam-width", type=int, default=20); p.add_argument("--device", default=None)
p.add_argument("--seed", type=int, default=42); p.add_argument("--debug", action="store_true"); p.add_argument("--protocol", choices=["faithful", "controlled"], default="faithful")
a = p.parse_args(); set_seed(a.seed)
if a.steps is not None and a.epochs is not None: p.error("Choose either --steps or --epochs.")
root = dataset_dir(a.dataset); dataset = make_synthetic_dataset(a.seed) if a.debug or a.dataset == "synthetic" else load_dataset(root)
index_path = root / "artifacts" / "item_indexes" / a.representation / a.item_index / "artifact.pt"
if index_path.exists() and not a.debug: index = ItemIndexArtifact.load(index_path, dataset.split_hash)
else:
    representation = HashedRepresentationBuilder(seed=a.seed).build(dataset)
    index = RQKMeansBuilder(codebook_size=16, seed=a.seed).build(dataset, representation)
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
path = (Path("outputs") if a.debug or a.dataset == "synthetic" else root / "outputs") / f"tiger_{dataset.name}_seed{a.seed}.json"
steps = a.steps if a.steps is not None else (None if (a.debug or a.dataset == "synthetic") else 200_000)
epochs = a.epochs if a.epochs is not None else (2 if steps is None else None)
trainer = Trainer(epochs=epochs, steps=steps, batch_size=a.batch_size, lr=a.lr, optimizer="adagrad", warmup_steps=a.warmup_steps,
    device=device, max_history=a.max_history, log_path=path.with_suffix(".jsonl"))
model = TigerModel(dataset.num_items, index, a.hidden_dim, a.heads, a.layers, a.dropout, a.max_history, a.d_ff, a.d_kv, a.beam_width)
resource = trainer.fit(model, dataset); valid = trainer.evaluate(model, dataset.valid_examples); test = trainer.evaluate(model, dataset.test_examples)
output = {"dataset": dataset.name, "protocol": a.protocol, "device": device, "generator": {"architecture": "T5ForConditionalGeneration",
    "user_tokens": False, "max_history": a.max_history, "steps": steps, "epochs": epochs, "batch_size": a.batch_size, "optimizer": "adagrad",
    "lr": a.lr, "warmup_steps": a.warmup_steps, "beam_width": a.beam_width}, "valid_metrics": valid, "test_metrics": test,
    "index": index.diagnostics(), "resource": resource}
write_json(path, output); torch.save({"model": model.state_dict(), "output": output}, path.with_suffix(".pt"))
print(output); print(f"Saved {path}")
