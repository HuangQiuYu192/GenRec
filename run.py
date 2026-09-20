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
p.add_argument("--protocol", choices=["paper", "standard"], default="standard", help="paper=fixed-step TIGER recipe; standard=epoch validation and early stopping.")
p.add_argument("--steps", type=int, default=None, help="Maximum optimizer updates (paper protocol default: 200000).")
p.add_argument("--epochs", type=int, default=None, help="Maximum epochs (standard protocol default: 400 for Beauty).")
p.add_argument("--batch-size", type=int, default=None, help="Protocol default: paper=256, standard=1024."); p.add_argument("--lr", type=float, default=.01); p.add_argument("--warmup-steps", type=int, default=10_000)
p.add_argument("--eval-batch-size", type=int, default=32, help="Smaller batch for beam-search validation/test.")
p.add_argument("--hidden-dim", type=int, default=128); p.add_argument("--heads", type=int, default=6); p.add_argument("--layers", type=int, default=4)
p.add_argument("--d-ff", type=int, default=1024); p.add_argument("--d-kv", type=int, default=64); p.add_argument("--dropout", type=float, default=.1)
p.add_argument("--max-history", type=int, default=None, help="Protocol default: paper=20, standard=50."); p.add_argument("--beam-width", type=int, default=20); p.add_argument("--device", default=None)
p.add_argument("--user-tokens", action=argparse.BooleanOptionalAction, default=True, help="Use TIGER's hashed user-ID token input.")
p.add_argument("--user-token-count", type=int, default=2000, help="Paper default number of hashed user tokens.")
p.add_argument("--eval-every-steps", type=int, default=None); p.add_argument("--eval-every-epochs", type=int, default=None)
p.add_argument("--early-stop-patience", type=int, default=None, help="Number of non-improving validation evaluations; disabled when omitted.")
p.add_argument("--early-stop-metric", choices=["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"], default="NDCG@10"); p.add_argument("--min-delta", type=float, default=0.0)
p.add_argument("--seed", type=int, default=42); p.add_argument("--debug", action="store_true")
a = p.parse_args(); set_seed(a.seed)
if a.steps is not None and a.epochs is not None: p.error("Choose either --steps or --epochs.")
if a.protocol == "paper" and a.epochs is not None: p.error("The paper protocol is fixed-step; use --steps, not --epochs.")
if a.protocol == "standard" and a.steps is not None: p.error("The standard protocol is epoch-based; use --epochs, not --steps.")
root = dataset_dir(a.dataset); dataset = make_synthetic_dataset(a.seed) if a.debug or a.dataset == "synthetic" else load_dataset(root)
index_path = root / "artifacts" / "item_indexes" / a.representation / a.item_index / "artifact.pt"
if index_path.exists() and not a.debug: index = ItemIndexArtifact.load(index_path, dataset.split_hash)
else:
    representation = HashedRepresentationBuilder(seed=a.seed).build(dataset)
    index = RQKMeansBuilder(codebook_size=16, seed=a.seed).build(dataset, representation)
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
path = (Path("outputs") if a.debug or a.dataset == "synthetic" else root / "outputs") / f"tiger_{dataset.name}_{a.protocol}_seed{a.seed}.json"
if a.protocol == "paper":
    steps = a.steps if a.steps is not None else (100 if (a.debug or a.dataset == "synthetic") else 200_000)
    epochs = None
    eval_every_steps = a.eval_every_steps if a.eval_every_steps is not None else 5_000
    eval_every_epochs = a.eval_every_epochs
    patience = a.early_stop_patience  # Disabled by default: this is the paper's fixed-step recipe.
    batch_size = a.batch_size if a.batch_size is not None else 256
    max_history = a.max_history if a.max_history is not None else 20
else:
    steps = a.steps
    epochs = a.epochs if a.epochs is not None else (2 if (a.debug or a.dataset == "synthetic") else 400)
    eval_every_steps = a.eval_every_steps
    eval_every_epochs = a.eval_every_epochs if a.eval_every_epochs is not None else 1
    patience = a.early_stop_patience if a.early_stop_patience is not None else 20
    batch_size = a.batch_size if a.batch_size is not None else 1024
    max_history = a.max_history if a.max_history is not None else 50
trainer = Trainer(epochs=epochs, steps=steps, batch_size=batch_size, lr=a.lr, optimizer="adagrad", warmup_steps=a.warmup_steps,
    device=device, max_history=max_history, log_path=path.with_suffix(".jsonl"), eval_batch_size=a.eval_batch_size, eval_every_steps=eval_every_steps, eval_every_epochs=eval_every_epochs,
    early_stop_patience=patience, early_stop_metric=a.early_stop_metric, min_delta=a.min_delta,
    checkpoint_path=path.with_name(path.stem + "_best.pt"))
model = TigerModel(dataset.num_items, index, a.hidden_dim, a.heads, a.layers, a.dropout, max_history, a.d_ff, a.d_kv, a.beam_width,
    user_tokens=a.user_tokens, user_token_count=a.user_token_count)
resource = trainer.fit(model, dataset, dataset.valid_examples); valid = trainer.evaluate(model, dataset.valid_examples); test = trainer.evaluate(model, dataset.test_examples)
output = {"dataset": dataset.name, "protocol": a.protocol, "device": device, "generator": {"architecture": "T5ForConditionalGeneration",
    "user_tokens": a.user_tokens, "user_token_count": a.user_token_count if a.user_tokens else 0, "max_history": max_history, "steps": steps, "epochs": epochs, "batch_size": batch_size, "optimizer": "adagrad",
    "lr": a.lr, "warmup_steps": a.warmup_steps, "eval_batch_size": a.eval_batch_size, "beam_width": a.beam_width, "eval_every_steps": eval_every_steps, "eval_every_epochs": eval_every_epochs,
    "early_stop_patience": patience, "early_stop_metric": a.early_stop_metric, "min_delta": a.min_delta}, "valid_metrics": valid, "test_metrics": test,
    "index": index.diagnostics(), "resource": resource}
write_json(path, output); torch.save({"model": model.state_dict(), "output": output}, path.with_suffix(".pt"))
print(output); print(f"Saved {path}")
