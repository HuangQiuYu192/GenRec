import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genrec.artifacts import RepresentationArtifact
from genrec.data import dataset_dir, load_dataset
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder


p = argparse.ArgumentParser(description="Build a reproducible Semantic-ID index.")
p.add_argument("--dataset", required=True); p.add_argument("--representation", default="hashed")
p.add_argument("--item-index", choices=["rqkmeans", "rqvae"], default="rqkmeans"); p.add_argument("--artifact-name", default=None)
p.add_argument("--code-length", type=int, default=3); p.add_argument("--codebook-size", type=int, default=256); p.add_argument("--seed", type=int, default=42)
p.add_argument("--iterations", type=int, default=20, help="RQ-KMeans iterations.")
p.add_argument("--latent-dim", type=int, default=32); p.add_argument("--hidden-dims", default="512,256,128")
p.add_argument("--epochs", type=int, default=20_000); p.add_argument("--batch-size", type=int, default=1024); p.add_argument("--lr", type=float, default=0.4)
p.add_argument("--optimizer", choices=["adagrad", "adamw"], default="adagrad"); p.add_argument("--commitment-weight", type=float, default=.25)
p.add_argument("--no-kmeans-init", action="store_true"); p.add_argument("--kmeans-iterations", type=int, default=20)
p.add_argument("--quantization-warmup-steps", type=int, default=0); p.add_argument("--device", default=None); p.add_argument("--log-every", type=int, default=100)
a = p.parse_args()
root = dataset_dir(a.dataset); dataset = load_dataset(root)
representation = RepresentationArtifact.load(root / "artifacts" / "representations" / a.representation / "artifact.pt", dataset.split_hash)
if a.item_index == "rqkmeans":
    builder = RQKMeansBuilder(a.code_length, a.codebook_size, a.iterations, a.seed)
else:
    hidden_dims = tuple(int(value) for value in a.hidden_dims.split(",") if value.strip())
    builder = RQVAEBuilder(a.code_length, a.codebook_size, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, device=a.device,
        latent_dim=a.latent_dim, hidden_dims=hidden_dims, optimizer=a.optimizer, commitment_weight=a.commitment_weight,
        kmeans_init=not a.no_kmeans_init, kmeans_iterations=a.kmeans_iterations, quantization_warmup_steps=a.quantization_warmup_steps, log_every=a.log_every)
artifact = builder.build(dataset, representation)
name = a.artifact_name or a.item_index
path = root / "artifacts" / "item_indexes" / a.representation / name / "artifact.pt"; artifact.save(path)
print(f"Saved {path}: {artifact.diagnostics()}")
