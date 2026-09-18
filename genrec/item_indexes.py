"""Item-to-Semantic-ID builders used by generative-retrieval baselines."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch
import torch.nn.functional as F

from .artifacts import ItemIndexArtifact
from .utils import set_seed


def representation_fingerprint(representation) -> str:
    """Fingerprint both the representation recipe and its actual vectors."""
    digest = hashlib.sha256()
    digest.update(repr(sorted(representation.metadata.items())).encode())
    digest.update(representation.embeddings.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _squared_distance(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance without materializing a [N, K, D] tensor."""
    return values.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * values @ centers.T


def _kmeans(values: torch.Tensor, count: int, iterations: int, generator: torch.Generator) -> torch.Tensor:
    if not 1 <= count <= len(values): raise ValueError("Invalid K-means codebook size.")
    centers = values[torch.randperm(len(values), generator=generator, device=values.device)[:count]].clone()
    for _ in range(iterations):
        assignments = _squared_distance(values, centers).argmin(1)
        counts = torch.bincount(assignments, minlength=count)
        sums = torch.zeros_like(centers).index_add_(0, assignments, values)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        # Keep a centre alive instead of silently retaining an arbitrary stale value.
        empty = (~nonempty).nonzero(as_tuple=False).flatten()
        if len(empty): centers[empty] = values[torch.randperm(len(values), generator=generator, device=values.device)[:len(empty)]]
    return centers


def _residual_codes(values: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = values
    levels = []
    for codebook in codebooks:
        assignments = _squared_distance(residual, codebook).argmin(1)
        levels.append(assignments)
        residual = residual - codebook[assignments]
    return torch.stack(levels, 1), residual


def _usage(codes: torch.Tensor, codebook_size: int) -> list[dict]:
    return [{"active": int(torch.unique(codes[:, level]).numel()), "total": codebook_size,
             "ratio": float(torch.unique(codes[:, level]).numel() / codebook_size)} for level in range(codes.shape[1])]


def _disambiguate(codes: torch.Tensor, codebook_size: int) -> tuple[torch.Tensor, list[int], bool, int]:
    """Append one deterministic token only when multiple items share a SID."""
    groups = {}
    for item, code in enumerate(codes[1:].tolist(), 1): groups.setdefault(tuple(code), []).append(item)
    maximum = max(map(len, groups.values()), default=1)
    vocab_sizes = [codebook_size] * codes.shape[1]
    if maximum == 1: return codes, vocab_sizes, False, maximum
    suffix = torch.zeros(codes.shape[0], 1, dtype=torch.long)
    for group in groups.values():
        for index, item in enumerate(group): suffix[item] = index
    return torch.cat((codes, suffix), 1), vocab_sizes + [maximum], True, maximum


def _artifact(dataset, representation, codes, codebook_size: int, metadata: dict) -> ItemIndexArtifact:
    codes, vocab_sizes, collision_suffix, maximum_collision = _disambiguate(codes, codebook_size)
    return ItemIndexArtifact(codes, vocab_sizes, {"split_hash": dataset.split_hash,
        "representation_fingerprint": representation_fingerprint(representation),
        "collision_suffix": collision_suffix, "maximum_collision": maximum_collision, **metadata})


class RQKMeansBuilder:
    """Non-neural residual K-means control index."""
    def __init__(self, code_length=3, codebook_size=16, iterations=20, seed=42):
        self.code_length, self.codebook_size, self.iterations, self.seed = code_length, codebook_size, iterations, seed

    def build(self, dataset, representation):
        values = representation.embeddings[1:].float(); size = min(self.codebook_size, len(values))
        generator = torch.Generator(device=values.device).manual_seed(self.seed)
        residual, codebooks = values.clone(), []
        for _ in range(self.code_length):
            centers = _kmeans(residual, size, self.iterations, generator)
            assignments = _squared_distance(residual, centers).argmin(1)
            codebooks.append(centers); residual = residual - centers[assignments]
        codes = torch.zeros(dataset.num_items, self.code_length, dtype=torch.long)
        codes[1:] = _residual_codes(values, torch.stack(codebooks))[0].cpu()
        return _artifact(dataset, representation, codes, size, {"kind": "rqkmeans", "seed": self.seed,
            "iterations": self.iterations, "fit_scope": "train_only" if representation.metadata["fit_scope"] == "train_only" else "frozen",
            "codebook_usage": _usage(codes[1:], size)})


@dataclass(frozen=True)
class RQVAEConfig:
    """Paper-faithful defaults; stability controls are explicit opt-in ablations."""
    code_length: int = 3
    codebook_size: int = 256
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    epochs: int = 20_000
    batch_size: int = 1024
    lr: float = 0.4
    optimizer: str = "adagrad"
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iterations: int = 20
    quantization_warmup_steps: int = 0
    seed: int = 42
    device: str | None = None
    log_every: int = 100

    def __post_init__(self):
        if self.code_length < 1 or self.codebook_size < 1 or self.latent_dim < 1: raise ValueError("Codebook dimensions must be positive.")
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0: raise ValueError("epochs, batch_size, and lr must be positive.")
        if self.optimizer not in {"adagrad", "adamw"}: raise ValueError("optimizer must be adagrad or adamw.")
        if self.commitment_weight < 0: raise ValueError("commitment_weight must be non-negative.")


def _mlp(dimensions: list[int]) -> torch.nn.Sequential:
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(dimensions) - 2: layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class RQVAEBuilder:
    """TIGER-style DNN + residual VQ tokenizer with auditable diagnostics."""
    def __init__(self, code_length=3, codebook_size=256, hidden_dim=None, epochs=20_000, batch_size=1024, lr=0.4,
                 seed=42, device=None, **kwargs):
        # ``hidden_dim`` remains accepted for backwards compatibility; use
        # ``hidden_dims`` for the paper topology or an explicit ablation.
        if hidden_dim is not None and "hidden_dims" not in kwargs: kwargs["hidden_dims"] = (hidden_dim,)
        self.config = RQVAEConfig(code_length=code_length, codebook_size=codebook_size, epochs=epochs,
            batch_size=batch_size, lr=lr, seed=seed, device=device, **kwargs)

    def _initialize_codebooks(self, latent, size, device):
        config = self.config; generator = torch.Generator(device=device).manual_seed(config.seed)
        if not config.kmeans_init:
            return torch.randn(config.code_length, size, config.latent_dim, device=device) / config.latent_dim ** .5
        residual, codebooks = latent.detach(), []
        for _ in range(config.code_length):
            codebook = _kmeans(residual, size, config.kmeans_iterations, generator)
            codebooks.append(codebook); residual = residual - codebook[_squared_distance(residual, codebook).argmin(1)]
        return torch.stack(codebooks)

    def build(self, dataset, representation):
        config = self.config; set_seed(config.seed)
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        values = representation.embeddings[1:].float().to(device); size = min(config.codebook_size, len(values))
        encoder = _mlp([values.shape[1], *config.hidden_dims, config.latent_dim]).to(device)
        decoder = _mlp([config.latent_dim, *reversed(config.hidden_dims), values.shape[1]]).to(device)
        with torch.no_grad(): initial_latent = encoder(values)
        codebooks = torch.nn.Parameter(self._initialize_codebooks(initial_latent, size, device))
        parameters = list(encoder.parameters()) + list(decoder.parameters()) + [codebooks]
        optimizer = torch.optim.Adagrad(parameters, lr=config.lr) if config.optimizer == "adagrad" else torch.optim.AdamW(parameters, lr=config.lr)
        generator = torch.Generator(device=device).manual_seed(config.seed + 1); step = 0; final = {}
        for epoch in range(1, config.epochs + 1):
            totals = {"loss": 0.0, "reconstruction": 0.0, "quantization": 0.0, "items": 0}
            for indices in torch.randperm(len(values), generator=generator, device=device).split(config.batch_size):
                source = values[indices]; latent = encoder(source); residual = latent; quantized = torch.zeros_like(latent); quantization = 0.0
                for codebook in codebooks:
                    assignment = _squared_distance(residual, codebook).argmin(1); selected = codebook[assignment]
                    quantized = quantized + selected
                    quantization = quantization + F.mse_loss(selected, residual.detach()) + config.commitment_weight * F.mse_loss(residual, selected.detach())
                    residual = residual - selected
                reconstruction = F.mse_loss(decoder(latent + (quantized - latent).detach()), source)
                warmup = min(1.0, step / max(config.quantization_warmup_steps, 1)) if config.quantization_warmup_steps else 1.0
                loss = reconstruction + warmup * quantization
                optimizer.zero_grad(); loss.backward(); optimizer.step(); step += 1
                amount = len(source); totals["loss"] += loss.item() * amount; totals["reconstruction"] += reconstruction.item() * amount
                totals["quantization"] += quantization.detach().item() * amount; totals["items"] += amount
            with torch.no_grad(): current_codes, residual = _residual_codes(encoder(values), codebooks)
            final = {key: value / totals["items"] for key, value in totals.items() if key != "items"}
            final.update({"residual_mse": float(residual.square().mean()), "codebook_usage": _usage(current_codes, size)})
            if config.log_every and (epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0):
                usage = ", ".join(f"L{i}:{row['active']}/{row['total']}" for i, row in enumerate(final["codebook_usage"]))
                print(f"[rqvae] epoch={epoch}/{config.epochs} loss={final['loss']:.6f} recon={final['reconstruction']:.6f} usage={usage}", flush=True)
        codes = torch.zeros(dataset.num_items, config.code_length, dtype=torch.long); codes[1:] = current_codes.cpu()
        return _artifact(dataset, representation, codes, size, {"kind": "rqvae", "fit_scope": "train_only" if representation.metadata["fit_scope"] == "train_only" else "frozen",
            **asdict(config), "final_diagnostics": final})
