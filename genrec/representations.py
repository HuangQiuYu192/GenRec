from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .artifacts import RepresentationArtifact


class HashedRepresentationBuilder:
    def __init__(self, dim=32, seed=42): self.dim, self.seed = dim, seed
    def build(self, dataset):
        generator = torch.Generator().manual_seed(self.seed)
        embeddings = torch.randn(dataset.num_items, self.dim, generator=generator)
        embeddings[0].zero_()
        return RepresentationArtifact(embeddings, torch.arange(dataset.num_items),
          {"kind":"hashed", "dim":self.dim, "seed":self.seed, "fit_scope":"none", "split_hash":dataset.split_hash})


class CollaborativeRepresentationBuilder:
    def __init__(self, dim=32, seed=42): self.dim, self.seed = dim, seed
    def build(self, dataset):
        matrix = torch.zeros(dataset.num_items, dataset.num_items)
        for sequence in dataset.train_sequences:
            for left, right in zip(sequence, sequence[1:]): matrix[left, right] += 1; matrix[right, left] += 1
        generator = torch.Generator().manual_seed(self.seed)
        projection = torch.randn(dataset.num_items, self.dim, generator=generator)
        embeddings = matrix @ projection
        embeddings = F.normalize(embeddings, dim=1); embeddings[0].zero_()
        return RepresentationArtifact(embeddings, torch.arange(dataset.num_items),
          {"kind":"collaborative", "dim":self.dim, "seed":self.seed, "fit_scope":"train_only", "split_hash":dataset.split_hash})


@dataclass(frozen=True)
class SentenceT5Config:
    """Content encoder options. Defaults follow the TIGER Sentence-T5 input protocol."""
    model_name: str = "sentence-transformers/sentence-t5-base"
    revision: str | None = None
    batch_size: int = 64
    device: str | None = None
    max_length: int = 128
    pooling: str = "mean"
    normalize: str = "none"
    standardize: str = "none"
    precision: str = "auto"
    local_files_only: bool = False
    log_every: int = 100

    def __post_init__(self):
        if self.batch_size < 1 or self.max_length < 1: raise ValueError("batch_size and max_length must be positive.")
        if self.pooling not in {"mean", "first"}: raise ValueError("pooling must be mean or first.")
        if self.normalize not in {"none", "l2"}: raise ValueError("normalize must be none or l2.")
        if self.standardize not in {"none", "per_dimension"}: raise ValueError("standardize must be none or per_dimension.")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}: raise ValueError("Unsupported precision.")


class SentenceT5RepresentationBuilder:
    """Frozen Sentence-T5 item encoder with paper-style masked mean pooling.

    Sentence-T5's encoder output is used directly: no SentenceTransformer pooling
    head or 2_Dense projection is applied. The default text template is built by
    ``load_item_texts(..., template='tiger')``.
    """
    def __init__(self, model_name="sentence-transformers/sentence-t5-base", batch_size=64, device=None, **kwargs):
        self.config = SentenceT5Config(model_name=model_name, batch_size=batch_size, device=device, **kwargs)

    @staticmethod
    def _autocast(device: torch.device, precision: str):
        if device.type != "cuda" or precision == "float32": return torch.autocast(device_type=device.type, enabled=False)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(precision, torch.float16)
        return torch.autocast(device_type="cuda", dtype=dtype)

    def build(self, dataset, item_texts):
        if len(item_texts) != dataset.num_items:
            raise ValueError(f"Expected {dataset.num_items} item texts, got {len(item_texts)}.")
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as error:
            raise ImportError("Sentence-T5 needs transformers and sentencepiece. Run: pip install -r requirements.txt") from error
        config = self.config
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        loading = {"revision": config.revision, "local_files_only": config.local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **loading)
        encoder = T5EncoderModel.from_pretrained(config.model_name, **loading).to(device).eval()
        hidden_dim = getattr(encoder.config, "d_model", None) or getattr(encoder.config, "hidden_size")
        encoded = [torch.zeros(hidden_dim, dtype=torch.float32)]
        total = dataset.num_items - 1
        with torch.inference_mode():
            for start in range(1, len(item_texts), config.batch_size):
                batch = item_texts[start:start + config.batch_size]
                inputs = tokenizer(batch, max_length=config.max_length, padding=True, truncation=True, return_tensors="pt").to(device)
                with self._autocast(device, config.precision):
                    hidden = encoder(**inputs).last_hidden_state
                    if config.pooling == "mean":
                        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    else:
                        vectors = hidden[:, 0]
                encoded.extend(vectors.float().cpu())
                completed = min(start + len(batch) - 1, total)
                if config.log_every and (completed == total or completed % (config.log_every * config.batch_size) < len(batch)):
                    print(f"[sentence_t5] encoded {completed}/{total} items", flush=True)
        embeddings = torch.stack(encoded)
        if config.standardize == "per_dimension":
            active = embeddings[1:]; mean = active.mean(0); std = active.std(0, unbiased=False).clamp_min(1e-6)
            embeddings[1:] = (active - mean) / std
        if config.normalize == "l2": embeddings[1:] = F.normalize(embeddings[1:], dim=1)
        embeddings[0].zero_()
        return RepresentationArtifact(embeddings, torch.arange(dataset.num_items), {
            "kind": "sentence_t5", "encoder": "T5EncoderModel", "pooling_head": "none",
            "dim": embeddings.shape[1], "fit_scope": "frozen_pretrained", "split_hash": dataset.split_hash,
            **asdict(config),
        })
