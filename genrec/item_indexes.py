import torch
from .artifacts import ItemIndexArtifact


class RQKMeansBuilder:
    def __init__(self, code_length=3, codebook_size=16, iterations=12, seed=42):
        self.code_length, self.codebook_size, self.iterations, self.seed = code_length, codebook_size, iterations, seed
    def build(self, dataset, representation):
        values = representation.embeddings[1:].clone(); residual = values.clone(); codes = torch.zeros(dataset.num_items, self.code_length, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        for level in range(self.code_length):
            count = min(self.codebook_size, len(residual)); centers = residual[torch.randperm(len(residual), generator=generator)[:count]].clone()
            for _ in range(self.iterations):
                assignments = torch.cdist(residual, centers).argmin(1)
                for index in range(count):
                    mask = assignments == index
                    if mask.any(): centers[index] = residual[mask].mean(0)
            codes[1:, level] = assignments
            residual -= centers[assignments]
        return ItemIndexArtifact(codes, [min(self.codebook_size, len(values))] * self.code_length,
          {"kind":"rqkmeans", "seed":self.seed, "fit_scope":"train_only" if representation.metadata["fit_scope"] == "train_only" else "frozen", "split_hash":dataset.split_hash,
           "representation_hash": stable_repr(representation)})


class RQVAEBuilder(RQKMeansBuilder):
    """A compact vector-quantization baseline; uses residual codebook updates without a neural encoder."""
    def build(self, dataset, representation):
        artifact = super().build(dataset, representation); artifact.metadata["kind"] = "rqvae_lite"; return artifact


def stable_repr(representation): return representation.metadata.get("split_hash", "") + str(representation.dim)

