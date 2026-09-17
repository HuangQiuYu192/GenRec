import torch
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
        embeddings = torch.nn.functional.normalize(embeddings, dim=1); embeddings[0].zero_()
        return RepresentationArtifact(embeddings, torch.arange(dataset.num_items),
          {"kind":"collaborative", "dim":self.dim, "seed":self.seed, "fit_scope":"train_only", "split_hash":dataset.split_hash})

