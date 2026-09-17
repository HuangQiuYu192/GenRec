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


class RQVAEBuilder:
    """Train an MLP residual-quantized autoencoder, then freeze its Semantic IDs."""
    def __init__(self, code_length=3, codebook_size=256, hidden_dim=128, epochs=30, batch_size=512, lr=1e-3, seed=42, device=None):
        self.code_length, self.codebook_size, self.hidden_dim = code_length, codebook_size, hidden_dim
        self.epochs, self.batch_size, self.lr, self.seed, self.device = epochs, batch_size, lr, seed, device

    def build(self, dataset, representation):
        torch.manual_seed(self.seed); device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        values = representation.embeddings[1:].float().to(device); dim = values.shape[1]; size = min(self.codebook_size, len(values))
        encoder = torch.nn.Sequential(torch.nn.Linear(dim, self.hidden_dim), torch.nn.ReLU(), torch.nn.Linear(self.hidden_dim, dim)).to(device)
        decoder = torch.nn.Sequential(torch.nn.Linear(dim, self.hidden_dim), torch.nn.ReLU(), torch.nn.Linear(self.hidden_dim, dim)).to(device)
        codebooks = torch.nn.Parameter(torch.randn(self.code_length, size, dim, device=device) / dim ** .5)
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + [codebooks], lr=self.lr)
        for _ in range(self.epochs):
            order = torch.randperm(len(values), device=device)
            for indices in order.split(self.batch_size):
                source = values[indices]; latent = encoder(source); residual, quantized, commitments, codebook_losses = latent, torch.zeros_like(latent), [], []
                for level in range(self.code_length):
                    distances = torch.cdist(residual, codebooks[level]); assignment = distances.argmin(1); selected = codebooks[level][assignment]
                    quantized = quantized + selected; commitments.append(torch.nn.functional.mse_loss(residual, selected.detach()))
                    codebook_losses.append(torch.nn.functional.mse_loss(selected, residual.detach()))
                    residual = residual - selected
                straight_through = latent + (quantized - latent).detach()
                loss = torch.nn.functional.mse_loss(decoder(straight_through), source) + .25 * torch.stack(commitments).mean() + torch.stack(codebook_losses).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            residual = encoder(values); codes = torch.zeros(dataset.num_items, self.code_length, dtype=torch.long)
            for level in range(self.code_length):
                assignment = torch.cdist(residual, codebooks[level]).argmin(1); codes[1:, level] = assignment.cpu(); residual -= codebooks[level][assignment]
        # TIGER requires an item address; append a deterministic collision suffix when needed.
        groups = {}
        for item, code in enumerate(codes[1:].tolist(), 1): groups.setdefault(tuple(code), []).append(item)
        maximum = max(map(len, groups.values())); vocab_sizes = [size] * self.code_length
        if maximum > 1:
            suffix = torch.zeros(dataset.num_items, 1, dtype=torch.long)
            for group in groups.values():
                for index, item in enumerate(group): suffix[item] = index
            codes = torch.cat((codes, suffix), 1); vocab_sizes.append(maximum)
        return ItemIndexArtifact(codes, vocab_sizes,
          {"kind":"rqvae", "seed":self.seed, "epochs":self.epochs, "fit_scope":"train_only" if representation.metadata["fit_scope"] == "train_only" else "frozen",
           "split_hash":dataset.split_hash, "representation_hash":stable_repr(representation), "collision_suffix": maximum > 1})


def stable_repr(representation): return representation.metadata.get("split_hash", "") + str(representation.dim)
