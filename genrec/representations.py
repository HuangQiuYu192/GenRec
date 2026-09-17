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


class SentenceT5RepresentationBuilder:
    """Frozen Sentence-T5 mean-pooled content embedding builder for faithful TIGER."""
    def __init__(self, model_name="sentence-transformers/sentence-t5-base", batch_size=64, device=None):
        self.model_name, self.batch_size, self.device = model_name, batch_size, device

    def build(self, dataset, item_texts):
        if len(item_texts) != dataset.num_items:
            raise ValueError(f"Expected {dataset.num_items} item texts, got {len(item_texts)}.")
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as error:
            raise ImportError("Sentence-T5 needs transformers and sentencepiece. Run: pip install -r requirements.txt") from error
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoder = T5EncoderModel.from_pretrained(self.model_name).to(device).eval()
        encoded = [torch.zeros(encoder.config.d_model if hasattr(encoder.config, "d_model") else encoder.config.hidden_size)]
        with torch.no_grad():
            for start in range(1, len(item_texts), self.batch_size):
                batch = item_texts[start:start + self.batch_size]
                inputs = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                hidden = encoder(**inputs).last_hidden_state; mask = inputs["attention_mask"].unsqueeze(-1)
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                encoded.extend(vectors.cpu())
        embeddings = torch.stack(encoded)
        return RepresentationArtifact(embeddings, torch.arange(dataset.num_items),
          {"kind":"sentence_t5", "model_name":self.model_name, "dim":embeddings.shape[1], "fit_scope":"frozen_pretrained", "split_hash":dataset.split_hash})
