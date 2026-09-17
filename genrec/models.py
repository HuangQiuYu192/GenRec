import torch
from torch import nn


class BaseGRModel(nn.Module):
    def training_step(self, batch): raise NotImplementedError
    @torch.no_grad()
    def recommend(self, batch, k, decoder=None): raise NotImplementedError


class TigerModel(BaseGRModel):
    """Small TIGER-style model: encode item history, predict each SID level independently."""
    def __init__(self, num_items, item_index, hidden_dim=64):
        super().__init__(); self.item_index = item_index
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in item_index.vocab_sizes)

    def _state(self, histories):
        embedded = self.item_embedding(histories); _, state = self.encoder(embedded); return state[-1]

    def training_step(self, batch):
        state = self._state(batch["history"]); codes = self.item_index.item_to_code[batch["target"]].to(state.device)
        losses = [nn.functional.cross_entropy(head(state), codes[:, level]) for level, head in enumerate(self.heads)]
        loss = torch.stack(losses).mean(); return {"loss": loss, "metrics": {"rec_loss": loss.detach().item()}}

    @torch.no_grad()
    def recommend(self, batch, k, decoder=None):
        state = self._state(batch["history"]); logits = [head(state) for head in self.heads]
        candidates = self.item_index.item_to_code[1:].to(state.device)
        scores = sum(logit[:, candidates[:, level]] for level, logit in enumerate(logits))
        return scores.topk(min(k, candidates.shape[0]), dim=1).indices.add(1)

