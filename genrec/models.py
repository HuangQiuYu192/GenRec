"""TIGER retrieval model and its benchmark-facing interface."""
import torch
from torch import nn


class BaseGRModel(nn.Module):
    def training_step(self, batch): raise NotImplementedError
    @torch.no_grad()
    def recommend(self, batch, k, decoder=None): raise NotImplementedError


class TigerSerializer:
    """Level-specific SID vocabulary and explicit item-boundary serialization."""
    PAD, BOS, SEP = 0, 1, 2
    def __init__(self, vocab_sizes):
        self.vocab_sizes = list(vocab_sizes); self.offsets = []; cursor = 3
        for size in vocab_sizes: self.offsets.append(cursor); cursor += size
        self.vocab_size = cursor
    def code_tokens(self, codes): return codes + torch.tensor(self.offsets, device=codes.device)
    def local_code(self, token, level): return token - self.offsets[level]


class TigerModel(BaseGRModel):
    """Frozen RQ-VAE SIDs + seq2seq Transformer + trie-constrained beam search."""
    def __init__(self, num_items, item_index, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_history=50):
        super().__init__()
        if item_index is None: raise ValueError("TigerModel requires an external ItemIndexArtifact, but item_index=null.")
        if hidden_dim % num_heads: raise ValueError("hidden_dim must be divisible by num_heads.")
        self.item_index, self.max_history = item_index, max_history; self.serializer = TigerSerializer(item_index.vocab_sizes)
        self.register_buffer("item_to_code", item_index.item_to_code.clone())
        self.token_embedding = nn.Embedding(self.serializer.vocab_size, hidden_dim, padding_idx=TigerSerializer.PAD)
        self.position_embedding = nn.Embedding(max_history * (item_index.code_length + 1) + item_index.code_length + 1, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.output = nn.Linear(hidden_dim, self.serializer.vocab_size, bias=False)

    def _embed(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def _history_tokens(self, histories):
        rows = []
        for history in histories.tolist():
            tokens = []
            for item in [item for item in history if item][-self.max_history:]:
                tokens.extend(self.serializer.code_tokens(self.item_to_code[item]).tolist()); tokens.append(TigerSerializer.SEP)
            rows.append(tokens or [TigerSerializer.SEP])
        width = max(map(len, rows)); result = histories.new_full((len(rows), width), TigerSerializer.PAD)
        for index, row in enumerate(rows): result[index, :len(row)] = torch.tensor(row, device=histories.device)
        return result

    def _decode_logits(self, source, target):
        source_padding, target_padding = source.eq(TigerSerializer.PAD), target.eq(TigerSerializer.PAD)
        length = target.shape[1]; causal = torch.full((length, length), float("-inf"), device=target.device).triu(1)
        hidden = self.transformer(self._embed(source), self._embed(target), tgt_mask=causal, src_key_padding_mask=source_padding,
                                  tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding)
        return self.output(hidden)

    def training_step(self, batch):
        source, codes = self._history_tokens(batch["history"]), self.item_to_code[batch["target"]]
        labels = self.serializer.code_tokens(codes)
        decoder_input = torch.cat((labels.new_full((labels.shape[0], 1), TigerSerializer.BOS), labels[:, :-1]), 1)
        logits = self._decode_logits(source, decoder_input); losses = []
        for level, size in enumerate(self.serializer.vocab_sizes):
            start = self.serializer.offsets[level]; losses.append(nn.functional.cross_entropy(logits[:, level, start:start + size], codes[:, level]))
        loss = torch.stack(losses).mean()
        return {"loss": loss, "metrics": {"rec_loss": loss.detach().item()}}

    def _trie(self):
        trie = {}
        for code in self.item_index.code_to_items:
            node = trie
            for token in self.serializer.code_tokens(torch.tensor(code)).tolist(): node = node.setdefault(token, {})
        return trie

    @torch.no_grad()
    def recommend(self, batch, k, decoder=None):
        source, trie, all_predictions = self._history_tokens(batch["history"]), self._trie(), []
        beam_width = max(k, 20)
        for row in range(source.shape[0]):
            beams = [([TigerSerializer.BOS], 0.0, trie)]
            for _ in range(self.item_index.code_length):
                candidates = []
                prefixes = torch.tensor([prefix for prefix, _, _ in beams], device=source.device)
                logits_batch = self._decode_logits(source[row:row + 1].expand(len(beams), -1), prefixes)[:, -1]
                for (prefix, score, node), logits in zip(beams, logits_batch):
                    valid = list(node); values = torch.log_softmax(logits[valid], 0)
                    candidates.extend((prefix + [token], score + value, node[token]) for token, value in zip(valid, values.tolist()))
                beams = sorted(candidates, key=lambda entry: entry[1], reverse=True)[:beam_width]
            ranked = []
            for prefix, _, _ in beams:
                code = [self.serializer.local_code(token, level) for level, token in enumerate(prefix[1:])]
                ranked.extend(self.item_index.get_items(code))
                if len(ranked) >= k: break
            all_predictions.append((ranked + [0] * k)[:k])
        return torch.tensor(all_predictions, device=source.device)
