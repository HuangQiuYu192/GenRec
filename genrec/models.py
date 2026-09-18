"""TIGER retrieval model backed by a T5 encoder-decoder."""
import torch
from torch import nn


class BaseGRModel(nn.Module):
    def training_step(self, batch): raise NotImplementedError
    @torch.no_grad()
    def recommend(self, batch, k, decoder=None): raise NotImplementedError


class TigerSerializer:
    """Disjoint vocabulary ranges for each SID level and item separators."""
    PAD, EOS, SEP = 0, 1, 2

    def __init__(self, vocab_sizes):
        self.vocab_sizes = list(vocab_sizes); self.offsets = []; cursor = 3
        for size in vocab_sizes: self.offsets.append(cursor); cursor += size
        self.vocab_size = cursor

    def code_tokens(self, codes): return codes + torch.tensor(self.offsets, device=codes.device)
    def local_code(self, token, level): return token - self.offsets[level]


class TigerModel(BaseGRModel):
    """TIGER's T5-style seq2seq generator over frozen Semantic IDs.

    The model deliberately omits user-ID tokens: this is the requested
    non-personalized ablation, while every remaining architectural setting is
    the paper's 4 encoder/4 decoder layers, d_model=128, d_ff=1024, six heads
    with d_kv=64, ReLU, and dropout 0.1.
    """
    def __init__(self, num_items, item_index, hidden_dim=128, num_heads=6, num_layers=4, dropout=.1,
                 max_history=20, d_ff=1024, d_kv=64, beam_width=20):
        super().__init__()
        if item_index is None: raise ValueError("TigerModel requires an ItemIndexArtifact.")
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except ImportError as error:
            raise ImportError("TIGER's T5 generator requires transformers. Run: pip install -r requirements.txt") from error
        self.item_index, self.max_history, self.beam_width = item_index, max_history, beam_width
        self.serializer = TigerSerializer(item_index.vocab_sizes)
        self.register_buffer("item_to_code", item_index.item_to_code.clone())
        config = T5Config(vocab_size=self.serializer.vocab_size, d_model=hidden_dim, d_kv=d_kv, d_ff=d_ff,
            num_layers=num_layers, num_decoder_layers=num_layers, num_heads=num_heads, dropout_rate=dropout,
            feed_forward_proj="relu", pad_token_id=TigerSerializer.PAD, eos_token_id=TigerSerializer.EOS,
            decoder_start_token_id=TigerSerializer.PAD, use_cache=False)
        self.t5 = T5ForConditionalGeneration(config)

    def _history_tokens(self, histories):
        rows = []
        for history in histories.tolist():
            tokens = []
            for item in [item for item in history if item][-self.max_history:]:
                tokens.extend(self.serializer.code_tokens(self.item_to_code[item]).tolist())
                tokens.append(TigerSerializer.SEP)
            rows.append(tokens or [TigerSerializer.SEP])
        width = max(map(len, rows)); result = histories.new_full((len(rows), width), TigerSerializer.PAD)
        for index, row in enumerate(rows): result[index, :len(row)] = torch.tensor(row, device=histories.device)
        return result

    def _decode_logits(self, source, decoder_input):
        return self.t5(input_ids=source, attention_mask=source.ne(TigerSerializer.PAD),
            decoder_input_ids=decoder_input).logits

    def training_step(self, batch):
        source = self._history_tokens(batch["history"])
        codes = self.item_to_code[batch["target"]]
        labels = torch.cat((self.serializer.code_tokens(codes), codes.new_full((len(codes), 1), TigerSerializer.EOS)), 1)
        output = self.t5(input_ids=source, attention_mask=source.ne(TigerSerializer.PAD), labels=labels)
        return {"loss": output.loss, "metrics": {"rec_loss": output.loss.detach().item()}}

    def _trie(self):
        trie = {}
        for code in self.item_index.code_to_items:
            node = trie
            for token in self.serializer.code_tokens(torch.tensor(code)).tolist(): node = node.setdefault(token, {})
        return trie

    @torch.no_grad()
    def recommend(self, batch, k, decoder=None):
        source, trie = self._history_tokens(batch["history"]), self._trie()
        beam_width = max(k, self.beam_width)
        beams = [[([TigerSerializer.PAD], 0.0, trie)] for _ in range(source.shape[0])]
        for _ in range(self.item_index.code_length):
            owners, flat = [], []
            for row, row_beams in enumerate(beams): owners.extend([row] * len(row_beams)); flat.extend(row_beams)
            prefixes = torch.tensor([prefix for prefix, _, _ in flat], device=source.device)
            logits_batch = self._decode_logits(source[torch.tensor(owners, device=source.device)], prefixes)[:, -1]
            next_beams = [[] for _ in beams]
            for row, (prefix, score, node), logits in zip(owners, flat, logits_batch):
                valid = list(node); values = torch.log_softmax(logits[valid], 0)
                next_beams[row].extend((prefix + [token], score + value, node[token]) for token, value in zip(valid, values.tolist()))
            beams = [sorted(row_beams, key=lambda entry: entry[1], reverse=True)[:beam_width] for row_beams in next_beams]
        predictions = []
        for row_beams in beams:
            ranked = []
            for prefix, _, _ in row_beams:
                code = [self.serializer.local_code(token, level) for level, token in enumerate(prefix[1:])]
                ranked.extend(self.item_index.get_items(code))
                if len(ranked) >= k: break
            predictions.append((ranked + [0] * k)[:k])
        return torch.tensor(predictions, device=source.device)
