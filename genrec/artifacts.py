from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import torch

from .utils import stable_hash


def _verify(metadata, split_hash):
    found = metadata.get("split_hash")
    if found != split_hash:
        raise ValueError(f"Artifact was built from split hash {found}, but current dataset split hash is {split_hash}. Please rebuild the artifact.")


@dataclass
class RepresentationArtifact:
    embeddings: torch.Tensor
    item_ids: torch.Tensor
    metadata: dict

    def __getitem__(self, item_id):
        return self.embeddings[item_id]

    @property
    def dim(self): return self.embeddings.shape[1]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(self, path)

    @classmethod
    def load(cls, path: Path, split_hash: str):
        result = torch.load(path, map_location="cpu"); _verify(result.metadata, split_hash); return result


@dataclass
class ItemIndexArtifact:
    item_to_code: torch.Tensor
    vocab_sizes: list
    metadata: dict

    def __post_init__(self):
        self._code_to_items = None; self._trie = None

    @property
    def code_length(self): return self.item_to_code.shape[1]

    @property
    def code_to_items(self):
        if self._code_to_items is None:
            result = defaultdict(list)
            for item, code in enumerate(self.item_to_code.tolist()):
                if item: result[tuple(code)].append(item)
            self._code_to_items = dict(result)
        return self._code_to_items

    def get_code(self, item_id): return self.item_to_code[item_id]
    def get_items(self, code): return self.code_to_items.get(tuple(torch.as_tensor(code).tolist()), [])

    def build_trie(self):
        trie = {}
        for code in self.code_to_items:
            node = trie
            for token in code: node = node.setdefault(token, {})
        self._trie = trie
        return trie

    def diagnostics(self):
        groups = list(self.code_to_items.values()); total = self.item_to_code.shape[0] - 1
        collisions = sum(len(g) for g in groups if len(g) > 1)
        return {"unique_code_ratio": len(groups) / max(total, 1), "collision_rate": collisions / max(total, 1),
                "capacity_bits": sum(torch.log2(torch.tensor(v)).item() for v in self.vocab_sizes)}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(self, path)

    @classmethod
    def load(cls, path: Path, split_hash: str):
        result = torch.load(path, map_location="cpu"); _verify(result.metadata, split_hash); result.__post_init__(); return result

