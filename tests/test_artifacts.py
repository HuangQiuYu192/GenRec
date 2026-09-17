import pytest
from genrec.data import make_synthetic_dataset
from genrec.representations import HashedRepresentationBuilder
from genrec.item_indexes import RQKMeansBuilder

def test_index_handles_collisions_and_split_guard():
    data=make_synthetic_dataset(); index=RQKMeansBuilder(codebook_size=2).build(data, HashedRepresentationBuilder().build(data))
    assert index.get_items(index.get_code(1)); assert index.build_trie()
    with pytest.raises(ValueError):
        from genrec.artifacts import _verify
        _verify(index.metadata, "different")

