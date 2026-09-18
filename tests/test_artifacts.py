import pytest
from genrec.data import make_synthetic_dataset
from genrec.representations import HashedRepresentationBuilder
from genrec.item_indexes import RQKMeansBuilder, RQVAEBuilder

def test_index_handles_collisions_and_split_guard():
    data=make_synthetic_dataset(); index=RQKMeansBuilder(codebook_size=2).build(data, HashedRepresentationBuilder().build(data))
    assert index.get_items(index.get_code(1)); assert index.build_trie()
    with pytest.raises(ValueError):
        from genrec.artifacts import _verify
        _verify(index.metadata, "different")


def test_rqvae_builds_diagnostic_semantic_ids():
    data = make_synthetic_dataset(users=4, items=8)
    representation = HashedRepresentationBuilder(dim=6).build(data)
    index = RQVAEBuilder(codebook_size=4, latent_dim=3, hidden_dims=(5,), epochs=2,
        batch_size=4, lr=.01, optimizer="adamw", kmeans_iterations=2, log_every=0).build(data, representation)
    assert index.item_to_code.shape[0] == data.num_items
    assert len(index.metadata["final_diagnostics"]["codebook_usage"]) == 3
    assert index.get_items(index.get_code(1))
