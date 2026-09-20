from genrec.data import make_synthetic_dataset
from genrec.item_indexes import RQKMeansBuilder
from genrec.models import TigerSerializer
from genrec.training import make_batches, prefix_examples
from genrec.representations import HashedRepresentationBuilder


def test_tiger_sid_vocabulary_has_collision_level():
    data = make_synthetic_dataset(users=2, items=4)
    index = RQKMeansBuilder(codebook_size=2, iterations=1).build(data, HashedRepresentationBuilder(dim=4).build(data))
    serializer = TigerSerializer(index.vocab_sizes)
    assert index.item_to_code.shape[1] == 4  # three RQ codes plus TIGER collision token
    assert serializer.vocab_size == 3 + sum(index.vocab_sizes)


def test_user_aware_examples_survive_batching():
    examples = prefix_examples([[1, 2, 3]], max_history=20, user_ids=[17])
    batch = next(make_batches(examples, 8, "cpu"))
    assert batch["user"].tolist() == [17, 17]
    assert batch["target"].tolist() == [2, 3]


def test_serializer_allocates_a_disjoint_user_vocabulary():
    serializer = TigerSerializer([2, 3], user_token_count=7)
    assert serializer.vocab_size == 3 + 2 + 3 + 7
    assert serializer.user_token_count == 7
    assert serializer.user_token(__import__("torch").tensor([0, 7])).tolist() == [8, 8]
