from genrec.data import make_synthetic_dataset
from genrec.item_indexes import RQKMeansBuilder
from genrec.models import TigerSerializer
from genrec.representations import HashedRepresentationBuilder


def test_tiger_sid_vocabulary_has_collision_level():
    data = make_synthetic_dataset(users=2, items=4)
    index = RQKMeansBuilder(codebook_size=2, iterations=1).build(data, HashedRepresentationBuilder(dim=4).build(data))
    serializer = TigerSerializer(index.vocab_sizes)
    assert index.item_to_code.shape[1] == 4  # three RQ codes plus TIGER collision token
    assert serializer.vocab_size == 3 + sum(index.vocab_sizes)
