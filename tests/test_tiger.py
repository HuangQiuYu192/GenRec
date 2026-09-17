from genrec.data import make_synthetic_dataset
from genrec.representations import HashedRepresentationBuilder
from genrec.item_indexes import RQKMeansBuilder
from genrec.models import TigerModel
from genrec.training import Trainer

def test_tiger_smoke():
    data=make_synthetic_dataset(users=8,items=16); index=RQKMeansBuilder(codebook_size=4,iterations=2).build(data,HashedRepresentationBuilder().build(data))
    model=TigerModel(data.num_items,index,16); trainer=Trainer(epochs=1,batch_size=4); trainer.fit(model,data)
    assert trainer.evaluate(model,data.test_examples)["Recall@20"] == 1.0
