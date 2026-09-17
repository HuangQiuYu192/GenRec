import torch
from genrec.evaluation import ranking_metrics

def test_ranking_metrics():
    result=ranking_metrics(torch.tensor([[2,1],[3,4]]),torch.tensor([2,4]),ks=(1,2))
    assert result["Recall@1"] == .5 and result["Recall@2"] == 1.0

