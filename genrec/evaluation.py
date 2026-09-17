import math
import torch


def ranking_metrics(predictions, targets, ks=(5, 10, 20)):
    result = {}
    for k in ks:
        hits = predictions[:, :min(k, predictions.shape[1])].eq(targets[:, None])
        found = hits.any(1).float(); positions = hits.float().argmax(1) + 1
        result[f"Recall@{k}"] = found.mean().item()
        result[f"NDCG@{k}"] = (found / torch.log2(positions.float() + 1)).mean().item()
    return result

