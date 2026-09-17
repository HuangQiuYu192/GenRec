# GenRec

一个面向生成式推荐研究的轻量、可复现 benchmark。它统一数据划分、artifact 缓存、评测、训练和资源记录，同时将模型内部算法保留在各个 baseline 中。

## Quick start

```bash
pip install -r requirements.txt

# 无需下载数据的端到端 smoke run
python run.py --debug --epochs 2

# 从 UCSD 官方 Amazon 链接下载 Beauty 的官方 5-core reviews 后处理
python scripts/prepare_dataset.py --amazon beauty --download --dataset amazon_beauty

# 如需从完整 reviews 文件自行重建 5-core：
python scripts/prepare_dataset.py --amazon beauty --amazon-source full --download --dataset amazon_beauty

# 或将自备 CSV（user_id,item_id,timestamp）处理为 benchmark cache
python scripts/prepare_dataset.py --input data/interactions.csv --dataset amazon_beauty
python scripts/build_representation.py --dataset amazon_beauty --representation hashed
python scripts/build_item_index.py --dataset amazon_beauty --representation hashed --item-index rqkmeans
python run.py --dataset amazon_beauty --representation hashed --item-index rqkmeans
```

CSV 必须包含 `user_id,item_id,timestamp` 三列。Amazon Beauty 来自 [UCSD Amazon product data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 所列的官方 5-core review file（Beauty 为 198,502 条 reviews）。该文件仍会通过本项目的迭代 5-core 校验（用户与物品交互数均至少为 5）；可用 `--amazon-source full` 下载完整 reviews 并在本地重建。随后按用户时间排序，最后两个交互依次作为 validation/test，其余为 train。所有可训练 artifact 只拟合 train。

## Included V1

- 可复现的 sequential split 和 split hash
- `RepresentationArtifact` 与 `ItemIndexArtifact` 的带 metadata 缓存和 split-mismatch 防护
- hashed content-style representation、train-only collaborative representation
- RQ-KMeans 与轻量 RQ-VAE item index
- TIGER-style SID prediction baseline、candidate-constrained decoder
- item-level Recall/NDCG、collision/index diagnostics、JSON outputs 和资源记录

配置既可在命令行覆盖，也可放在 `configs/`。新增 baseline 只需实现 `BaseGRModel.training_step()` 与 `recommend()`，无需修改 trainer、dataset 或 evaluator。

## TIGER pipeline

TIGER is implemented in two frozen stages: an RQ-VAE produces Semantic IDs, then a Transformer encoder-decoder autoregressively generates the next ID under trie-constrained beam search. Generated IDs are always mapped back to ranked item IDs before Recall/NDCG evaluation.

```bash
python scripts/build_representation.py --dataset amazon_beauty --representation hashed --dim 32
python scripts/build_item_index.py --dataset amazon_beauty --representation hashed \
  --item-index rqvae --code-length 3 --codebook-size 256 --epochs 30 --device cuda
python run.py --dataset amazon_beauty --representation hashed --item-index rqvae \
  --epochs 30 --batch-size 256 --hidden-dim 128 --heads 4 --layers 2 --max-history 50
```

`hashed` is a fast infrastructure smoke-test representation, so it must be reported as `protocol=controlled`; a faithful paper reproduction additionally needs a frozen item-content encoder (such as Sentence-T5) and paper-matched hyperparameters.
