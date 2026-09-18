# GenRec

一个面向生成式推荐研究的轻量、可复现 benchmark。它统一数据划分、artifact 缓存、评测、训练和资源记录，同时将模型内部算法保留在各个 baseline 中。

## Quick start

```bash
pip install -r requirements.txt

# 无需下载数据的端到端 smoke run
python run.py --debug --epochs 2

# 下载官方 Beauty 5-core reviews 与正常 metadata，并处理为可审计数据文件
python -m genrec.data.prepare --dataset Beauty --download
python scripts/build_representation.py --dataset Beauty --representation sentence_t5 --device cuda
python scripts/build_item_index.py --dataset Beauty --representation sentence_t5 --item-index rqvae --device cuda
python run.py --dataset Beauty --representation sentence_t5 --item-index rqvae --protocol faithful
```

Amazon Beauty 来自 [UCSD Amazon product data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 所列的官方 5-core review file（Beauty 为 198,502 条 reviews）及其正常 metadata 文件。默认不再重复 k-core 过滤；按用户时间排序后，最后两个交互依次作为 validation/test，其余为 train。

处理后数据统一位于 `data/Beauty/`：

- `interactions.txt`：每行 `raw_user_id raw_item_id_1 raw_item_id_2 ...`，物品按时间顺序排列。
- `items.jsonl`：每行一个 active item 的 `item_id`、`title`、`categories`、`brand`。
- `manifest.json`：数据来源、规模与 split hash；`dataset.pt` 是派生加载缓存。
- `stats.json`：交互/用户/物品规模、稀疏度、序列长度、item 流行度、长尾占比、划分与内部 ID 映射策略。
- `artifacts/`、`outputs/`：同一数据集对应的表示、SID 与实验产物。

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
