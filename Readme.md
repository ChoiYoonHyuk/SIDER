<img width="4362" height="1832" alt="Image" src="https://github.com/user-attachments/assets/60a6914e-fd95-4ff9-9266-a3ffaf4e13af" />

## Overview

**SIDER: Syndrome-Guided Edge-Relation Decoding for Heterophilic Graph Mining** provides a single-node-classification training pipeline for both homophilic and heterophilic graph benchmarks.

The current implementation includes multiple graph-learning backbones and automatically selects a suitable default model depending on the dataset type:

- **Homophilic citation graphs**: GCNII-style propagation by default.
- **Small heterophilic graphs**: H2GCN-lite by default.
- **Large heterophilic graphs**: GPR-GNN-style propagation by default.

The code supports reproducible training, early stopping, multiple random runs, label smoothing, DropEdge, directed/undirected graph handling, and validation-based model selection.

## Supported Models

The `--model` argument supports:

| Model | Description |
| --- | --- |
| `auto` | Automatically selects a model based on the dataset ID. |
| `gcnii` | GCNII-style deep graph convolution with initial residual propagation. |
| `appnp` | APPNP-style personalized propagation. |
| `dagnn` | DAGNN-style adaptive propagation aggregation. |
| `gpr` | GPR-GNN-style learnable propagation weights. |
| `mlp` | Feature-only MLP baseline. |
| `h2gcn` | Lightweight H2GCN-style model for heterophilic graphs. |
| `linkx` | Lightweight LINKX-style model combining structural and feature embeddings. |

## Dataset IDs

Run the script with an integer dataset ID:

| ID | Dataset | Source |
| --- | --- | --- |
| `0` | Cora | Planetoid |
| `1` | CiteSeer | Planetoid |
| `2` | PubMed | Planetoid |
| `3` | Chameleon | WikipediaNetwork |
| `4` | Squirrel | WikipediaNetwork |
| `5` | Actor | Actor |
| `6` | Cornell | WebKB |
| `7` | Texas | WebKB |
| `8` | Wisconsin | WebKB |
| `9` | Penn94 | LINKXDataset |
| `10` | arXiv-year | OGB `ogbn-arxiv` with quantile year labels |
| `11` | snap-patents | SNAP patents `.mat` file with quantile year labels |

## Execution

The main entry point expects a dataset ID as the first positional argument.

```bash
python main.py 0   # Cora
python main.py 1   # CiteSeer
python main.py 2   # PubMed
python main.py 3   # Chameleon
python main.py 4   # Squirrel
python main.py 5   # Actor
python main.py 6   # Cornell
python main.py 7   # Texas
python main.py 8   # Wisconsin
python main.py 9   # Penn94
python main.py 10  # arXiv-year
python main.py 11  # snap-patents
```

## Example Commands

Train with automatic model selection:

```bash
python main.py 3 --model auto
```

Run H2GCN-lite on a small heterophilic dataset:

```bash
python main.py 4 --model h2gcn --runs 10
```

Run GPR-GNN-style propagation on Penn94:

```bash
python main.py 9 --model gpr --runs 5
```

Train a feature-only MLP baseline:

```bash
python main.py 6 --model mlp
```


## Important Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `data` | required | Integer dataset ID from `0` to `11`. |
| `--model` | `auto` | Model choice: `auto`, `gcnii`, `appnp`, `dagnn`, `gpr`, `mlp`, `h2gcn`, or `linkx`. |
| `--root` | `/tmp` | Root directory for dataset storage. |
| `--epochs` | model-specific | Maximum number of training epochs. |
| `--runs` | `1` | Number of repeated runs using consecutive seeds. |
| `--seed` | `42` | Base random seed. |
| `--hidden` | model-specific | Hidden dimension. |
| `--layers` | model-specific | Number of GCNII layers or model-specific depth. |
| `--dropout` | model-specific | Dropout rate. |
| `--dropedge` | model-specific | Edge dropout rate during training. |
| `--lr` | model-specific | Learning rate. |
| `--weight-decay` | `None` | Overrides both `--wd1` and `--wd2`. |
| `--alpha` | model-specific | Teleport / residual mixing parameter. |
| `--lamda` | model-specific | GCNII lambda parameter. |
| `--propagation-steps` | model-specific | Number of propagation steps for APPNP, DAGNN, GPR, or H2GCN. |
| `--selection` | model-specific | Model checkpoint selection criterion: `val_loss`, `val_acc`, or `hybrid`. |
| `--patience` | model-specific | Early stopping patience. |
| `--label-smoothing` | model-specific | Label smoothing coefficient. |
| `--grad-clip` | `0.0` | Gradient clipping norm. |
| `--split-id` | `0` | Split index for datasets with multiple masks. |
| `--train-per-class` | `20` | Number of training nodes per class for balanced random splits. |
| `--num-val` | `500` | Number of validation nodes for random splits. |
| `--num-test` | `1000` | Number of test nodes for random splits. |
| `--train-prop` | `0.5` | Training proportion for large heterophily datasets. |
| `--val-prop` | `0.25` | Validation proportion for large heterophily datasets. |
| `--force-undirected` | enabled | Converts edges to undirected. |
| `--keep-directed` | disabled | Keeps the original directed edge orientation. |
| `--quiet` | disabled | Disables tqdm progress display. |

## Default Model Selection

When `--model auto` is used:

| Dataset IDs | Dataset Type | Default Model |
| --- | --- | --- |
| `0`-`2` | Homophilic citation graphs | `gcnii` |
| `3`-`8` | Small heterophilic graphs | `h2gcn` |
| `9`-`11` | Large heterophilic graphs | `gpr` |


## Notes

- For `h2gcn` and `linkx`, self-loops are not added to the normalized adjacency. This avoids prematurely mixing ego and neighbor channels in heterophilic settings.
- For `gpr` on heterophilic datasets, the default initialization uses alternating signed propagation weights through `--gpr-init alt`.
- For `arXiv-year` and `snap-patents`, continuous years are converted into balanced ordinal classes using quantile binning.
- `snap-patents` can be downloaded automatically with `gdown`, or loaded from a local `.mat` file using `--snap-patents-path`.

