# DualRec: Dual-Path Recommendation for Stable and Unstable Users

Official implementation of:

> **Modeling Consumption Habit Heterogeneity: Dual-Path Recommendation for Stable and Unstable Users**
> Yuan-Yuan Xu, Chao Pang, Heng-Ru Zhang, Fan Min. *Knowledge-Based Systems* (under review).

DualRec models the heterogeneity of users' consumption habits. A **dual-metric
mechanism** (standardized Euclidean distance + Jaccard similarity) classifies
each user as **stable** or **unstable**. Two tailored recommendation paths are
then applied:

| User type | Path | Method | Role |
|-----------|------|--------|------|
| Stable | IID | Sequential recommendation with **Wasserstein self-attention** | Resolve behavioral uncertainty |
| Unstable | OOD | Causal recommendation with **counterfactual inference** | Mitigate outdated-interaction bias |

## Repository structure

```
DualRec/
├── code/
│   ├── Classify/      # Dual-metric user classification (stable / unstable)
│   ├── IID/           # Stable-user path: stochastic sequential recommendation
│   ├── OOD/           # Unstable-user path: causal counterfactual recommendation
│   ├── eval_merged.py # Merge both paths and report combined metrics (Table 4)
│   ├── evaluate.py
│   └── run_*.sh       # Convenience scripts
└── data/              # NOT included — see "Datasets" below
```

### `code/Classify/`
Splits each dataset into workday / weekend subsets, computes the feature-space
Euclidean distance and the interaction-space Jaccard similarity, and partitions
users into `iid_users` (stable) and `ood_users` (unstable). One entry script per
dataset: `Classification.py` (Meituan), `Classification_Taobao.py`,
`Classification_ml1m.py`, `Classification_Beauty.py`.

### `code/IID/` (stable users)
Stochastic sequential recommendation. Each item is embedded as a Gaussian
distribution; a causal-masked **Wasserstein self-attention** layer aggregates
the sequence. Trained with BPR + PVN + stability-consistency losses.
- Entry: `main004.py` → `DistSAModel`
- Saves test predictions to `output/<Dataset>_predictions_test.npy`.

### `code/OOD/` (unstable users)
Causal counterfactual recommendation (COR-style). A feature shift is modeled as
an intervention; outdated interactions are reweighted and a counterfactual
preference is inferred, constrained by a counterfactual-consistency loss.
- Entry: `main002.py` → `COR`
- Saves test predictions to `output/<Dataset>_best_test_predict.npy`.

### `code/eval_merged.py`
Routes each user to its path (stable → IID, unstable → OOD), concatenates the
two sets of predictions, and reports **Recall@K / NDCG@K / MRR@K** for the IID
part, the OOD part, and the **combined DualRec** result (paper Table 4).

## Requirements

- Python ≥ 3.7
- PyTorch (tested with CUDA-enabled GPUs; CPU falls back automatically)
- NumPy, SciPy, scikit-learn

```bash
pip install torch numpy scipy scikit-learn
```

## Datasets

Datasets are **not** bundled (size / license constraints). Download from the
official sources and place each under `data/<Dataset>/`:

| Dataset | Source |
|---------|--------|
| Meituan | https://www.biendata.xyz/competition/smp2021_2/ |
| Taobao  | https://tianchi.aliyun.com/dataset/56 |
| ML-1M   | https://grouplens.org/datasets/movielens/1m/ |
| Beauty  | https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation |

After downloading, run the preprocessing scripts under `code/Classify/` (and the
per-dataset `datasetting.py` / `user_feature_setting.py`) to produce:

```
data/<Dataset>/
├── iid_users.npy            # stable user ids
├── ood_users.npy            # unstable user ids
├── user_feature.npy         # standardized behavioral features (workday vs weekend)
├── iid_users.txt            # stable-user interaction sequences (for the IID path)
├── ood_train.npy / ood_val.npy / ood_test.npy   # unstable-user splits (for the OOD path)
└── ...
```

## Usage

Run the three stages in order. Edit the active `--data_dir` / `--dataset` line
at the top of each entry script to switch datasets.

```bash
# 1) Classify users (per dataset)
cd code/Classify
python Classification.py            # Meituan
python Classification_Taobao.py
python Classification_ml1m.py
python Classification_Beauty.py

# 2a) Stable-user path (IID)
cd ../IID
python main004.py                   # -> output/<Dataset>_predictions_test.npy

# 2b) Unstable-user path (OOD)
cd ../OOD
python main002.py                   # -> output/<Dataset>_best_test_predict.npy

# 3) Merge both paths and report combined metrics
cd ..
python eval_merged.py
```

## Key hyperparameters

Stable path (`code/IID/main004.py`): `--lr` (0.0001–0.002),
`--attention_probs_dropout_prob` (= ε_s, 0.1–1.0), `--pvn_weight` (λ),
`--contrast_weight` (γ, stability-consistency loss).

Unstable path (`code/OOD/main002.py`): `--lr` (0.0001–0.002), `--dropout`
(= ε_u, 0.1–1.0), `--anneal_cap` (β, 0.0–1.0), `--lambda1/2/3`
(weights of L_KL / L_reg / L_CONS in the composite unstable loss).

Fixed across datasets: max sequence length `n = 50`, batch size `256`,
200 epochs with early stopping (patience 20), Adam (β1=0.9, β2=0.999).

## Citation

```bibtex
@article{dualrec,
  title  = {Modeling Consumption Habit Heterogeneity: Dual-Path Recommendation for Stable and Unstable Users},
  author = {Xu, Yuan-Yuan and Pang, Chao and Zhang, Heng-Ru and Min, Fan},
  journal= {Knowledge-Based Systems},
  year   = {2026},
  note   = {Under review}
}
```

## Contact

Heng-Ru Zhang — zhanghr@swpu.edu.cn
Chao Pang — 202422000648@stu.swpu.edu.cn

Southwest Petroleum University, Chengdu, China.
