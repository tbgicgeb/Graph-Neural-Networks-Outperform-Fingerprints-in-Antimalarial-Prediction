# Graph-Neural-Networks-Outperform-Fingerprints-in-Antimalarial-Prediction
Graph Neural Networks Outperform Fingerprint Models for Predicting Antimalarial Activity of Structurally Novel Compounds: A Systematic Out-of-Distribution Benchmark

Four models   **Random Forest**, **XGBoost**, **ChemProp** (D-MPNN) and **GNEprop** (GIN)   are compared on 27,493 *Plasmodium falciparum* screening compounds, under three ways of splitting the data and in two settings each (default settings and hyperparameter-tuned).

**Main result.** On the hardest split   where test compounds come from chemical neighbourhoods the model has never seen   the two graph models beat the two fingerprint models by **+0.041 AUPRC** (7 of 8 folds, paired *t* *p* = 0.007).

## The three splits

Every model is run on all three. They get progressively harder, which is the point of the study.

| Split | How test compounds are chosen | How similar test is to training |
|---|---|---|
| **Random** | at random | mean Tanimoto 0.581   34.8% of test actives have a near-duplicate in training |
| **Scaffold** | whole Bemisâ€“Murcko scaffolds held out | 0.510   17.5% near-duplicates |
| **Scaffold-cluster** | whole clusters of related scaffolds held out | 0.402   3.0% near-duplicates |

Scaffold-cluster is the out-of-distribution benchmark and the one the headline numbers come from. Splits are fixed (seed 42), so everyone gets the same folds.

## The two settings

| Setting | What it means |
|---|---|
| **Default** | Each model is evaluated using the default hyperparameter settings provided by its respective library, without additional tuning.|
| **Tuned** | hyperparameters searched for each model, then the best configuration used |

Both settings are reported, because the comparison changes between them: at default settings the four models look alike, and the graph advantage only appears once every model has been properly tuned.

## Running the models 

Random Forest and XGBoost run on a normal CPU. ChemProp and GNEprop need a CUDA GPU, and each needs its own conda environment because the two frameworks want different versions of PyTorch.

### Environment 1   fingerprint models, figures and tables

```bash
conda create -n antimalarial python=3.11 -y
conda activate antimalarial
pip install -r requirements.txt
```

### Environment 2   ChemProp

```bash
conda create -n chemprop_env python=3.11 -y
conda activate chemprop_env
pip install chemprop==2.2.2
pip install pandas scikit-learn
```

### Environment 3   GNEprop

GNEprop is not on PyPI. Clone it from Genentech and install its dependencies:

```bash
conda create -n gneprop python=3.11 -y
conda activate gneprop
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.4.0 pytorch-lightning==1.9.5
pip install rdkit==2023.03.2 pandas scikit-learn
git clone https://github.com/Genentech/gneprop
```

### Versions used in the paper

Python 3.11 Â· RDKit 2023.03.2 Â· scikit-learn 1.6.1 Â· XGBoost 3.2.0 Â·
ChemProp 2.2.2 Â· PyTorch 2.2.1 (CUDA 12.1) Â· PyTorch Lightning 1.9.5 Â·
PyTorch Geometric 2.4.0 Â· Optuna 4.8.0.
GPUs: NVIDIA RTX A4000 (16 GB); NVIDIA L40S (46 GB, for the hyperparameter search).

### 1. Dataset and splits   `code/1_data/`

| Script | What it does |
|---|---|
| `01_prepare_dataset.py` | curates the dataset and computes descriptors |
| `02_make_splits.py` | builds the random, scaffold and scaffold-cluster 8-fold splits |
| `03_check_split_similarity.py` | measures how similar test compounds are to training |

### 2. Models at default settings   `code/2_models_default/`

Each script covers all three splits.

| Script | Models |
|---|---|
| `01_random_forest_and_xgboost.py` | Random Forest, XGBoost (CPU) |
| `02_chemprop.py` | ChemProp (GPU) |
| `03_gneprop.py` | GNEprop (GPU) |

### 3. Models after hyperparameter tuning   `code/3_models_tuned/`

| Script | Model and search |
|---|---|
| `01_random_forest.py` | 50 Optuna trials per fold, nested 3-fold inner CV |
| `02_xgboost.py` | same protocol |
| `03_chemprop.py` | per-fold search on a scaffold-disjoint inner validation set |
| `04_chemprop_random_and_scaffold_splits.py` | the tuned ChemProp on the two easier splits |
| `05_gneprop.py` | configuration chosen on fold-0 validation score, applied to all folds |

## Model frameworks

This repository holds the benchmark only, not the frameworks themselves:

- **GNEprop**   Genentech, Apache-2.0. Scalia et al., *Nat Biotechnol* 44:1154 1167 (2026).
- **ChemProp**   <https://github.com/chemprop/chemprop> (v2.2.2).

## License

- **Third-party model frameworks** keep their own licences. GNEprop is Apache-2.0 (Genentech) and ChemProp is MIT; neither is bundled in this repository, so install each from its own source and follow the terms that come with it. The same applies to RDKit, scikit-learn, XGBoost and PyTorch.

If you reuse any part of this repository, cite the manuscript and give the upstream frameworks and data sources their own attribution.





