"""
Fingerprint Models — Random Forest & XGBoost
===================================================
Usage:
    python code/2_models_default/01_random_forest_and_xgboost.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = "malaria_dataset_clean.csv"
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
FIGURE_DIR  = "results/diagnostics/baselines"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("=" * 60)
print("BASELINE MODELS — RF & XGBOOST")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA & FINGERPRINTS
# ─────────────────────────────────────────────────────────────
df     = pd.read_csv(INPUT_FILE)
smiles = df['smiles'].tolist()
y      = df['activity'].values
print(f"Loaded: {len(df):,} compounds | Actives: {y.sum():,}")

def get_fp(smi, radius=2, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(nbits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(nbits, dtype=np.float32)

print("Computing Morgan fingerprints (radius=2, 2048 bits)...")
X = np.array([get_fp(s) for s in smiles])
print(f"Feature matrix: {X.shape}")

# Class imbalance ratio for XGBoost
n_pos = int(y.sum())
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"Class imbalance: 1:{n_neg/n_pos:.1f} "
      f"(scale_pos_weight={scale_pos_weight:.1f})")

# ─────────────────────────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────────────────────────
split_files = {
    'Random':           f"{SPLITS_DIR}/random_splits.pkl",
    'Scaffold':         f"{SPLITS_DIR}/scaffold_splits.pkl",
    'Scaffold-Cluster': f"{SPLITS_DIR}/scaffold_cluster_splits.pkl",
}
splits = {}
for name, path in split_files.items():
    with open(path, 'rb') as f:
        splits[name] = pickle.load(f)
    print(f"Loaded {name}: {len(splits[name])} folds")

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def bedroc(y_true, y_score, alpha=20.0):
    n = len(y_true)
    na = int(y_true.sum())
    if na == 0 or na == n:
        return 0.5
    order    = np.argsort(-y_score)
    ys       = y_true[order]
    ra       = na / n
    ri_sum   = sum(np.exp(-alpha*(i+1)/n)
                   for i, yi in enumerate(ys) if yi == 1)
    rie      = ri_sum / (ra*(1-np.exp(-alpha)) / (np.exp(alpha/n)-1))
    rie_max  = (1-np.exp(-alpha*ra)) / (ra*(1-np.exp(-alpha)))
    rie_min  = (1-np.exp( alpha*ra)) / (ra*(1-np.exp(-alpha)))
    return (rie - rie_min) / (rie_max - rie_min)

def evaluate(y_true, y_prob):
    if y_true.sum() == 0:
        return dict(auprc=0.0, auroc=0.5, f1=0.0, bedroc=0.0)
    auprc  = average_precision_score(y_true, y_prob)
    auroc  = roc_auc_score(y_true, y_prob)
    bd     = bedroc(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    if y_pred.sum() == 0:
        thr    = np.percentile(y_prob, 100*(1-y_true.mean()))
        y_pred = (y_prob >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(auprc=auprc, auroc=auroc, f1=f1, bedroc=bd)

# ─────────────────────────────────────────────────────────────
# GENERIC TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def run_model(model_name, ModelClass, params, splits_dict, X, y):
    all_rows  = []
    pr_curves = {}   # split_name → list of (recall, precision)

    for split_name, fold_list in splits_dict.items():
        print(f"\n  [{model_name}] {split_name}")
        fold_rows  = []
        pr_this    = []

        for fd in fold_list:
            train_idx = np.array(fd['train'])
            test_idx  = np.array(fd['test'])
            y_test    = y[test_idx]

            if y_test.sum() == 0:
                print(f"    Fold {fd['fold']}: no actives — skip")
                continue

            model = ModelClass(**params)
            model.fit(X[train_idx], y[train_idx])
            y_prob = model.predict_proba(X[test_idx])[:, 1]

            m = evaluate(y_test, y_prob)
            m.update(fold=fd['fold'], split=split_name,
                     model=model_name,
                     n_test=len(test_idx),
                     n_test_actives=int(y_test.sum()))
            fold_rows.append(m)

            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            pr_this.append((rec, prec,
                            average_precision_score(y_test, y_prob)))

            print(f"    Fold {fd['fold']}: "
                  f"AUPRC={m['auprc']:.3f}  "
                  f"AUROC={m['auroc']:.3f}  "
                  f"BEDROC={m['bedroc']:.3f}  "
                  f"F1={m['f1']:.3f}")

        if fold_rows:
            ap_vals = [r['auprc'] for r in fold_rows]
            print(f"\n  → {split_name}: "
                  f"AUPRC = {np.mean(ap_vals):.3f} ± {np.std(ap_vals):.3f}")

        all_rows.extend(fold_rows)
        pr_curves[split_name] = pr_this

    return pd.DataFrame(all_rows), pr_curves

# ─────────────────────────────────────────────────────────────
# MODEL 1: RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1: RANDOM FOREST + Morgan Fingerprints")
print("=" * 60)

rf_params = dict(
    n_estimators    = 500,
    max_depth       = None,
    min_samples_leaf= 2,
    class_weight    = 'balanced',   # handles imbalance
    n_jobs          = -1,
    random_state    = RANDOM_SEED,
)
print(f"Params: {rf_params}")

rf_df, rf_prc = run_model(
    'Random Forest', RandomForestClassifier,
    rf_params, splits, X, y
)
rf_df.to_csv(f"{RESULTS_DIR}/rf_results_per_fold.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/rf_results_per_fold.csv")

# ─────────────────────────────────────────────────────────────
# MODEL 2: XGBOOST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: XGBOOST + Morgan Fingerprints")
print("=" * 60)

xgb_params = dict(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos_weight,  # handles imbalance
    eval_metric       = 'aucpr',
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbosity         = 0,
)
print(f"Params: {xgb_params}")

xgb_df, xgb_prc = run_model(
    'XGBoost', xgb.XGBClassifier,
    xgb_params, splits, X, y
)
xgb_df.to_csv(f"{RESULTS_DIR}/xgb_results_per_fold.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/xgb_results_per_fold.csv")

# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY (Paper Table 2 — partial)")
print("=" * 60)

combined = pd.concat([rf_df, xgb_df], ignore_index=True)
combined.to_csv(f"{RESULTS_DIR}/baselines_all_folds.csv", index=False)

split_order = ['Random', 'Scaffold', 'Scaffold-Cluster']
model_order = ['Random Forest', 'XGBoost']

header = (f"{'Model':<20} {'Split':<20}"
          f" {'AUPRC':>14} {'AUROC':>14}"
          f" {'BEDROC':>14} {'F1':>12}")
print(header)
print("─" * 80)

summary_rows = []
for m in model_order:
    for s in split_order:
        sub = combined[(combined['model']==m) & (combined['split']==s)]
        if len(sub) == 0:
            continue
        n   = len(sub)
        sem = lambda x: x.std() / np.sqrt(n)
        r = {
            'Model':  m, 'Split': s,
            'AUPRC_mean':  sub['auprc'].mean(),
            'AUPRC_sem':   sem(sub['auprc']),
            'AUROC_mean':  sub['auroc'].mean(),
            'AUROC_sem':   sem(sub['auroc']),
            'BEDROC_mean': sub['bedroc'].mean(),
            'BEDROC_sem':  sem(sub['bedroc']),
            'F1_mean':     sub['f1'].mean(),
            'F1_sem':      sem(sub['f1']),
            'n_folds':     n,
        }
        summary_rows.append(r)
        print(
            f"{m:<20} {s:<20}"
            f" {r['AUPRC_mean']:.3f}±{r['AUPRC_sem']:.3f}"
            f" {r['AUROC_mean']:.3f}±{r['AUROC_sem']:.3f}"
            f" {r['BEDROC_mean']:.3f}±{r['BEDROC_sem']:.3f}"
            f" {r['F1_mean']:.3f}±{r['F1_sem']:.3f}"
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{RESULTS_DIR}/baselines_summary.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/baselines_summary.csv")

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("GENERATING FIGURES")
print("─" * 40)

colors = {
    'Random Forest': '#4CAF50',
    'XGBoost':       '#FF7043',
}

# Figure A: AUPRC bar comparison
fig, ax = plt.subplots(figsize=(10, 5))
n_models = len(model_order)
x        = np.arange(len(split_order))
width    = 0.35
offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

for i, m in enumerate(model_order):
    means, sems = [], []
    for s in split_order:
        sub = combined[(combined['model']==m) & (combined['split']==s)]
        means.append(sub['auprc'].mean() if len(sub)>0 else 0)
        n = len(sub)
        sems.append(sub['auprc'].std()/np.sqrt(n) if n>0 else 0)

    ax.bar(x + offsets[i], means, width=width,
           yerr=sems, color=colors[m], alpha=0.85,
           capsize=5, label=m, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(split_order, fontsize=12)
ax.set_ylabel('AUPRC (mean ± SEM, 8 folds)', fontsize=12)
ax.set_title('Random Forest & XGBoost — AUPRC by Splitting Strategy',
             fontsize=13, fontweight='bold')
ax.axhline(y.mean(), color='gray', linestyle=':',
           label=f'Random baseline ({y.mean():.3f})')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/auprc_rf_xgb.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURE_DIR}/auprc_rf_xgb.png")

# Figure B: PR curves (scaffold-cluster split only — most difficult)
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title('Precision-Recall Curves\n(Scaffold-Cluster Split)',
             fontsize=12, fontweight='bold')
ax.axhline(y.mean(), color='gray', linestyle=':',
           label=f'Random ({y.mean():.3f})')

for model_name, prc_dict in [('Random Forest', rf_prc),
                               ('XGBoost',       xgb_prc)]:
    curves = prc_dict.get('Scaffold-Cluster', [])
    sub    = combined[
        (combined['model']==model_name) &
        (combined['split']=='Scaffold-Cluster')
    ]
    mean_ap = sub['auprc'].mean() if len(sub)>0 else 0
    if curves:
        for rec, prec, ap in curves:
            ax.plot(rec, prec, color=colors[model_name],
                    alpha=0.3, linewidth=0.8)
        # Bold mean-representative curve (first fold)
        rec0, prec0, _ = curves[0]
        ax.plot(rec0, prec0, color=colors[model_name],
                linewidth=2.0,
                label=f"{model_name} (AP={mean_ap:.3f})")

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/pr_curves_scaffold_cluster.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURE_DIR}/pr_curves_scaffold_cluster.png")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("\nNEXT: python code/2_models_default/02_chemprop.py")
