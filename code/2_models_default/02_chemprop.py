"""
ChemProp v2.2.2 
==========================================

Run from:
    conda activate chemprop2_env
    cd /path/to/this/repository
    python code/2_models_default/02_chemprop.py

Prerequisites:
    - malaria_dataset_clean.csv       (from code/1_data/01_prepare_dataset.py)
    - splits/random_splits.pkl        (from code/1_data/02_make_splits.py)
    - splits/scaffold_splits.pkl      (from code/1_data/02_make_splits.py)
    - splits/scaffold_cluster_splits.pkl (from code/1_data/02_make_splits.py)
    - pip install chemprop matplotlib scikit-learn (in chemprop2_env)

"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = "malaria_dataset_clean.csv"
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
FIGURE_DIR  = "results/diagnostics/chemprop"
RANDOM_SEED = 42

# ChemProp hyperparameters (matching paper defaults)
N_EPOCHS    = 30
HIDDEN_SIZE = 300
DEPTH       = 3        # message passing steps
DROPOUT     = 0.0
FFN_LAYERS  = 2
BATCH_SIZE  = 50

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# VERIFY CHEMPROP v2 INSTALLATION
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("CHEMPROP v2 (D-MPNN)")
print("=" * 60)

try:
    import chemprop
    version = chemprop.__version__
    major   = int(version.split('.')[0])
    print(f"ChemProp version : {version}")
    if major < 2:
        print("ERROR: This script requires ChemProp v2.x")
        print("You have v1.x installed.")
        print("Run: pip install chemprop --upgrade")
        sys.exit(1)
    print("ChemProp v2 confirmed ✓")
except ImportError:
    print("ERROR: ChemProp not installed in this environment.")
    print("Run: pip install chemprop")
    sys.exit(1)

# Import ChemProp v2 modules
from chemprop import data as cpdata
from chemprop import nn   as cpnn
from chemprop import models as cpmodels
from chemprop import featurizers

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# ─────────────────────────────────────────────────────────────
# GPU / CPU SETUP
# ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICES     = 1
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    ACCELERATOR = "cpu"
    DEVICES     = 1
    print("No GPU detected — running on CPU")

# ─────────────────────────────────────────────────────────────
# LOAD DATA & SPLITS
# ─────────────────────────────────────────────────────────────
df     = pd.read_csv(INPUT_FILE)
smiles = df['smiles'].tolist()
y      = df['activity'].values.astype(float)
print(f"\nDataset : {len(df):,} compounds | Actives: {int(y.sum()):,}")

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
    n  = len(y_true)
    na = int(y_true.sum())
    if na == 0 or na == n:
        return 0.5
    order  = np.argsort(-y_score)
    ys     = y_true[order]
    ra     = na / n
    ri_sum = sum(np.exp(-alpha*(i+1)/n)
                 for i, v in enumerate(ys) if v == 1)
    rie     = ri_sum / (ra*(1-np.exp(-alpha))/(np.exp(alpha/n)-1))
    rie_max = (1-np.exp(-alpha*ra)) / (ra*(1-np.exp(-alpha)))
    rie_min = (1-np.exp( alpha*ra)) / (ra*(1-np.exp(-alpha)))
    return (rie - rie_min) / (rie_max - rie_min)

def evaluate(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if int(y_true.sum()) == 0:
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
# BUILD CHEMPROP v2 MODEL
# ─────────────────────────────────────────────────────────────
def build_model():
    """
    Build ChemProp v2 MPNN for binary classification.
    Architecture matches Stokes et al. 2020 defaults.
    Explicitly adds BinaryAUPRC metric so val/prc is logged.
    """
    mp  = cpnn.BondMessagePassing(
        d_h      = HIDDEN_SIZE,
        depth    = DEPTH,
        dropout  = DROPOUT,
    )
    agg = cpnn.MeanAggregation()
    ffn = cpnn.BinaryClassificationFFN(
        input_dim  = HIDDEN_SIZE,
        n_layers   = FFN_LAYERS,
        dropout    = DROPOUT,
    )
    # Explicitly pass BinaryAUPRC + BinaryAUROC metrics so both
    # val/prc and val/roc are logged during validation.
    # First metric in list is used for early stopping/checkpointing.
    metric_list = [
        cpnn.metrics.BinaryAUPRC(),   # val/prc → primary for checkpointing
        cpnn.metrics.BinaryAUROC(),   # val/roc → secondary
    ]
    model = cpmodels.MPNN(
        message_passing = mp,
        agg             = agg,
        predictor       = ffn,
        metrics         = metric_list,
    )
    return model

# ─────────────────────────────────────────────────────────────
# TRAIN ONE FOLD (ChemProp v2)
# ─────────────────────────────────────────────────────────────
def train_fold(train_smi, train_lbl,
               val_smi,   val_lbl,
               test_smi,  fold_idx, split_slug):
    """
    Full ChemProp v2 training and prediction for one fold.
    Returns predicted probabilities for test_smi.
    """
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # ── Build datasets ────────────────────────────────────────
    def make_dataset(smi_list, lbl_list=None):
        if lbl_list is not None:
            points = [
                cpdata.MoleculeDatapoint.from_smi(
                    s, np.array([float(l)])
                )
                for s, l in zip(smi_list, lbl_list)
            ]
        else:
            # Test set — dummy label
            points = [
                cpdata.MoleculeDatapoint.from_smi(
                    s, np.array([0.0])
                )
                for s in smi_list
            ]
        return cpdata.MoleculeDataset(points, featurizer=feat)

    train_ds = make_dataset(train_smi, train_lbl)
    val_ds   = make_dataset(val_smi,   val_lbl)
    test_ds  = make_dataset(test_smi)

    train_loader = cpdata.build_dataloader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0
    )
    val_loader   = cpdata.build_dataloader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )
    test_loader  = cpdata.build_dataloader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )

    # ── Build model ───────────────────────────────────────────
    torch.manual_seed(RANDOM_SEED + fold_idx)
    model = build_model()

    # ── Trainer ───────────────────────────────────────────────
    ckpt_dir = f"chemprop_tmp/{split_slug}/fold{fold_idx}"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = ckpt_dir,
        monitor   = "val/prc",          # AUPRC — our primary metric
        mode      = "max",
        save_top_k= 1,
        filename  = "best",
        verbose   = False,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val/prc",           # AUPRC — consistent with primary metric
        patience = 10,
        mode     = "max",
        verbose  = False,
    )

    trainer = pl.Trainer(
        max_epochs         = N_EPOCHS,
        accelerator        = ACCELERATOR,
        devices            = DEVICES,
        logger             = False,
        enable_progress_bar= False,
        enable_model_summary= False,
        callbacks          = [checkpoint_cb, early_stop_cb],
    )

    # ── Train ─────────────────────────────────────────────────
    trainer.fit(model, train_loader, val_loader)

    # ── Predict on test set ───────────────────────────────────
    # Load best checkpoint
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        model = cpmodels.MPNN.load_from_checkpoint(best_ckpt)

    model.eval()
    pred_trainer = pl.Trainer(
        accelerator         = ACCELERATOR,
        devices             = DEVICES,
        logger              = False,
        enable_progress_bar = False,
        enable_model_summary= False,
    )

    # weights_only=False needed for lightning >= 2.6
    try:
        raw_preds = pred_trainer.predict(
            model, test_loader, weights_only=False
        )
    except TypeError:
        raw_preds = pred_trainer.predict(model, test_loader)

    # raw_preds is a list of tensors, shape [batch, 1]
    y_prob = torch.cat(raw_preds, dim=0).squeeze(-1).numpy()
    y_prob = np.clip(y_prob, 0.0, 1.0)

    return y_prob

# ─────────────────────────────────────────────────────────────
# MAIN LOOP — ALL SPLITS, ALL FOLDS
# ─────────────────────────────────────────────────────────────
all_rows  = []
pr_curves = {}

for split_name, fold_list in splits.items():
    print(f"\n{'='*60}")
    print(f"[ChemProp] Split: {split_name}")
    print(f"{'='*60}")

    split_slug = split_name.lower().replace('-','_').replace(' ','_')
    fold_rows  = []
    pr_this    = []

    for fd in fold_list:
        fold_idx  = fd['fold']
        train_idx = np.array(fd['train'])
        val_idx   = np.array(fd['val'])
        test_idx  = np.array(fd['test'])

        y_test    = y[test_idx]

        if int(y_test.sum()) == 0:
            print(f"  Fold {fold_idx}: no actives — skip")
            continue

        print(f"\n  Fold {fold_idx}: "
              f"train={len(train_idx):,} "
              f"val={len(val_idx):,} "
              f"test={len(test_idx):,} "
              f"| actives={int(y_test.sum())}")

        # ── RESUME: skip if checkpoint exists ─────────────────
        ckpt_file = (f"chemprop_tmp/{split_slug}/"
                     f"fold{fold_idx}/fold_result.csv")
        if os.path.exists(ckpt_file):
            print(f"  RESUME: loading checkpoint")
            prev = pd.read_csv(ckpt_file).iloc[0].to_dict()
            fold_rows.append(prev)
            continue
        # ──────────────────────────────────────────────────────

        train_smi = [smiles[i] for i in train_idx]
        val_smi   = [smiles[i] for i in val_idx]
        test_smi  = [smiles[i] for i in test_idx]
        train_lbl = y[train_idx]
        val_lbl   = y[val_idx]

        try:
            y_prob = train_fold(
                train_smi, train_lbl,
                val_smi,   val_lbl,
                test_smi,  fold_idx, split_slug
            )
        except Exception as e:
            print(f"  ERROR in fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if len(y_prob) != len(test_idx):
            print(f"  Length mismatch {len(y_prob)} vs "
                  f"{len(test_idx)} — skipping")
            continue

        m = evaluate(y_test, y_prob)
        m.update(
            fold           = fold_idx,
            split          = split_name,
            model          = 'ChemProp',
            n_test         = len(test_idx),
            n_test_actives = int(y_test.sum()),
        )
        fold_rows.append(m)

        # Save checkpoint
        os.makedirs(
            f"chemprop_tmp/{split_slug}/fold{fold_idx}",
            exist_ok=True
        )
        pd.DataFrame([m]).to_csv(ckpt_file, index=False)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_this.append((rec, prec,
                        average_precision_score(y_test, y_prob)))

        print(f"  ✓ AUPRC={m['auprc']:.4f}  "
              f"AUROC={m['auroc']:.4f}  "
              f"BEDROC={m['bedroc']:.4f}  "
              f"F1={m['f1']:.4f}")

    if fold_rows:
        ap = [r['auprc'] for r in fold_rows]
        print(f"\n  {split_name} → "
              f"AUPRC = {np.mean(ap):.4f} ± {np.std(ap):.4f}")

    all_rows.extend(fold_rows)
    pr_curves[split_name] = pr_this

# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
if not all_rows:
    print("\nNo results collected. Check errors above.")
    sys.exit(1)

results_df = pd.DataFrame(all_rows)
results_df.to_csv(
    f"{RESULTS_DIR}/chemprop_results_per_fold.csv", index=False
)
print(f"\nSaved: {RESULTS_DIR}/chemprop_results_per_fold.csv")

# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("ChemProp Summary (Paper Table 2)")
print("─"*70)
print(f"{'Split':<22} {'AUPRC':>14} {'AUROC':>14} "
      f"{'BEDROC':>14} {'F1':>12}")
print("─"*70)

summary_rows = []
for s in ['Random', 'Scaffold', 'Scaffold-Cluster']:
    sub = results_df[results_df['split'] == s]
    if len(sub) == 0:
        continue
    n   = len(sub)
    sem = lambda x: x.std() / np.sqrt(n)
    r   = dict(
        Model='ChemProp', Split=s,
        AUPRC_mean  = sub['auprc'].mean(),
        AUPRC_sem   = sem(sub['auprc']),
        AUROC_mean  = sub['auroc'].mean(),
        AUROC_sem   = sem(sub['auroc']),
        BEDROC_mean = sub['bedroc'].mean(),
        BEDROC_sem  = sem(sub['bedroc']),
        F1_mean     = sub['f1'].mean(),
        F1_sem      = sem(sub['f1']),
        n_folds     = n,
    )
    summary_rows.append(r)
    print(
        f"{s:<22}"
        f" {r['AUPRC_mean']:.3f}±{r['AUPRC_sem']:.3f}"
        f" {r['AUROC_mean']:.3f}±{r['AUROC_sem']:.3f}"
        f" {r['BEDROC_mean']:.3f}±{r['BEDROC_sem']:.3f}"
        f" {r['F1_mean']:.3f}±{r['F1_sem']:.3f}"
    )

pd.DataFrame(summary_rows).to_csv(
    f"{RESULTS_DIR}/chemprop_summary.csv", index=False
)
print(f"\nSaved: {RESULTS_DIR}/chemprop_summary.csv")

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
# Figure 1: PR curves (scaffold-cluster)
target_split = 'Scaffold-Cluster'
curves = pr_curves.get(target_split, [])

if curves:
    fig, ax = plt.subplots(figsize=(6, 5))
    baseline = y.mean()
    ax.axhline(baseline, color='gray', linestyle=':',
               linewidth=1.5, label=f'Random ({baseline:.3f})')

    sub    = results_df[results_df['split'] == target_split]
    mean_ap = sub['auprc'].mean() if len(sub) > 0 else 0

    for rec, prec, _ in curves:
        ax.plot(rec, prec, color='#1565C0', alpha=0.25, linewidth=0.8)
    rec0, prec0, _ = curves[0]
    ax.plot(rec0, prec0, color='#1565C0', linewidth=2,
            label=f"ChemProp (AP={mean_ap:.3f})")

    ax.set_title('ChemProp — Precision-Recall Curves\n'
                 '(Scaffold-Cluster Split)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/pr_curves_chemprop.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURE_DIR}/pr_curves_chemprop.png")

# Figure 2: AUPRC bar chart across splits
split_order = ['Random', 'Scaffold', 'Scaffold-Cluster']
means, sems = [], []
for s in split_order:
    sub = results_df[results_df['split'] == s]
    if len(sub) > 0:
        means.append(sub['auprc'].mean())
        sems.append(sub['auprc'].sem())
    else:
        means.append(0)
        sems.append(0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(split_order, means, yerr=sems,
       color='#1565C0', alpha=0.85, capsize=5,
       width=0.5, edgecolor='white')
ax.axhline(y.mean(), color='gray', linestyle=':',
           linewidth=1.5,
           label=f'Random baseline ({y.mean():.3f})')

for i, (m, s) in enumerate(zip(means, sems)):
    if m > 0:
        ax.text(i, m + s + 0.005, f'{m:.3f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

ax.set_ylabel('AUPRC (mean ± SEM, 8 folds)', fontsize=12)
ax.set_title('ChemProp — AUPRC by Splitting Strategy',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim([0, min(1.0, max(means)*1.3)])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/auprc_chemprop.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURE_DIR}/auprc_chemprop.png")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print("\nNEXT:")
print("  conda activate gneprop")
print("  python code/2_models_default/03_gneprop.py")
