"""
GNEprop — Train from Scratch + Final Comparison
=========================================================
Uses GNEprop's native output (final_metrics.pkl) to extract
AUPRC (test_ap) and AUROC (test_auc) directly.

Key design decisions confirmed from GNEprop source:
  - Column names: 'SMILES' and 'Activity' (from data.py)
  - Split pickle: list[(train_np, val_np, test_np)] (from data.py line 206)
  - Metrics output: final_metrics.pkl with test_ap, test_auc, test_f1
  - No pretrained encoder: trains from scratch on antimalarial data

Usage (from the top folder of this repository):
    conda activate gneprop
    export GNEPROP_DIR=/path/to/your/gneprop/clone
    python code/2_models_default/03_gneprop.py

"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import subprocess
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GNEPROP_DIR    = os.path.expanduser(
    os.environ.get("GNEPROP_DIR", "~/gneprop"))
GNEPROP_SCRIPT = os.path.join(GNEPROP_DIR, "gneprop_pyg.py")

INPUT_FILE  = "malaria_dataset_clean.csv"
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
FIGURE_DIR  = "results/diagnostics/gneprop"
LOG_BASE    = "gneprop_logs"
RANDOM_SEED = 42

# GNEprop hyperparameters 
PARAMS = dict(
    lr                 = "4.9379e-05",
    hidden_size        = "500",
    depth              = "5",
    num_readout_layers = "1",
    dropout            = "0.13",
    lr_strategy        = "warmup_cosine_step",
    aggr               = "mean",
    gpus               = "1",
    max_epochs         = "30",  
    metric             = "val_ap",
    num_workers        = "0",  # 0 avoids dataloader multiprocess hangs
    ig_baseline_ratio  = "0.3",
    adv                = "flag",
    adv_m              = "3",  
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR,  exist_ok=True)
os.makedirs(LOG_BASE,    exist_ok=True)

print("=" * 60)
print("GNEprop")
print(f"  Mode    : FULL (8 folds × 3 splits = 24 runs)")
print(f"  Encoder : From scratch (no pretrained weights)")
print(f"  Metrics : Read from GNEprop final_metrics.pkl")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────────
if not os.path.isfile(GNEPROP_SCRIPT):
    print(f"\nERROR: {GNEPROP_SCRIPT} not found")
    sys.exit(1)
if not os.path.isfile(INPUT_FILE):
    print(f"\nERROR: {INPUT_FILE} not found — run code/1_data/01_prepare_dataset.py first")
    sys.exit(1)
print("All prerequisites found ✓\n")

# ─────────────────────────────────────────────────────────────
# LOAD DATA & SPLITS
# ─────────────────────────────────────────────────────────────
df     = pd.read_csv(INPUT_FILE)
smiles = df['smiles'].tolist()
y      = df['activity'].values
print(f"Dataset: {len(df):,} compounds | Actives: {int(y.sum()):,}")

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
def write_fold_data(fd, smiles, y, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)

    train_idx = np.array(fd['train'])
    val_idx   = np.array(fd['val'])
    test_idx  = np.array(fd['test'])

    # Combined CSV — all rows: train first, then val, then test
    # GNEprop reads this as single dataset, split by pickle indices
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    pd.DataFrame({
        'SMILES':   [smiles[i] for i in all_idx],
        'Activity': y[all_idx].astype(int)
    }).to_csv(f"{fold_dir}/combined.csv", index=False)

    # Split pickle — confirmed format from data.py:
    #   split_indices = dataset.index_predetermined[seed]  # seed=0
    #   train_ix, val_ix, test_ix = split_indices          # unpack 3
    #   MoleculeSubset(dataset, train_ix)                  # numpy indexing
    n_tr = len(train_idx)
    n_va = len(val_idx)
    n_te = len(test_idx)
    split_list = [
        (
            np.array(list(range(0,         n_tr)),          dtype=np.int64),
            np.array(list(range(n_tr,      n_tr + n_va)),   dtype=np.int64),
            np.array(list(range(n_tr+n_va, n_tr+n_va+n_te)),dtype=np.int64),
        )
    ]
    with open(f"{fold_dir}/split_indices.pkl", 'wb') as f:
        pickle.dump(split_list, f)

    # Also save separate test CSV (for reference — not used by GNEprop)
    pd.DataFrame({
        'SMILES':   [smiles[i] for i in test_idx],
        'Activity': y[test_idx].astype(int)
    }).to_csv(f"{fold_dir}/test_reference.csv", index=False)

    return train_idx, val_idx, test_idx

# ─────────────────────────────────────────────────────────────
# RUN ONE GNEPROP FOLD
# ─────────────────────────────────────────────────────────────
def run_gneprop(fold_dir, log_dir):
    """
    Run GNEprop on combined.csv with index_predetermined split.
    Returns (success, log_dir_with_timestamp) or (False, None).
    """
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, GNEPROP_SCRIPT,
        '--dataset_path',             f"{fold_dir}/combined.csv",
        '--split_type',               'index_predetermined',
        '--index_predetermined_file', f"{fold_dir}/split_indices.pkl",
        '--lr',                  PARAMS['lr'],
        '--hidden_size',         PARAMS['hidden_size'],
        '--depth',               PARAMS['depth'],
        '--num_readout_layers',  PARAMS['num_readout_layers'],
        '--dropout',             PARAMS['dropout'],
        '--lr_strategy',         PARAMS['lr_strategy'],
        '--aggr',                PARAMS['aggr'],
        '--gpus',                PARAMS['gpus'],
        '--max_epochs',          PARAMS['max_epochs'],
        '--metric',              PARAMS['metric'],
        '--num_workers',         PARAMS['num_workers'],
        '--log_directory',       log_dir,
        '--ig_baseline_ratio',   PARAMS['ig_baseline_ratio'],
        '--exclude_bn_bias',
        '--adv',                 PARAMS['adv'],
        '--adv_m',               PARAMS['adv_m'],
        # NO --pretrain_path → trains from scratch
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=GNEPROP_DIR,
        timeout=None   # No timeout — let GNEprop run as long as needed
    )

    # Save logs for debugging
    with open(f"{log_dir}/stdout.txt", 'w') as f:
        f.write(result.stdout)
    with open(f"{log_dir}/stderr.txt", 'w') as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"      ERROR (rc={result.returncode})")
        print(f"      Last stderr:")
        for line in result.stderr.strip().split('\n')[-8:]:
            print(f"        {line}")
        return False, None

    return True, log_dir

def run_gneprop_safe(fold_dir, log_dir):
    """Wrapper that catches timeout and continues to next fold."""
    import subprocess
    try:
        return run_gneprop(fold_dir, log_dir)
    except subprocess.TimeoutExpired:
        print(f"      TIMEOUT — fold took too long, skipping")
        print(f"      Re-run script to retry this fold")
        return False, None

# ─────────────────────────────────────────────────────────────
# PARSE final_metrics.pkl
# Confirmed structure from inspection:
#   {'test_ap':  {'mean': X, 'sem': Y, 'std': Z},
#    'test_auc': {'mean': X, ...},
#    'test_f1':  {'mean': X, ...}, ...}
# ─────────────────────────────────────────────────────────────
def parse_metrics(log_dir):
    """
    Find and parse final_metrics.pkl from GNEprop output.
    Returns dict with auprc, auroc, f1 or None on failure.
    """
    # GNEprop saves: log_dir/<timestamp>/final_metrics.pkl
    pkl_files = glob.glob(
        f"{log_dir}/**/final_metrics.pkl", recursive=True
    )

    if not pkl_files:
        print(f"      No final_metrics.pkl found in {log_dir}")
        all_files = glob.glob(f"{log_dir}/**/*", recursive=True)
        print(f"      Files present: {all_files[:10]}")
        return None

    # Take most recent if multiple
    pkl_path = max(pkl_files, key=os.path.getmtime)
    print(f"      Metrics file: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        metrics = pickle.load(f)

    print(f"      Keys: {list(metrics.keys())}")

    
    auprc = metrics.get('test_ap',  {}).get('mean', None)
    auroc = metrics.get('test_auc', {}).get('mean', None)
    f1    = metrics.get('test_f1',  {}).get('mean', None)

    if auprc is None:
        print(f"      WARNING: test_ap not found in metrics")
        return None

    
    if auroc == 0.0:
        print(f"      NOTE: test_auc=0.0 (GNEprop AUROC edge case)")

    return dict(auprc=auprc, auroc=auroc, f1=f1)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
all_rows = []

for split_name, fold_list in splits.items():
    print(f"\n{'='*60}")
    print(f"Split: {split_name}  ({len(fold_list)} folds)")
    print(f"{'='*60}")

    split_slug = (split_name.lower()
                            .replace('-', '_')
                            .replace(' ', '_'))
    fold_rows = []

    for fd in fold_list:
        fold_idx  = fd['fold']
        test_idx  = np.array(fd['test'])
        y_test    = y[test_idx]

        print(f"\n  ── Fold {fold_idx} ──────────────────────────")
        print(f"  train={len(fd['train']):,}  "
              f"val={len(fd['val']):,}  "
              f"test={len(test_idx):,}  "
              f"| actives={int(y_test.sum())}")

        if int(y_test.sum()) == 0:
            print(f"  No actives in test — skip")
            continue

        fold_dir = f"{LOG_BASE}/{split_slug}/fold{fold_idx}"
        log_dir  = f"{fold_dir}/gneprop_output"

        # ── RESUME: skip if checkpoint exists ─────────────────
        ckpt_file = f"{fold_dir}/fold_result.csv"
        if os.path.exists(ckpt_file):
            print(f"  RESUME: fold already done — loading checkpoint")
            prev = pd.read_csv(ckpt_file).iloc[0].to_dict()
            fold_rows.append(prev)
            print(f"  AUPRC={prev.get('auprc', '?'):.4f}  "
                  f"AUROC={prev.get('auroc', '?'):.4f}  "
                  f"F1={prev.get('f1', '?'):.4f}")
            continue
        # ──────────────────────────────────────────────────────

        # Write fold data
        write_fold_data(fd, smiles, y, fold_dir)
        print(f"  Fold data written → combined.csv + split_indices.pkl")
        print(f"  Running GNEprop (from scratch, no pretrained encoder)...")

        success, out_dir = run_gneprop_safe(fold_dir, log_dir)

        if not success:
            print(f"  GNEprop FAILED — check {log_dir}/stderr.txt")
            continue

        print(f"  GNEprop finished ✓")

        # Parse metrics from final_metrics.pkl
        m_raw = parse_metrics(log_dir)

        if m_raw is None:
            print(f"  Could not parse metrics — skip")
            continue

        # Build result row
        m = dict(
            fold           = fold_idx,
            split          = split_name,
            model          = 'GNEprop (Scratch)',
            auprc          = m_raw['auprc'],
            auroc          = m_raw['auroc'] if m_raw['auroc'] is not None else np.nan,
            f1             = m_raw['f1']    if m_raw['f1']    is not None else np.nan,
            bedroc         = np.nan,   # not computed by GNEprop natively
            n_test         = len(test_idx),
            n_test_actives = int(y_test.sum()),
        )
        fold_rows.append(m)

        # Save checkpoint immediately
        pd.DataFrame([m]).to_csv(ckpt_file, index=False)
        print(f"  Checkpoint saved: {ckpt_file}")

        print(f"\n  ✓ Fold {fold_idx} Results:")
        print(f"    AUPRC  = {m['auprc']:.4f}")
        print(f"    AUROC  = {m['auroc']:.4f}")
        print(f"    F1     = {m['f1']:.4f}")
        print(f"    BEDROC = N/A (not computed by GNEprop natively)")

    # Split summary
    if fold_rows:
        ap = [r['auprc'] for r in fold_rows]
        ro = [r['auroc'] for r in fold_rows
              if not np.isnan(r.get('auroc', np.nan))]
        print(f"\n  ── {split_name} Summary ({len(fold_rows)} folds) ──")
        print(f"    AUPRC = {np.mean(ap):.4f} ± {np.std(ap):.4f}")
        if ro:
            print(f"    AUROC = {np.mean(ro):.4f} ± {np.std(ro):.4f}")

    all_rows.extend(fold_rows)

# ─────────────────────────────────────────────────────────────
# SAVE GNEPROP RESULTS
# ─────────────────────────────────────────────────────────────
if not all_rows:
    print("\nNo GNEprop results collected.")
    print("Check gneprop_logs/*/fold*/gneprop_output/stderr.txt")
    sys.exit(1)

gneprop_df = pd.DataFrame(all_rows)
gneprop_df.to_csv(
    f"{RESULTS_DIR}/gneprop_scratch_results_per_fold.csv", index=False
)
print(f"\nSaved: {RESULTS_DIR}/gneprop_scratch_results_per_fold.csv")

# ─────────────────────────────────────────────────────────────
# COMPILE FINAL TABLE — ALL MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COMPARISON — ALL MODELS")
print("=" * 60)

result_files = {
    'Random Forest':     f"{RESULTS_DIR}/rf_results_per_fold.csv",
    'XGBoost':           f"{RESULTS_DIR}/xgb_results_per_fold.csv",
    'ChemProp':          f"{RESULTS_DIR}/chemprop_results_per_fold.csv",
    'GNEprop (Scratch)': f"{RESULTS_DIR}/gneprop_scratch_results_per_fold.csv",
}

frames = []
for label, path in result_files.items():
    if os.path.exists(path):
        tmp = pd.read_csv(path)
        tmp['model'] = label
        frames.append(tmp)
        print(f"  Loaded: {label}  ({len(tmp)} rows)")
    else:
        print(f"  Missing: {label}")

if not frames:
    print("No results to compile.")
    sys.exit(0)

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(f"{RESULTS_DIR}/all_models_all_folds.csv", index=False)

# ── PRINT SUMMARY TABLE ───────────────────────────────────────
split_order = ['Random', 'Scaffold', 'Scaffold-Cluster']
model_order = list(result_files.keys())

print("\n" + "─" * 88)
print("PAPER TABLE 2: Performance Summary (mean ± SEM, 8-fold CV)")
print("Note: BEDROC = N/A for GNEprop (not computed natively)")
print("─" * 88)
print(f"{'Model':<24} {'Split':<20}"
      f" {'AUPRC':>14} {'AUROC':>14} {'F1':>12}")
print("─" * 88)

summary_rows = []
for m in model_order:
    for s in split_order:
        sub = combined[
            (combined['model'] == m) & (combined['split'] == s)
        ]
        if len(sub) == 0:
            continue
        n   = len(sub)
        sem = lambda x: x.dropna().std() / np.sqrt(x.dropna().count())

        r = dict(
            Model       = m,
            Split       = s,
            AUPRC_mean  = sub['auprc'].mean(),
            AUPRC_sem   = sem(sub['auprc']),
            AUROC_mean  = sub['auroc'].mean(),
            AUROC_sem   = sem(sub['auroc']),
            F1_mean     = sub['f1'].mean(),
            F1_sem      = sem(sub['f1']),
            BEDROC_mean = sub['bedroc'].mean() if 'bedroc' in sub else np.nan,
            BEDROC_sem  = sem(sub['bedroc'])   if 'bedroc' in sub else np.nan,
            n_folds     = n,
        )
        summary_rows.append(r)
        print(
            f"{m:<24} {s:<20}"
            f" {r['AUPRC_mean']:.3f}±{r['AUPRC_sem']:.3f}"
            f" {r['AUROC_mean']:.3f}±{r['AUROC_sem']:.3f}"
            f" {r['F1_mean']:.3f}±{r['F1_sem']:.3f}"
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{RESULTS_DIR}/FINAL_SUMMARY_TABLE.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/FINAL_SUMMARY_TABLE.csv")

# ─────────────────────────────────────────────────────────────
# FINAL COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────
model_colors = {
    'Random Forest':     '#4CAF50',
    'XGBoost':           '#FF7043',
    'ChemProp':          '#1565C0',
    'GNEprop (Scratch)': '#7B1FA2',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Model Performance — Antimalarial Activity Prediction\n'
    'AUPRC (mean ± SEM, 8-fold CV)',
    fontsize=14, fontweight='bold'
)

x = np.arange(len(model_order))

for ax, split_name in zip(axes, split_order):
    means, sems, colors = [], [], []
    for m in model_order:
        sub = combined[
            (combined['model'] == m) & (combined['split'] == split_name)
        ]
        mn = sub['auprc'].mean() if len(sub) > 0 else 0.0
        se = (sub['auprc'].std() / np.sqrt(len(sub))
              if len(sub) > 1 else 0.0)
        means.append(mn)
        sems.append(se)
        colors.append(model_colors.get(m, '#888888'))

    bars = ax.bar(x, means, width=0.55, yerr=sems,
                  color=colors, alpha=0.85, capsize=5,
                  edgecolor='white', linewidth=1.2)

    for bar, mn, se in zip(bars, means, sems):
        if mn > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                mn + se + 0.005,
                f'{mn:.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    ax.axhline(y.mean(), color='gray', linestyle=':',
               linewidth=1.5, alpha=0.8,
               label=f'Random ({y.mean():.3f})')
    ax.set_title(f'{split_name} Split',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace(' ', '\n').replace('(', '\n(')
         for m in model_order],
        fontsize=8
    )
    ax.set_ylim([0, min(1.0, (max(means) if max(means) > 0 else 0.3)*1.35)])
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out_fig = f"{FIGURE_DIR}/final_model_comparison.png"
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_fig}")

print("\n" + "=" * 60)
print("COMPLETE")
print(f"  Final table : {RESULTS_DIR}/FINAL_SUMMARY_TABLE.csv")
print(f"  Main figure : {FIGURE_DIR}/final_model_comparison.png")
print("=" * 60)
