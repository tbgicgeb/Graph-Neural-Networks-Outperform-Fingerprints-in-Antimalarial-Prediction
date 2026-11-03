"""
Hyperparameter Tuning — XGBoost Only
===============================================
Bayesian TPE optimisation (Optuna) with nested cross-validation.


Usage:
    cd /path/to/this/repository
    pip install optuna
    python code/3_models_tuned/02_xgboost.py

"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score)
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Optuna version : {optuna.__version__}")
except ImportError:
    print("ERROR: optuna not installed")
    print("Run: pip install optuna")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_FILE    = "malaria_dataset_clean.csv"
SPLITS_DIR    = "splits"
RESULTS_DIR   = "results"
FIGURE_DIR    = "results/diagnostics/tuning"

N_TRIALS      = 50    # Optuna Bayesian TPE trials per outer fold
N_INNER_FOLDS = 3     # Inner CV folds for hyperparameter selection
N_JOBS        = -1    # Use all CPU cores
RANDOM_SEED   = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR,  exist_ok=True)

print("=" * 60)
print("XGBoost — Hyperparameter Tuning")
print(f"  Method         : Bayesian TPE (Optuna)")
print(f"  Outer folds    : 8 (Scaffold-Cluster)")
print(f"  Inner folds    : {N_INNER_FOLDS}-fold CV on training data")
print(f"  Trials         : {N_TRIALS} per outer fold")
print(f"  Primary metric : AUPRC")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df     = pd.read_csv(INPUT_FILE)
smiles = df['smiles'].tolist()
y      = df['activity'].values

print(f"\nDataset  : {len(df):,} compounds")
print(f"Actives  : {int(y.sum()):,} ({y.mean()*100:.1f}%)")
print(f"Imbalance: 1:{(1-y.mean())/y.mean():.1f}")

# Imbalance ratio for scale_pos_weight
SPW = float((y == 0).sum() / (y == 1).sum())
print(f"scale_pos_weight reference value: {SPW:.2f}")

# ─────────────────────────────────────────────────────────────
# MORGAN FINGERPRINTS
# ─────────────────────────────────────────────────────────────
print("\nComputing Morgan fingerprints (radius=2, 2048 bits)...")
fps = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fps.append(np.array(fp, dtype=np.float32))
X = np.array(fps)
print(f"Feature matrix: {X.shape}")

# ─────────────────────────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────────────────────────
splits = {}
for name, fname in [
    ('Random',           'random_splits.pkl'),
    ('Scaffold',         'scaffold_splits.pkl'),
    ('Scaffold-Cluster', 'scaffold_cluster_splits.pkl'),
]:
    with open(f"{SPLITS_DIR}/{fname}", 'rb') as f:
        splits[name] = pickle.load(f)
    print(f"Loaded {name}: {len(splits[name])} folds")

sc_splits = splits['Scaffold-Cluster']

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def bedroc(y_true, y_score, alpha=20.0):
    n, na = len(y_true), int(y_true.sum())
    if na == 0 or na == n:
        return 0.5
    order   = np.argsort(-y_score)
    ys      = y_true[order]
    ra      = na / n
    ri_sum  = sum(np.exp(-alpha*(i+1)/n)
                  for i, v in enumerate(ys) if v == 1)
    rie     = ri_sum / (ra*(1-np.exp(-alpha))/(np.exp(alpha/n)-1))
    rie_max = (1-np.exp(-alpha*ra)) / (ra*(1-np.exp(-alpha)))
    rie_min = (1-np.exp( alpha*ra)) / (ra*(1-np.exp(-alpha)))
    return (rie - rie_min) / (rie_max - rie_min)

def evaluate(y_true, y_prob):
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    bd    = bedroc(y_true, y_prob)
    yp    = (y_prob >= 0.5).astype(int)
    if yp.sum() == 0:
        thr = np.percentile(y_prob, 100*(1 - y_true.mean()))
        yp  = (y_prob >= thr).astype(int)
    f1 = f1_score(y_true, yp, zero_division=0)
    return dict(auprc=auprc, auroc=auroc, bedroc=bd, f1=f1)

# ─────────────────────────────────────────────────────────────
# INNER CV — used by Optuna objective
# ─────────────────────────────────────────────────────────────
def inner_cv_auprc(params, X_train, y_train):
    """
    3-fold stratified CV on training data only.
    Returns mean AUPRC — used as Optuna objective.
    """
    model = xgb.XGBClassifier(
        n_estimators     = params['n_estimators'],
        max_depth        = params['max_depth'],
        learning_rate    = params['learning_rate'],
        subsample        = params['subsample'],
        colsample_bytree = params['colsample_bytree'],
        min_child_weight = params['min_child_weight'],
        gamma            = params['gamma'],
        reg_alpha        = params['reg_alpha'],
        reg_lambda       = params['reg_lambda'],
        scale_pos_weight = params['scale_pos_weight'],
        eval_metric      = 'aucpr',
        n_jobs           = N_JOBS,
        random_state     = RANDOM_SEED,
        verbosity        = 0,
    )

    skf    = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                             random_state=RANDOM_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[va_idx], y_train[va_idx])],
                  verbose=False)
        prob = model.predict_proba(X_train[va_idx])[:, 1]
        scores.append(average_precision_score(y_train[va_idx], prob))

    return float(np.mean(scores))

# ─────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE — XGBOOST
# ─────────────────────────────────────────────────────────────
def make_objective(X_train, y_train):
    """
    Returns an Optuna objective function that tunes XGBoost
    on the given training data using inner CV.
    """
    def objective(trial):
        params = {
            # Tree structure
            'n_estimators': trial.suggest_int(
                'n_estimators', 100, 1500, step=100),
            'max_depth': trial.suggest_int(
                'max_depth', 3, 10),
            'min_child_weight': trial.suggest_int(
                'min_child_weight', 1, 20),
            'gamma': trial.suggest_float(
                'gamma', 0.0, 5.0),
            # Learning rate
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.005, 0.3, log=True),
            # Subsampling — key for regularisation
            'subsample': trial.suggest_float(
                'subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.5, 1.0),
            # L1/L2 regularisation
            'reg_alpha': trial.suggest_float(
                'reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 0.5, 10.0),
            # Class imbalance handling
            # Search around the observed ratio (5.1) ± 50%
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight', SPW * 0.3, SPW * 2.0),
        }
        return inner_cv_auprc(params, X_train, y_train)
    return objective

def build_xgb(params):
    """Build final XGBoost model from best params dict."""
    return xgb.XGBClassifier(
        n_estimators     = params['n_estimators'],
        max_depth        = params['max_depth'],
        learning_rate    = params['learning_rate'],
        subsample        = params['subsample'],
        colsample_bytree = params['colsample_bytree'],
        min_child_weight = params['min_child_weight'],
        gamma            = params['gamma'],
        reg_alpha        = params['reg_alpha'],
        reg_lambda       = params['reg_lambda'],
        scale_pos_weight = params['scale_pos_weight'],
        eval_metric      = 'aucpr',
        n_jobs           = N_JOBS,
        random_state     = RANDOM_SEED,
        verbosity        = 0,
    )

# ─────────────────────────────────────────────────────────────
# PHASE 1: NESTED CV TUNING ON SCAFFOLD-CLUSTER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 1: Nested CV Tuning — Scaffold-Cluster Split")
print("=" * 60)

best_params_per_fold = {}
tuning_history       = {}
sc_fold_results      = []

for fd in sc_splits:
    fold_idx  = fd['fold']
    train_idx = np.array(fd['train'])
    val_idx   = np.array(fd['val'])
    test_idx  = np.array(fd['test'])

    trainval_idx = np.concatenate([train_idx, val_idx])
    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]
    X_te = X[test_idx]
    y_te = y[test_idx]

    print(f"\n  ── Outer Fold {fold_idx} ──────────────────────────────")
    print(f"  train+val = {len(trainval_idx):,}  "
          f"test = {len(test_idx):,}  "
          f"| test_actives = {int(y_te.sum())}")
    print(f"  Running {N_TRIALS} Optuna trials "
          f"({N_INNER_FOLDS}-fold inner CV)...")

    # ── Optuna study ──────────────────────────────────────────
    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(
            seed             = RANDOM_SEED + fold_idx,
            n_startup_trials = 10,
        ),
    )
    study.optimize(
        make_objective(X_tv, y_tv),
        n_trials         = N_TRIALS,
        show_progress_bar= False,
    )

    best_p     = study.best_params
    best_inner = study.best_value
    best_params_per_fold[fold_idx] = best_p

    tuning_history[fold_idx] = [
        (t.number, t.value)
        for t in study.trials if t.value is not None
    ]

    print(f"  Best inner AUPRC : {best_inner:.4f}")
    print(f"  Best params:")
    for k, v in best_p.items():
        print(f"    {k:<22} = {v:.4f}"
              if isinstance(v, float) else
              f"    {k:<22} = {v}")

    # ── Retrain on full train+val ──────────────────────────────
    final_model = build_xgb(best_p)
    final_model.fit(X_tv, y_tv, verbose=False)

    # ── Evaluate on held-out test ──────────────────────────────
    y_prob = final_model.predict_proba(X_te)[:, 1]
    m = evaluate(y_te, y_prob)
    m.update(
        fold           = fold_idx,
        split          = 'Scaffold-Cluster',
        model          = 'XGBoost (Tuned)',
        n_test         = len(test_idx),
        n_test_actives = int(y_te.sum()),
        inner_auprc    = best_inner,
    )
    sc_fold_results.append(m)

    print(f"\n  ✓ Test Results:")
    print(f"    AUPRC  = {m['auprc']:.4f}  "
          f"(inner CV was {best_inner:.4f})")
    print(f"    AUROC  = {m['auroc']:.4f}")
    print(f"    BEDROC = {m['bedroc']:.4f}")
    print(f"    F1     = {m['f1']:.4f}")

# Scaffold-Cluster summary
ap  = [r['auprc'] for r in sc_fold_results]
sem = np.std(ap) / np.sqrt(len(ap))
print(f"\n  ── Scaffold-Cluster Summary ({len(sc_fold_results)} folds) ──")
print(f"    XGBoost (Tuned) AUPRC = {np.mean(ap):.4f} ± {sem:.4f}")

# Compare with default
print(f"\n  Comparison with default XGBoost:")
try:
    default_df = pd.read_csv(f"{RESULTS_DIR}/xgb_results_per_fold.csv")
    def_sc   = default_df[default_df['split'] == 'Scaffold-Cluster']['auprc']
    def_mean = def_sc.mean()
    def_sem  = def_sc.std() / np.sqrt(len(def_sc))
    delta    = np.mean(ap) - def_mean
    print(f"    Default : {def_mean:.4f} ± {def_sem:.4f}")
    print(f"    Tuned   : {np.mean(ap):.4f} ± {sem:.4f}")
    print(f"    Delta   : {delta:+.4f} ({delta/def_mean*100:+.1f}%)")
except FileNotFoundError:
    print("    (default results not found — run code/2_models_default/01_random_forest_and_xgboost.py first)")

# ─────────────────────────────────────────────────────────────
# PHASE 2: EVALUATE ON ALL THREE SPLITS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Evaluating Tuned XGBoost on All Three Splits")
print("=" * 60)

all_rows = []

for split_name, fold_list in splits.items():
    print(f"\n  Split: {split_name}")

    for fd in fold_list:
        fold_idx  = fd['fold']
        train_idx = np.array(fd['train'])
        val_idx   = np.array(fd['val'])
        test_idx  = np.array(fd['test'])

        params = best_params_per_fold[fold_idx]

        trainval_idx = np.concatenate([train_idx, val_idx])
        X_tv = X[trainval_idx]
        y_tv = y[trainval_idx]
        X_te = X[test_idx]
        y_te = y[test_idx]

        # Reuse SC results already computed
        if split_name == 'Scaffold-Cluster':
            row = next(r for r in sc_fold_results
                       if r['fold'] == fold_idx)
            all_rows.append(row)
            continue

        model = build_xgb(params)
        model.fit(X_tv, y_tv, verbose=False)
        y_prob = model.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, y_prob)
        m.update(
            fold           = fold_idx,
            split          = split_name,
            model          = 'XGBoost (Tuned)',
            n_test         = len(test_idx),
            n_test_actives = int(y_te.sum()),
        )
        all_rows.append(m)
        print(f"    Fold {fold_idx}: AUPRC={m['auprc']:.4f}  "
              f"AUROC={m['auroc']:.4f}  "
              f"BEDROC={m['bedroc']:.4f}  "
              f"F1={m['f1']:.4f}")

# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)
results_df.to_csv(
    f"{RESULTS_DIR}/tuned_xgb_results_per_fold.csv", index=False
)
print(f"\nSaved: {RESULTS_DIR}/tuned_xgb_results_per_fold.csv")

with open(f"{RESULTS_DIR}/tuning_xgb_best_params.json", 'w') as f:
    json.dump(
        {str(k): v for k, v in best_params_per_fold.items()},
        f, indent=2, default=str
    )
print(f"Saved: {RESULTS_DIR}/tuning_xgb_best_params.json")

# ─────────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("TUNED XGBoost — SUMMARY TABLE (mean ± SEM, 8-fold CV)")
print("─" * 70)
print(f"{'Split':<22} {'AUPRC':>14} {'AUROC':>14} "
      f"{'BEDROC':>12} {'F1':>12}")
print("─" * 70)

for split_name in ['Random', 'Scaffold', 'Scaffold-Cluster']:
    sub = results_df[results_df['split'] == split_name]
    n   = len(sub)
    if n == 0:
        continue
    sem = lambda col: sub[col].std() / np.sqrt(n)
    print(
        f"{split_name:<22}"
        f" {sub['auprc'].mean():.3f}±{sem('auprc'):.3f}"
        f" {sub['auroc'].mean():.3f}±{sem('auroc'):.3f}"
        f" {sub['bedroc'].mean():.3f}±{sem('bedroc'):.3f}"
        f" {sub['f1'].mean():.3f}±{sem('f1'):.3f}"
    )

# ─────────────────────────────────────────────────────────────
# COMPARE DEFAULT vs TUNED
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("DEFAULT vs TUNED — AUPRC COMPARISON")
print("─" * 70)

try:
    default_df = pd.read_csv(f"{RESULTS_DIR}/xgb_results_per_fold.csv")
    for split_name in ['Random', 'Scaffold', 'Scaffold-Cluster']:
        def_sub  = default_df[default_df['split'] == split_name]['auprc']
        tun_sub  = results_df[results_df['split'] == split_name]['auprc']
        def_mean = def_sub.mean()
        tun_mean = tun_sub.mean()
        delta    = tun_mean - def_mean
        def_sem  = def_sub.std() / np.sqrt(len(def_sub))
        tun_sem  = tun_sub.std() / np.sqrt(len(tun_sub))
        sign     = '+' if delta >= 0 else ''
        print(f"{split_name:<22}"
              f"  Default: {def_mean:.3f}±{def_sem:.3f}"
              f"  Tuned: {tun_mean:.3f}±{tun_sem:.3f}"
              f"  Δ = {sign}{delta:.3f} ({sign}{delta/def_mean*100:.1f}%)")
except FileNotFoundError:
    print("  Default XGBoost results not found — run code/2_models_default/01_random_forest_and_xgboost.py first")

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Trial History — all 8 folds
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
fig.suptitle('XGBoost Hyperparameter Tuning — Trial History\n'
             '(Scaffold-Cluster, 8 Outer Folds)',
             fontsize=13, fontweight='bold')

for fold_idx, ax in enumerate(axes):
    if fold_idx not in tuning_history:
        ax.set_visible(False)
        continue

    history     = tuning_history[fold_idx]
    trials      = [h[0] for h in history]
    scores      = [h[1] for h in history]
    best_so_far = [max(scores[:i+1]) for i in range(len(scores))]

    ax.scatter(trials, scores, alpha=0.4, s=15,
               color='#FF7043', label='Trial score')
    ax.plot(trials, best_so_far, color='#BF360C',
            linewidth=2, label='Best so far')

    best_val = max(scores) if scores else 0
    ax.set_title(f'Fold {fold_idx}  (best={best_val:.3f})',
                 fontsize=10)
    ax.set_xlabel('Trial', fontsize=9)
    ax.set_ylabel('Inner AUPRC', fontsize=9)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/xgb_tuning_history.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIGURE_DIR}/xgb_tuning_history.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2: Default vs Tuned — All Splits
# ─────────────────────────────────────────────────────────────
try:
    default_df  = pd.read_csv(f"{RESULTS_DIR}/xgb_results_per_fold.csv")
    split_order = ['Random', 'Scaffold', 'Scaffold-Cluster']
    x     = np.arange(len(split_order))
    width = 0.35

    def get_ms(df, split):
        sub = df[df['split'] == split]['auprc']
        n   = len(sub)
        return (sub.mean(), sub.std()/np.sqrt(n)) if n > 0 else (0, 0)

    def_vals = [get_ms(default_df, s) for s in split_order]
    tun_vals = [get_ms(results_df, s) for s in split_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2,
                   [v[0] for v in def_vals],
                   width, yerr=[v[1] for v in def_vals],
                   color='#FF7043', alpha=0.75, capsize=5,
                   label='XGBoost (Default)', edgecolor='white')
    bars2 = ax.bar(x + width/2,
                   [v[0] for v in tun_vals],
                   width, yerr=[v[1] for v in tun_vals],
                   color='#BF360C', alpha=0.9, capsize=5,
                   label='XGBoost (Tuned)', edgecolor='white')

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.004, f'{h:.3f}',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Delta annotations
    for i, (dv, tv) in enumerate(zip(def_vals, tun_vals)):
        delta = tv[0] - dv[0]
        sign  = '+' if delta >= 0 else ''
        color = '#1B5E20' if delta >= 0 else '#C62828'
        ax.text(x[i], max(dv[0]+dv[1], tv[0]+tv[1]) + 0.012,
                f'{sign}{delta:.3f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=color)

    ax.axhline(y.mean(), color='gray', linestyle=':',
               linewidth=1.5,
               label=f'Random baseline ({y.mean():.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(split_order, fontsize=12)
    ax.set_ylabel('AUPRC (mean ± SEM, 8-fold CV)', fontsize=12)
    ax.set_title('XGBoost: Default vs Tuned Hyperparameters\n'
                 'Antimalarial Activity Prediction',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ymin = min([v[0] for v in def_vals+tun_vals]) * 0.85
    ymax = max([v[0] for v in def_vals+tun_vals]) * 1.2
    ax.set_ylim([max(0, ymin), min(1.0, ymax)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/xgb_tuned_vs_default.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURE_DIR}/xgb_tuned_vs_default.png")

except FileNotFoundError:
    print("Skipping comparison figure — default results not found")

# ─────────────────────────────────────────────────────────────
# PRINT BEST PARAMS SUMMARY ACROSS FOLDS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("BEST HYPERPARAMETERS PER FOLD")
print("─" * 60)
print(f"{'Fold':<5} {'n_est':>6} {'depth':>6} "
      f"{'lr':>8} {'sub':>6} {'col':>6} "
      f"{'min_cw':>7} {'spw':>7}")
print("─" * 60)
for fold_idx in sorted(best_params_per_fold.keys()):
    p = best_params_per_fold[fold_idx]
    print(f"{fold_idx:<5} "
          f"{p['n_estimators']:>6} "
          f"{p['max_depth']:>6} "
          f"{p['learning_rate']:>8.4f} "
          f"{p['subsample']:>6.3f} "
          f"{p['colsample_bytree']:>6.3f} "
          f"{p['min_child_weight']:>7} "
          f"{p['scale_pos_weight']:>7.2f}")

print("\n" + "=" * 60)
print(" COMPLETE — XGBoost tuning done")
print(f"  Results : {RESULTS_DIR}/tuned_xgb_results_per_fold.csv")
print(f"  Params  : {RESULTS_DIR}/tuning_xgb_best_params.json")
print(f"  Figures : {FIGURE_DIR}/")
print("\nNext: python code/3_models_tuned/03_chemprop.py")
print("=" * 60)
