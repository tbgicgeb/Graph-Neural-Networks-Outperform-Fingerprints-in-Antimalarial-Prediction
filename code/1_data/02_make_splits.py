"""
Splitting Strategy
===========================
Generates all three splitting strategies following the Scalia et al., *Nat Biotechnol* 44:1154–1167 (2026) paper:
  1. Random splitting (StratifiedKFold)
  2. Scaffold splitting (Bemis-Murcko)
  3. Scaffold-Cluster splitting (sphere exclusion clustering)
  
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import StratifiedKFold

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = "malaria_dataset_clean.csv"
SPLITS_DIR  = "splits"
FIGURE_DIR  = "results/diagnostics/splits"
N_FOLDS     = 8
RANDOM_SEED = 42

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("=" * 60)
print("SPLITTING STRATEGY")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
df      = pd.read_csv(INPUT_FILE)
smiles  = df['smiles'].tolist()
labels  = df['activity'].values
scaffolds = df['scaffold'].tolist()
indices = np.arange(len(df))

print(f"Loaded: {len(df):,} compounds")
print(f"Actives: {labels.sum():,} ({labels.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# MORGAN FINGERPRINTS (for similarity analysis)
# ─────────────────────────────────────────────────────────────
print("\nComputing Morgan fingerprints for similarity analysis...")

def get_morgan_fp(smi, radius=2, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(nbits, dtype=bool)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return np.array(fp, dtype=bool)
    except Exception:
        return np.zeros(nbits, dtype=bool)

fps = np.array([get_morgan_fp(s) for s in smiles])
print(f"Fingerprints: {fps.shape}")

def tanimoto_batch(query, refs):
    """Vectorized Tanimoto between one query and many refs."""
    inter = np.sum(query & refs, axis=1)
    union = np.sum(query | refs, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(union > 0, inter / union, 0.0)
    return sims

# ─────────────────────────────────────────────────────────────
# SPLIT 1: RANDOM (StratifiedKFold — preserves class ratio)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("SPLIT 1: Random (Stratified K-Fold)")
print("─" * 40)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                      random_state=RANDOM_SEED)
random_splits = []

for fold_idx, (trainval_idx, test_idx) in enumerate(
        skf.split(indices, labels)):
    # 80:10:10 → train:val:test
    np.random.seed(RANDOM_SEED + fold_idx)
    np.random.shuffle(trainval_idx)
    cut      = int(len(trainval_idx) * (8/9))   # 80% of total
    train_idx = trainval_idx[:cut]
    val_idx   = trainval_idx[cut:]

    random_splits.append({
        'fold':  fold_idx,
        'train': train_idx.tolist(),
        'val':   val_idx.tolist(),
        'test':  test_idx.tolist(),
    })
    print(f"  Fold {fold_idx}: "
          f"train={len(train_idx):,} "
          f"val={len(val_idx):,} "
          f"test={len(test_idx):,} "
          f"| test_actives={labels[test_idx].sum()}")

with open(f"{SPLITS_DIR}/random_splits.pkl", 'wb') as f:
    pickle.dump(random_splits, f)
print(f"Saved: {SPLITS_DIR}/random_splits.pkl")

# ─────────────────────────────────────────────────────────────
# SPLIT 2: SCAFFOLD (Bemis-Murcko)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("SPLIT 2: Scaffold (Bemis-Murcko)")
print("─" * 40)

# Group indices by scaffold
# Note: some scaffolds may be NaN (float) for acyclic compounds
sc_to_idx = {}
for i, sc in enumerate(scaffolds):
    if not isinstance(sc, str) or sc.strip() == '':
        key = 'no_scaffold'
    else:
        key = sc
    sc_to_idx.setdefault(key, []).append(i)

# Sort scaffold groups by size descending
sc_groups = sorted(sc_to_idx.values(), key=len, reverse=True)
print(f"Unique scaffolds: {len(sc_groups):,}")

# Greedy fold assignment (balance total compounds per fold)
fold_counts = [0] * N_FOLDS
fold_members = [[] for _ in range(N_FOLDS)]
for grp in sc_groups:
    f = int(np.argmin(fold_counts))
    fold_members[f].extend(grp)
    fold_counts[f] += len(grp)

# Build 8-fold splits (test=one fold, val=next fold, train=rest)
scaffold_splits = []
for test_fold in range(N_FOLDS):
    val_fold  = (test_fold + 1) % N_FOLDS
    train_idx = np.array([
        i for f in range(N_FOLDS)
        if f != test_fold and f != val_fold
        for i in fold_members[f]
    ])
    val_idx  = np.array(fold_members[val_fold])
    test_idx = np.array(fold_members[test_fold])

    scaffold_splits.append({
        'fold':  test_fold,
        'train': train_idx.tolist(),
        'val':   val_idx.tolist(),
        'test':  test_idx.tolist(),
    })
    print(f"  Fold {test_fold}: "
          f"train={len(train_idx):,} "
          f"val={len(val_idx):,} "
          f"test={len(test_idx):,} "
          f"| test_actives={labels[test_idx].sum()}")

with open(f"{SPLITS_DIR}/scaffold_splits.pkl", 'wb') as f:
    pickle.dump(scaffold_splits, f)
print(f"Saved: {SPLITS_DIR}/scaffold_splits.pkl")

# ─────────────────────────────────────────────────────────────
# SPLIT 3: SCAFFOLD-CLUSTER
# Method: encode scaffolds → sphere-exclusion cluster →
#         assign clusters to folds
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("SPLIT 3: Scaffold-Cluster (Sphere Exclusion)")
print("─" * 40)

# Unique scaffold SMILES and their fps
unique_scs  = list(sc_to_idx.keys())
print(f"Unique scaffolds to cluster: {len(unique_scs):,}")

print("Computing scaffold fingerprints...")
sc_fps_list = []
valid_scs   = []
for sc in unique_scs:
    # Guard: skip NaN/float values (molecules with no ring system
    # produce empty Murcko scaffolds stored as NaN in the CSV)
    if not isinstance(sc, str):
        sc_fps_list.append(np.zeros(2048, dtype=bool))
        valid_scs.append('no_scaffold')
        continue
    if sc == 'no_scaffold' or sc.strip() == '':
        sc_fps_list.append(np.zeros(2048, dtype=bool))
        valid_scs.append(sc)
        continue
    mol = Chem.MolFromSmiles(sc)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sc_fps_list.append(np.array(fp, dtype=bool))
        valid_scs.append(sc)
    else:
        # Unparseable scaffold — treat as no_scaffold
        sc_fps_list.append(np.zeros(2048, dtype=bool))
        valid_scs.append('no_scaffold')

sc_fps_arr = np.array(sc_fps_list)
print(f"Scaffold fps: {sc_fps_arr.shape}")

# Sphere exclusion clustering (threshold = min similarity to merge)
CLUSTER_THRESHOLD = 0.4   # Tanimoto ≥ 0.4 → same cluster
                            # lower = fewer, larger clusters
                            # higher = more, smaller clusters

print(f"Sphere exclusion clustering (Tanimoto ≥ {CLUSTER_THRESHOLD})...")
cluster_ids   = [-1] * len(valid_scs)
cluster_centers_fp = []
n_clusters = 0

for i in range(len(valid_scs)):
    fp_i = sc_fps_arr[i]
    if n_clusters == 0:
        cluster_ids[i] = 0
        cluster_centers_fp.append(fp_i)
        n_clusters += 1
        continue

    centers_arr = np.array(cluster_centers_fp)
    sims = tanimoto_batch(fp_i, centers_arr)
    best_sim = sims.max()
    best_cl  = int(sims.argmax())

    if best_sim >= CLUSTER_THRESHOLD:
        cluster_ids[i] = best_cl
    else:
        cluster_ids[i] = n_clusters
        cluster_centers_fp.append(fp_i)
        n_clusters += 1

    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(valid_scs)} scaffolds → "
              f"{n_clusters} clusters")

print(f"Total scaffold clusters: {n_clusters:,}")
sc_to_cluster = {sc: cl for sc, cl in zip(valid_scs, cluster_ids)}

# Map compound indices to clusters
cluster_to_cpd = {}
for i, sc in enumerate(scaffolds):
    if not isinstance(sc, str) or sc.strip() == '':
        key = 'no_scaffold'
    else:
        key = sc
    cl = sc_to_cluster.get(key, n_clusters)  # fallback cluster
    cluster_to_cpd.setdefault(cl, []).append(i)

cluster_groups = sorted(cluster_to_cpd.values(), key=len, reverse=True)
print(f"Compound clusters : {len(cluster_groups):,}")
sizes = [len(g) for g in cluster_groups]
print(f"Largest cluster   : {sizes[0]:,} compounds")
print(f"Median cluster    : {int(np.median(sizes)):,} compounds")

# Greedy fold assignment
fold_counts_cl  = [0] * N_FOLDS
fold_members_cl = [[] for _ in range(N_FOLDS)]
for grp in cluster_groups:
    f = int(np.argmin(fold_counts_cl))
    fold_members_cl[f].extend(grp)
    fold_counts_cl[f] += len(grp)

# Build splits
scaffold_cluster_splits = []
for test_fold in range(N_FOLDS):
    val_fold  = (test_fold + 1) % N_FOLDS
    train_idx = np.array([
        i for f in range(N_FOLDS)
        if f != test_fold and f != val_fold
        for i in fold_members_cl[f]
    ])
    val_idx  = np.array(fold_members_cl[val_fold])
    test_idx = np.array(fold_members_cl[test_fold])

    scaffold_cluster_splits.append({
        'fold':  test_fold,
        'train': train_idx.tolist(),
        'val':   val_idx.tolist(),
        'test':  test_idx.tolist(),
    })
    print(f"  Fold {test_fold}: "
          f"train={len(train_idx):,} "
          f"val={len(val_idx):,} "
          f"test={len(test_idx):,} "
          f"| test_actives={labels[test_idx].sum()}")

with open(f"{SPLITS_DIR}/scaffold_cluster_splits.pkl", 'wb') as f:
    pickle.dump(scaffold_cluster_splits, f)
print(f"Saved: {SPLITS_DIR}/scaffold_cluster_splits.pkl")

# ─────────────────────────────────────────────────────────────
# SIMILARITY ANALYSIS (reproduces GNEprop Fig 3a)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("SIMILARITY ANALYSIS (GNEprop Fig 3a style)")
print("─" * 40)

all_splits = {
    'Random':           random_splits,
    'Scaffold':         scaffold_splits,
    'Scaffold-Cluster': scaffold_cluster_splits,
}

active_idx = np.where(labels == 1)[0]
np.random.seed(42)
# Sample up to 300 actives for speed
sample_actives = np.random.choice(
    active_idx,
    size=min(300, len(active_idx)),
    replace=False
)

def compute_max_sims(splits, fps, labels, query_idx, max_ref=500):
    """
    For each query in test set, compute max Tanimoto
    to training actives.
    """
    sims_all = []
    for fold_data in splits:
        test_set   = set(fold_data['test'])
        train_set  = set(fold_data['train'])
        tr_actives = [i for i in train_set if labels[i] == 1]
        ref_fps    = fps[tr_actives[:max_ref]]

        for qi in query_idx:
            if qi in test_set:
                s = tanimoto_batch(fps[qi], ref_fps)
                sims_all.append(float(s.max()) if len(s) > 0 else 0.0)
    return sims_all

sims_results = {}
for split_name, split_list in all_splits.items():
    print(f"  Computing {split_name} similarities...")
    sims_results[split_name] = compute_max_sims(
        split_list, fps, labels, sample_actives
    )
    print(f"    Mean: {np.mean(sims_results[split_name]):.3f}  "
          f"Median: {np.median(sims_results[split_name]):.3f}")

# Save similarity stats
stats_df = pd.DataFrame({
    'Split':  list(sims_results.keys()),
    'Mean':   [np.mean(v) for v in sims_results.values()],
    'Median': [np.median(v) for v in sims_results.values()],
    'Std':    [np.std(v)   for v in sims_results.values()],
})
stats_df.to_csv(f"{SPLITS_DIR}/split_similarity_stats.csv", index=False)
print(f"\n{stats_df.round(3).to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# FIGURE: Similarity distributions (GNEprop Fig 3a style)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(0, 1, 31)

colors_split = {
    'Random':           '#4CAF50',
    'Scaffold':         '#FF7043',
    'Scaffold-Cluster': '#7C4DFF',
}
for split_name, sims in sims_results.items():
    ax.hist(sims, bins=bins, density=True,
            alpha=0.65, color=colors_split[split_name],
            label=split_name, edgecolor='none')
    ax.axvline(np.mean(sims),
               color=colors_split[split_name],
               linestyle='--', linewidth=1.5, alpha=0.9)

ax.set_xlabel(
    'Maximum Tanimoto Similarity\n(Test Active → Training Actives)',
    fontsize=12
)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Train–Test Similarity by Splitting Strategy\n'
             '(lower = harder, more realistic evaluation)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{FIGURE_DIR}/splitting_similarity_comparison.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIGURE_DIR}/splitting_similarity_comparison.png")

print("\n" + "=" * 60)
print("COMPLETE")
print(f"Splits directory: {SPLITS_DIR}/")
print("  random_splits.pkl")
print("  scaffold_splits.pkl")
print("  scaffold_cluster_splits.pkl")
print("=" * 60)
print("\nNEXT: python code/2_models_default/01_random_forest_and_xgboost.py")
