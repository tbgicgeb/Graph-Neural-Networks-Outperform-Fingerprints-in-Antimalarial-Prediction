"""
Train-test similarity analysis.
This script performs pairwise comparisons between each test-set active compound and every training-set active compound across all 8 folds of each of the 3 splitting strategies.
"""

import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

DATA = 'malaria_dataset_clean.csv'
SPLITS_DIR = 'splits'
OUT_DIR = 'results/diagnostics/splits'
RADIUS, NBITS = 2, 2048

df = pd.read_csv(DATA)
smiles = df['smiles'].tolist()
labels = df['activity'].values
print(f"Loaded {len(df):,} compounds | actives {int(labels.sum()):,}")


def morgan(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(NBITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
    return np.array(fp, dtype=np.float32)


print("Computing Morgan fingerprints (radius 2, 2048 bits)...")
fps = np.array([morgan(s) for s in smiles], dtype=np.float32)
popcount = fps.sum(axis=1)
print(f"Fingerprints: {fps.shape}")


def max_sims_full(split_list):
    """Max Tanimoto from each test active to ANY training active, all folds."""
    out = []
    for fold in split_list:
        test_act = np.array([i for i in fold['test'] if labels[i] == 1])
        train_act = np.array([i for i in fold['train'] if labels[i] == 1])
        if len(test_act) == 0 or len(train_act) == 0:
            continue
        Q, R = fps[test_act], fps[train_act]
        inter = Q @ R.T                                     # |A and B|
        union = popcount[test_act][:, None] + popcount[train_act][None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.where(union > 0, inter / union, 0.0)
        out.append(sims.max(axis=1))
    return np.concatenate(out)


splits = {
    'Random':           'random_splits.pkl',
    'Scaffold':         'scaffold_splits.pkl',
    'Scaffold-Cluster': 'scaffold_cluster_splits.pkl',
}

rows, dists = [], {}
for name, fn in splits.items():
    with open(f'{SPLITS_DIR}/{fn}', 'rb') as fh:
        split_list = pickle.load(fh)
    s = max_sims_full(split_list)
    dists[name] = s
    rows.append({
        'Split': name, 'n_test_actives': len(s),
        'Mean': s.mean(), 'SD': s.std(), 'Median': np.median(s),
        'Q1': np.percentile(s, 25), 'Q3': np.percentile(s, 75),
        'frac_gt_0.7': float((s > 0.7).mean()),
    })
    print(f"  {name:17s} n={len(s):5d}  mean={s.mean():.3f}  "
          f"median={np.median(s):.3f}  sd={s.std():.3f}")

stats = pd.DataFrame(rows)
stats.to_csv(f'{SPLITS_DIR}/split_similarity_stats_corrected.csv', index=False)
np.savez(f'{SPLITS_DIR}/similarity_distributions.npz', **dists)
print(f"\n{stats.round(3).to_string(index=False)}")
print(f"\nSaved: {SPLITS_DIR}/split_similarity_stats_corrected.csv")
print(f"Saved: {SPLITS_DIR}/similarity_distributions.npz")
