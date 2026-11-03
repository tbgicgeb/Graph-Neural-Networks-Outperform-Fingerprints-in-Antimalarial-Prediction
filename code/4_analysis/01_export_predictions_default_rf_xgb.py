"""
generate per-compound RF and XGBoost predictions for every split.

"""

import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

RDLogger.DisableLog('rdApp.*')

RANDOM_SEED = 42
SPLITS = {'Random': 'random_splits.pkl',
          'Scaffold': 'scaffold_splits.pkl',
          'Scaffold-Cluster': 'scaffold_cluster_splits.pkl'}

df = pd.read_csv('malaria_dataset_clean.csv')
y = df['activity'].values
print(f'Loaded {len(df):,} compounds | actives {int(y.sum()):,}')


def morgan(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(2048, dtype=np.uint8)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                    dtype=np.uint8)


print('Computing Morgan fingerprints...')
X = np.array([morgan(s) for s in df['smiles']], dtype=np.uint8)
print(f'X = {X.shape}')

SPW = (y == 0).sum() / (y == 1).sum()

ref = pd.read_csv('results/baselines_all_folds.csv')

rows, preds = [], []
for split_name, fn in SPLITS.items():
    with open(f'splits/{fn}', 'rb') as fh:
        folds = pickle.load(fh)

    for k, fold in enumerate(folds):
        tr = np.asarray(fold['train'])
        te = np.asarray(fold['test'])

        for model_name in ('Random Forest', 'XGBoost'):
            if model_name == 'Random Forest':
                clf = RandomForestClassifier(
                    n_estimators=500, max_depth=None, min_samples_leaf=2,
                    class_weight='balanced', n_jobs=-1,
                    random_state=RANDOM_SEED)
            else:
                clf = XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=SPW, eval_metric='aucpr',
                    tree_method='hist', n_jobs=-1,
                    random_state=RANDOM_SEED)

            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            ap = average_precision_score(y[te], p)

            prev = ref[(ref['split'] == split_name) &
                       (ref['fold'] == k) &
                       (ref['model'] == model_name)]
            prev_ap = float(prev['auprc'].iloc[0]) if len(prev) else np.nan
            d = ap - prev_ap
            flag = '' if abs(d) < 0.02 else '   <-- MISMATCH'
            print(f'  {split_name:17s} fold {k} {model_name:14s} '
                  f'AUPRC {ap:.4f} (recorded {prev_ap:.4f}, d={d:+.4f}){flag}')

            rows.append(dict(split=split_name, fold=k, model=model_name,
                             auprc=ap, auprc_recorded=prev_ap))
            preds.append(pd.DataFrame(dict(
                split=split_name, fold=k, model=model_name,
                smiles=df['smiles'].values[te],
                y_true=y[te], y_score=p)))

pd.DataFrame(rows).to_csv('results/baseline_pred_check.csv', index=False)
pd.concat(preds).to_csv('results/baseline_predictions_by_split.csv', index=False)
print('\nSaved: results/baseline_predictions_by_split.csv')
print('Saved: results/baseline_pred_check.csv')
