
import json, pickle
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values
y = df['activity'].values.astype(int)

# Morgan fingerprints (radius 2, 2048 bits) 
print('fingerprints...', flush=True)
X = np.zeros((len(smiles), 2048), np.uint8)
for i, s in enumerate(smiles):
    m = Chem.MolFromSmiles(s)
    if m is not None:
        X[i] = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))
rf_cfg = json.load(open('results/tuning_rf_best_params.json'))
xgb_cfg = json.load(open('results/tuning_xgb_best_params.json'))

rows = []
for k, fd in enumerate(folds):
    tr = np.concatenate([np.asarray(fd["train"]), np.asarray(fd["val"])]); te = np.asarray(fd["test"])
    # Random Forest (per-fold tuned config)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_cfg[str(k)])
    rf.fit(X[tr], y[tr])
    p_rf = rf.predict_proba(X[te])[:, 1]
    # XGBoost (per-fold tuned config)
    xc = dict(xgb_cfg[str(k)]); xc.pop('scale_pos_weight', None)
    xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss',
                        tree_method='hist',
                        scale_pos_weight=xgb_cfg[str(k)].get('scale_pos_weight', 1), **xc)
    xgb.fit(X[tr], y[tr])
    p_xgb = xgb.predict_proba(X[te])[:, 1]
    for m_name, p in [('Random Forest', p_rf), ('XGBoost', p_xgb)]:
        for idx, sc in zip(te, p):
            rows.append(dict(model=m_name, split='Scaffold-Cluster', fold=k,
                             y_true=int(y[idx]), y_score=float(sc)))
    from sklearn.metrics import average_precision_score
    print(f'  fold {k}: RF AP {average_precision_score(y[te], p_rf):.3f}  '
          f'XGB AP {average_precision_score(y[te], p_xgb):.3f}', flush=True)

pd.DataFrame(rows).to_csv('results/rf_xgb_tuned_predictions_sc.csv', index=False)
print('saved results/rf_xgb_tuned_predictions_sc.csv')
