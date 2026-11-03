

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from chemprop import data as cpdata
from chemprop import models as cpmodels
from chemprop import featurizers

warnings.filterwarnings('ignore')

BATCH_SIZE = 50
SPLITS = {'Random': 'random_splits.pkl',
          'Scaffold': 'scaffold_splits.pkl',
          'Scaffold-Cluster': 'scaffold_cluster_splits.pkl'}

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].tolist()
y = df['activity'].values
print(f'Loaded {len(df):,} compounds | actives {int(y.sum()):,}')

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()


def bedroc(y_true, y_score, alpha=20.0):
    n = len(y_true)
    na = int(y_true.sum())
    if na == 0 or na == n:
        return 0.5
    order = np.argsort(-y_score)
    ys = y_true[order]
    ra = na / n
    ri_sum = sum(np.exp(-alpha * (i + 1) / n)
                 for i, yi in enumerate(ys) if yi == 1)
    rie = ri_sum / (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    return (rie - rie_min) / (rie_max - rie_min)


rows, preds = [], []
for split_name, fn in SPLITS.items():
    slug = split_name.lower().replace('-', '_').replace(' ', '_')
    with open(f'splits/{fn}', 'rb') as fh:
        folds = pickle.load(fh)

    for fd in folds:
        k = fd['fold']
        fold_dir = f'chemprop_tmp/{slug}/fold{k}'
        ckpt = f'{fold_dir}/best.ckpt'
        if not os.path.exists(ckpt):
            print(f'  [skip] {split_name} fold {k}: no checkpoint')
            continue

        test_idx = np.array(fd['test'])
        test_smi = [smiles[i] for i in test_idx]
        y_test = y[test_idx]

        points = [cpdata.MoleculeDatapoint.from_smi(s, np.array([0.0]))
                  for s in test_smi]
        ds = cpdata.MoleculeDataset(points, featurizer=feat)
        loader = cpdata.build_dataloader(ds, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

        model = cpmodels.MPNN.load_from_checkpoint(ckpt)
        model.eval()
        trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False,
                             enable_progress_bar=False,
                             enable_model_summary=False)
        try:
            raw = trainer.predict(model, loader, weights_only=False)
        except TypeError:
            raw = trainer.predict(model, loader)

        p = torch.cat(raw, dim=0).squeeze(-1).numpy()
        p = np.clip(p, 0.0, 1.0)

        ap = average_precision_score(y_test, p)
        prev_file = f'{fold_dir}/fold_result.csv'
        prev_ap = (float(pd.read_csv(prev_file).iloc[0]['auprc'])
                   if os.path.exists(prev_file) else np.nan)
        d = ap - prev_ap
        flag = '' if abs(d) < 0.02 else '   <-- MISMATCH'
        print(f'  {split_name:17s} fold {k}: AUPRC {ap:.4f} '
              f'(recorded {prev_ap:.4f}, d={d:+.4f}){flag}')

        rows.append(dict(split=split_name, fold=k, model='ChemProp (D-MPNN)',
                         auprc=ap, auroc=roc_auc_score(y_test, p),
                         bedroc=bedroc(y_test, p),
                         auprc_recorded=prev_ap, auprc_delta=d))
        preds.append(pd.DataFrame(dict(
            split=split_name, fold=k, model='ChemProp (D-MPNN)',
            smiles=test_smi, y_true=y_test, y_score=p)))

        del model, trainer
        torch.cuda.empty_cache()

out = pd.DataFrame(rows)
out.to_csv('results/chemprop_pred_check.csv', index=False)
pd.concat(preds).to_csv('results/chemprop_predictions_by_split.csv', index=False)
print(f'\nMax |AUPRC delta|: {out.auprc_delta.abs().max():.4f}')
print(out.groupby('split')[['auprc', 'auprc_recorded']].mean().round(4).to_string())
print('\nSaved: results/chemprop_predictions_by_split.csv')
