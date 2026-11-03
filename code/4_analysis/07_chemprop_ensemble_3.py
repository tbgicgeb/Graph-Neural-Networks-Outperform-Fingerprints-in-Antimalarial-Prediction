

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.metrics import average_precision_score

from chemprop import data as cpdata
from chemprop import models as cpmodels
from chemprop import featurizers

warnings.filterwarnings('ignore')

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values
y = df['activity'].values.astype(int)
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))


def predict(ckpt, te_idx):
    ds = cpdata.MoleculeDataset(
        [cpdata.MoleculeDatapoint.from_smi(smiles[i], np.array([0.0])) for i in te_idx],
        featurizer=feat)
    dl = cpdata.build_dataloader(ds, batch_size=64, num_workers=0, shuffle=False)
    model = cpmodels.MPNN.load_from_checkpoint(ckpt)
    model.eval()
    tr = pl.Trainer(accelerator='gpu', devices=1, logger=False,
                    enable_progress_bar=False, enable_model_summary=False)
    try:
        raw = tr.predict(model, dl, weights_only=False)
    except TypeError:
        raw = tr.predict(model, dl)
    del model, tr
    torch.cuda.empty_cache()
    return np.clip(torch.cat(raw).squeeze(-1).numpy(), 0, 1)


rows = []
for fd in folds:
    k = fd['fold']
    te = np.asarray(fd['test'])
    members = sorted(glob.glob(f'chemprop_strong_tmp_f{k}/f{k}/final*/best.ckpt'))
    if len(members) < 3:
        print(f'  fold {k}: only {len(members)} members, skipping')
        continue
    preds = [predict(m, te) for m in members]
    single = average_precision_score(y[te], preds[0])
    ens = average_precision_score(y[te], np.mean(preds, axis=0))
    rows.append(dict(fold=k, single=single, ensemble=ens))
    print(f'  fold {k}: single {single:.4f}  ensemble {ens:.4f}  (Δens {ens-single:+.4f})',
          flush=True)

r = pd.DataFrame(rows)
r.to_csv('results/chemprop_strong_single_vs_ensemble.csv', index=False)
print('\n' + '=' * 60)
print(f'Single-model tuned ChemProp : {r.single.mean():.4f} ± {r.single.sem():.4f}')
print(f'3-model ensemble            : {r.ensemble.mean():.4f} ± {r.ensemble.sem():.4f}')
print(f'ensemble contribution       : {(r.ensemble-r.single).mean():+.4f}')
print(f'reference — default 0.565 | standard-tuned 0.560 | GNEprop 0.601')
