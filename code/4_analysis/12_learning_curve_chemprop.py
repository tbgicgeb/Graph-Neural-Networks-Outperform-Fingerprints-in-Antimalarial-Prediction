
import os
import gc
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from chemprop import data as cpdata
from chemprop import nn as cpnn
from chemprop import models as cpmodels
from chemprop import featurizers

warnings.filterwarnings('ignore')

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
FOLDS = [0, 1, 2]
SEED = 42
OUT = 'results/chemprop_learning_curve.csv'
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values
y = df['activity'].values.astype(int)

# strong fixed configuration (sum aggregation + batch norm, OOD-tuned regime)
P = dict(hidden_size=500, depth=5, dropout=0.10, ffn_layers=2, ffn_hidden=500,
         batch_size=50, max_epochs=30)


def ds(idx, lab=True):
    pts = [cpdata.MoleculeDatapoint.from_smi(smiles[i],
           np.array([float(y[i])] if lab else [0.0])) for i in idx]
    return cpdata.MoleculeDataset(pts, featurizer=feat)


def train_predict(tr, va, te):
    trl = cpdata.build_dataloader(ds(tr), batch_size=P['batch_size'], num_workers=0, shuffle=True)
    val = cpdata.build_dataloader(ds(va), batch_size=P['batch_size'], num_workers=0, shuffle=False)
    tel = cpdata.build_dataloader(ds(te, False), batch_size=P['batch_size'], num_workers=0, shuffle=False)
    pl.seed_everything(SEED, workers=True)
    mp = cpnn.BondMessagePassing(d_h=P['hidden_size'], depth=P['depth'], dropout=P['dropout'])
    ffn = cpnn.BinaryClassificationFFN(input_dim=P['hidden_size'], hidden_dim=P['ffn_hidden'],
                                       n_layers=P['ffn_layers'], dropout=P['dropout'])
    model = cpmodels.MPNN(message_passing=mp, agg=cpnn.SumAggregation(), predictor=ffn,
                          batch_norm=True,
                          metrics=[cpnn.metrics.BinaryAUPRC(), cpnn.metrics.BinaryAUROC()])
    ck = ModelCheckpoint(dirpath='chemprop_lc_tmp', monitor='val/prc', mode='max',
                         save_top_k=1, filename='best')
    es = EarlyStopping(monitor='val/prc', patience=8, mode='max')
    tr_ = pl.Trainer(max_epochs=P['max_epochs'], accelerator='gpu', devices=1, logger=False,
                     enable_progress_bar=False, enable_model_summary=False, callbacks=[ck, es])
    tr_.fit(model, trl, val)
    if ck.best_model_path and os.path.exists(ck.best_model_path):
        model = cpmodels.MPNN.load_from_checkpoint(ck.best_model_path)
    model.eval()
    pt = pl.Trainer(accelerator='gpu', devices=1, logger=False,
                    enable_progress_bar=False, enable_model_summary=False)
    try:
        raw = pt.predict(model, tel, weights_only=False)
    except TypeError:
        raw = pt.predict(model, tel)
    out = np.clip(torch.cat(raw).squeeze(-1).numpy(), 0, 1)
    del model, tr_, pt
    torch.cuda.empty_cache(); gc.collect()
    return out


def main():
    folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))
    rows = []
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); done = set(zip(d.fold, d.fraction)); rows = d.to_dict('records')
    for k in FOLDS:
        fd = folds[k]
        va, te = np.asarray(fd['val']), np.asarray(fd['test'])
        for frac in FRACTIONS:
            if (k, frac) in done:
                continue
            tr = np.asarray(fd['train'])
            if frac < 1.0:
                tr, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=SEED + k)
            p = train_predict(tr, va, te)
            ap = average_precision_score(y[te], p)
            print(f'  fold {k} frac {frac:.2f} n={len(tr):5d}  AUPRC {ap:.4f}', flush=True)
            rows.append(dict(model='ChemProp', fold=k, fraction=frac, n_train=len(tr), auprc=ap))
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print('done; saved', OUT)


if __name__ == '__main__':
    main()
