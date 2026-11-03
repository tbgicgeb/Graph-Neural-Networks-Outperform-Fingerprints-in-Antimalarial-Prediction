"""
Best-effort ChemProp tuning with an expanded, OOD-aware search.

"""

import os
import gc
import pickle
import warnings

import numpy as np
import pandas as pd
import optuna
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from chemprop import data as cpdata
from chemprop import nn as cpnn
from chemprop import models as cpmodels
from chemprop import featurizers

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
N_TRIALS = 12
ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
_SUFFIX = os.environ.get('LC_OUT_SUFFIX', '')
_FOLDS = os.environ.get('LC_FOLDS')
_KEEP = set(int(x) for x in _FOLDS.split(',')) if _FOLDS else None
OUT = f'results/chemprop_strong_tuned_per_fold{_SUFFIX}.csv'
CKPT = f'chemprop_strong_tmp{_SUFFIX}'

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values
y = df['activity'].values.astype(int)
scaffold = df['scaffold'].values
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

AGG = {'mean': cpnn.MeanAggregation, 'sum': cpnn.SumAggregation,
       'norm': cpnn.NormAggregation}


def bedroc(y_true, y_score, alpha=20.0):
    n = len(y_true); na = int(y_true.sum())
    if na == 0 or na == n:
        return 0.5
    ys = y_true[np.argsort(-y_score)]
    ra = na / n
    ri = sum(np.exp(-alpha * (i + 1) / n) for i, v in enumerate(ys) if v == 1)
    rie = ri / (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    rmax = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rmin = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    return (rie - rmin) / (rmax - rmin)


def scaffold_inner_split(train_idx, frac_val=0.15, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for i in train_idx:
        groups.setdefault(scaffold[i], []).append(i)
    scafs = list(groups); rng.shuffle(scafs)
    target = int(round(frac_val * len(train_idx)))
    va, n = [], 0
    for s in scafs:
        if n >= target:
            break
        va.extend(groups[s]); n += len(groups[s])
    va_set = set(va)
    return np.array([i for i in train_idx if i not in va_set]), np.array(va)


def make_ds(idx, labelled=True):
    pts = [cpdata.MoleculeDatapoint.from_smi(
        smiles[i], np.array([float(y[i])] if labelled else [0.0])) for i in idx]
    return cpdata.MoleculeDataset(pts, featurizer=feat)


def build(p, seed=0):
    pl.seed_everything(seed, workers=True)
    mp = cpnn.BondMessagePassing(d_h=p['hidden_size'], depth=p['depth'],
                                 dropout=p['dropout'])
    ffn = cpnn.BinaryClassificationFFN(input_dim=p['hidden_size'],
                                       hidden_dim=p['ffn_hidden'],
                                       n_layers=p['ffn_layers'], dropout=p['dropout'])
    return cpmodels.MPNN(message_passing=mp, agg=AGG[p['agg']](), predictor=ffn,
                         batch_norm=p['batch_norm'],
                         warmup_epochs=p['warmup'], init_lr=p['max_lr'] / 10,
                         max_lr=p['max_lr'], final_lr=p['max_lr'] / 10,
                         metrics=[cpnn.metrics.BinaryAUPRC(), cpnn.metrics.BinaryAUROC()])


def train_predict(tr_idx, va_idx, te_idx, p, ckpt_dir, seed=0):
    os.makedirs(ckpt_dir, exist_ok=True)
    tr = cpdata.build_dataloader(make_ds(tr_idx), batch_size=p['batch_size'],
                                 num_workers=0, shuffle=True,
                                 class_balance=p['class_balance'], seed=seed)
    va = cpdata.build_dataloader(make_ds(va_idx), batch_size=p['batch_size'],
                                 num_workers=0, shuffle=False)
    te = cpdata.build_dataloader(make_ds(te_idx, labelled=False),
                                 batch_size=p['batch_size'], num_workers=0, shuffle=False)
    model = build(p, seed)
    ck = ModelCheckpoint(dirpath=ckpt_dir, monitor='val/prc', mode='max',
                         save_top_k=1, filename='best')
    es = EarlyStopping(monitor='val/prc', patience=10, mode='max')
    tr_ = pl.Trainer(max_epochs=p['max_epochs'], accelerator=ACCELERATOR, devices=1,
                     logger=False, enable_progress_bar=False,
                     enable_model_summary=False, callbacks=[ck, es])
    tr_.fit(model, tr, va)
    if ck.best_model_path and os.path.exists(ck.best_model_path):
        model = cpmodels.MPNN.load_from_checkpoint(ck.best_model_path)
    model.eval()
    pt = pl.Trainer(accelerator=ACCELERATOR, devices=1, logger=False,
                    enable_progress_bar=False, enable_model_summary=False)
    try:
        raw = pt.predict(model, te, weights_only=False)
    except TypeError:
        raw = pt.predict(model, te)
    out = np.clip(torch.cat(raw).squeeze(-1).numpy(), 0, 1)
    del model, tr_, pt
    torch.cuda.empty_cache(); gc.collect()
    return out


def suggest(trial):
    return dict(
        hidden_size=trial.suggest_categorical('hidden_size', [300, 500, 750]),
        depth=trial.suggest_int('depth', 3, 6),
        dropout=trial.suggest_float('dropout', 0.0, 0.35, step=0.05),
        ffn_layers=trial.suggest_int('ffn_layers', 1, 3),
        ffn_hidden=trial.suggest_categorical('ffn_hidden', [300, 500, 750]),
        agg=trial.suggest_categorical('agg', ['mean', 'sum', 'norm']),
        batch_norm=trial.suggest_categorical('batch_norm', [True, False]),
        max_lr=trial.suggest_float('max_lr', 1e-4, 3e-3, log=True),
        class_balance=trial.suggest_categorical('class_balance', [True, False]),
        batch_size=trial.suggest_categorical('batch_size', [32, 50, 64]),
        warmup=trial.suggest_int('warmup', 2, 5),
        max_epochs=trial.suggest_categorical('max_epochs', [30, 50]))


def main():
    folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))
    done = set(pd.read_csv(OUT).fold) if os.path.exists(OUT) else set()

    for fd in folds:
        k = fd['fold']
        if (_KEEP is not None and k not in _KEEP) or k in done:
            continue
        tr_full = np.concatenate([fd['train'], fd['val']])
        te = np.asarray(fd['test'])
        in_tr, in_va = scaffold_inner_split(tr_full, seed=RANDOM_SEED + k)
        print(f'\n=== fold {k}: inner-train {len(in_tr)}, OOD inner-val {len(in_va)} ===',
              flush=True)

        def objective(trial):
            p = suggest(trial)
            pred = train_predict(in_tr, in_va, in_va, p,
                                 f'{CKPT}/f{k}/t{trial.number}', seed=RANDOM_SEED)
            return average_precision_score(y[in_va], pred)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=N_TRIALS)
        best = study.best_params
        best.setdefault('ffn_hidden', best['hidden_size'])
        print(f'  best OOD-inner AUPRC {study.best_value:.4f} : {best}', flush=True)

        # 3-model ensemble at the selected config 
        preds = []
        for s in range(3):
            preds.append(train_predict(tr_full, in_va, te, best,
                                       f'{CKPT}/f{k}/final{s}', seed=RANDOM_SEED + s))
        pred = np.mean(preds, axis=0)

        row = dict(fold=k, split='Scaffold-Cluster', model='ChemProp (strong-tuned)',
                   auprc=average_precision_score(y[te], pred),
                   auroc=roc_auc_score(y[te], pred),
                   f1=f1_score(y[te], (pred >= 0.5).astype(int), zero_division=0),
                   bedroc=bedroc(y[te], pred), n_test=len(te),
                   n_test_actives=int(y[te].sum()),
                   **{f'best_{kk}': vv for kk, vv in best.items()})
        pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT),
                                   index=False)
        print(f'  fold {k} TEST AUPRC {row["auprc"]:.4f} (default 0.565)', flush=True)


if __name__ == '__main__':
    main()
