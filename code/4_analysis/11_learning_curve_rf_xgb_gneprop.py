"""
Learning curves under scaffold-cluster splitting.

Usage:
    python code/4_analysis/11_learning_curve_rf_xgb_gneprop.py --models rf xgb          # minutes
    python code/4_analysis/11_learning_curve_rf_xgb_gneprop.py --models gneprop --folds 0 1 2   # hours
"""

import os
import sys
import json
import glob
import pickle
import shutil
import argparse
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

ps = os.environ.get('LC_FRACTIONS')
FRACTIONS = ([float(x) for x in ps.split(',')] if ps
             else [0.25, 0.50, 0.75, 1.00])
SPLIT = 'Scaffold-Cluster'
SPLIT_FILE = 'splits/scaffold_cluster_splits.pkl'
RANDOM_SEED = 42
OUT = 'results/learning_curve.csv'

GNEPROP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'gneprop_pyg.py')

GNEPROP_PARAMS = dict(
    lr='1.4604932648780521e-05', hidden_size='300', depth='6',
    num_readout_layers='1', dropout='0.05', lr_strategy='warmup_cosine_step',
    aggr='sum', gpus='1', max_epochs='50', metric='val_ap', num_workers='0',
    ig_baseline_ratio='0.5', adv='flag', adv_m='3',
)


def append_row(row):
    """Persist each result as soon as it is produced.

    Individual GNEprop runs take tens of minutes, so a crash or an interrupt
    partway through the grid must not discard completed work.
    """
    df = pd.DataFrame([row])
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode='a', header=header, index=False)


def already_done(model, fold, frac):
    if not os.path.exists(OUT):
        return False
    d = pd.read_csv(OUT)
    return len(d[(d.model == model) & (d.fold == fold) &
                 (np.isclose(d.fraction, frac))]) > 0


def subsample(train_idx, y, frac, seed):
    """Stratified subsample of a fold's training indices."""
    if frac >= 1.0:
        return np.asarray(train_idx)
    keep, _ = train_test_split(
        np.asarray(train_idx), train_size=frac,
        stratify=y[np.asarray(train_idx)], random_state=seed)
    return keep


# ────────────────────────────── fingerprint models ──────────────────────────
def run_baselines(folds, df, y, models):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier
    RDLogger.DisableLog('rdApp.*')

    cache = 'results/_morgan_cache.npy'
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        def fp(s):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return np.zeros(2048, dtype=np.uint8)
            return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
                            dtype=np.uint8)
        X = np.array([fp(s) for s in df['smiles']], dtype=np.uint8)
        np.save(cache, X)
    print(f'  fingerprints {X.shape}')

    SPW = (y == 0).sum() / (y == 1).sum()
    rows = []
    for k, fold in enumerate(folds):
        te = np.asarray(fold['test'])
        for frac in FRACTIONS:
            tr = subsample(fold['train'], y, frac, RANDOM_SEED + k)
            for name in models:
                if name == 'rf':
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(
                        n_estimators=1000, max_depth=None, min_samples_leaf=1,
                        max_features='sqrt', class_weight='balanced',
                        n_jobs=-1, random_state=RANDOM_SEED)
                    label = 'Random Forest'
                elif name == 'xgb':
                    from xgboost import XGBClassifier
                    clf = XGBClassifier(
                        n_estimators=1100, max_depth=10, learning_rate=0.06,
                        subsample=0.83, colsample_bytree=0.72,
                        reg_alpha=1.94, reg_lambda=6.69,
                        scale_pos_weight=SPW, eval_metric='aucpr',
                        tree_method='hist', n_jobs=-1,
                        random_state=RANDOM_SEED)
                    label = 'XGBoost'
                else:
                    continue

                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                ap = average_precision_score(y[te], p)
                print(f'  fold {k} frac {frac:.2f} {label:14s} '
                      f'n_train={len(tr):6,d}  AUPRC {ap:.4f}')
                rows.append(dict(model=label, fold=k, fraction=frac,
                                 n_train=len(tr), auprc=ap,
                                 auroc=roc_auc_score(y[te], p)))
    return rows


# ────────────────────────────────── GNEprop ─────────────────────────────────
def run_gneprop(folds, df, y, fold_ids):
    smiles = df['smiles'].tolist()
    rows = []
    for k in fold_ids:
        fold = folds[k]
        va = np.asarray(fold['val'])
        te = np.asarray(fold['test'])
        for frac in FRACTIONS:
            if already_done('GNEprop', k, frac):
                print(f'  fold {k} frac {frac:.2f} GNEprop already done, skipping')
                continue
            tr = subsample(fold['train'], y, frac, RANDOM_SEED + k)
            work = f'learning_curve_tmp/fold{k}_frac{int(frac*100)}'
            log = f'{work}/gneprop_output'
            os.makedirs(work, exist_ok=True)

            all_idx = np.concatenate([tr, va, te])
            pd.DataFrame({'SMILES': [smiles[i] for i in all_idx],
                          'Activity': y[all_idx].astype(int)}
                         ).to_csv(f'{work}/combined.csv', index=False)
            n1, n2, n3 = len(tr), len(va), len(te)
            with open(f'{work}/split_indices.pkl', 'wb') as fh:
                pickle.dump([(np.arange(0, n1, dtype=np.int64),
                              np.arange(n1, n1 + n2, dtype=np.int64),
                              np.arange(n1 + n2, n1 + n2 + n3, dtype=np.int64))], fh)

            cmd = [sys.executable, GNEPROP_SCRIPT,
                   '--dataset_path', f'{work}/combined.csv',
                   '--split_type', 'index_predetermined',
                   '--index_predetermined_file', f'{work}/split_indices.pkl',
                   '--log_directory', log, '--exclude_bn_bias']
            for key, val in GNEPROP_PARAMS.items():
                cmd += [f'--{key}', val]

            print(f'  fold {k} frac {frac:.2f} GNEprop n_train={len(tr):,} ...',
                  flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f'    FAILED (rc={res.returncode})')
                print('    ' + res.stderr[-600:].replace('\n', '\n    '))
                continue

            metrics = sorted(glob.glob(f'{log}/*/final_metrics.pkl'))
            if not metrics:
                print('    no final_metrics.pkl produced')
                continue
            with open(metrics[-1], 'rb') as fh:
                m = pickle.load(fh)
            ap = float(m['test_ap']['mean'])
            auc = float(m['test_auc']['mean'])
            print(f'    AUPRC {ap:.4f}', flush=True)
            row = dict(model='GNEprop', fold=k, fraction=frac,
                       n_train=len(tr), auprc=ap, auroc=auc)
            append_row(row)
            rows.append(row)

            
            shutil.rmtree(log, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['rf'],
                    choices=['rf', 'xgb', 'gneprop'])
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(8)))
    args = ap.parse_args()

    df = pd.read_csv('malaria_dataset_clean.csv')
    y = df['activity'].values
    with open(SPLIT_FILE, 'rb') as fh:
        folds = pickle.load(fh)
    print(f'{len(df):,} compounds | {SPLIT} | fractions {FRACTIONS}')

    rows = []
    base = [m for m in args.models if m in ('rf', 'xgb')]
    if base:
        rows += run_baselines([folds[i] for i in args.folds], df, y, base)
    if 'gneprop' in args.models:
        rows += run_gneprop(folds, df, y, args.folds)

    
    frames = [pd.read_csv(OUT)] if os.path.exists(OUT) else []
    if rows:
        frames.append(pd.DataFrame(rows))
    new = pd.concat(frames, ignore_index=True)
    new = new.drop_duplicates(subset=['model', 'fold', 'fraction'], keep='last')
    new.to_csv(OUT, index=False)

    print('\n' + '=' * 70)
    print('LEARNING CURVE (AUPRC, scaffold-cluster)')
    print('=' * 70)
    piv = new.pivot_table(index='fraction', columns='model',
                          values='auprc', aggfunc='mean')
    print(piv.round(4).to_string())
    if {'GNEprop', 'Random Forest'} <= set(piv.columns):
        print('\nGNEprop − Random Forest:')
        print((piv['GNEprop'] - piv['Random Forest']).round(4).to_string())
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
