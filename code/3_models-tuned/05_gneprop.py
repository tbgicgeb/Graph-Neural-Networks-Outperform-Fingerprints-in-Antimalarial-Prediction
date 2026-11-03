"""
GNEprop tuned evaluation.


    conda activate gneprop
    export GNEPROP_DIR=/path/to/your/gneprop/clone
    python code/3_models_tuned/05_gneprop.py
"""

import os
import sys
import glob
import json
import pickle
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


_ALL_SPLITS = {'Scaffold-Cluster': 'splits/scaffold_cluster_splits.pkl',
               'Scaffold': 'splits/scaffold_splits.pkl',
               'Random': 'splits/random_splits.pkl'}
_sel = os.environ.get('LC_SPLITS')
SPLITS = ({k: _ALL_SPLITS[k] for k in _sel.split(',')} if _sel
          else _ALL_SPLITS)

GNEPROP_DIR = os.path.expanduser(
    os.environ.get('GNEPROP_DIR', '~/gneprop'))
GNEPROP_SCRIPT = os.path.join(GNEPROP_DIR, 'gneprop_pyg.py')

_SUFFIX = os.environ.get('LC_OUT_SUFFIX', '')
OUT = f'results/gneprop_strong_tuned_per_fold{_SUFFIX}.csv'
PRED_OUT = f'results/gneprop_strong_tuned_predictions{_SUFFIX}.csv'
LOG_ROOT = 'gneprop_strong_logs'


def _all_done_folds():
    """Folds already recorded across every shard, so workers never repeat one."""
    import glob as _glob
    seen = set()
    for f in _glob.glob('results/gneprop_strong_tuned_per_fold*.csv'):
        try:
            d = pd.read_csv(f)
            seen |= set(zip(d.split, d.fold))
        except Exception:
            pass
    return seen

CFG = json.load(open('../gneprop/results/gneprop_best_by_validation.json'))['params']

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].tolist()
y = df['activity'].values


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


def done(split, fold):
    return (split, fold) in _all_done_folds()


def run_fold(split, k, fold):
    work = f'{LOG_ROOT}/{split}/fold{k}'
    os.makedirs(work, exist_ok=True)
    tr, va, te = np.asarray(fold['train']), np.asarray(fold['val']), np.asarray(fold['test'])
    allidx = np.concatenate([tr, va, te])
    pd.DataFrame({'SMILES': [smiles[i] for i in allidx],
                  'Activity': y[allidx].astype(int)}).to_csv(f'{work}/combined.csv', index=False)
    n1, n2, n3 = len(tr), len(va), len(te)
    with open(f'{work}/split_indices.pkl', 'wb') as fh:
        pickle.dump([(np.arange(0, n1, dtype=np.int64),
                      np.arange(n1, n1 + n2, dtype=np.int64),
                      np.arange(n1 + n2, n1 + n2 + n3, dtype=np.int64))], fh)

    log = f'{work}/gneprop_output'
    cmd = [sys.executable, GNEPROP_SCRIPT,
           '--dataset_path', f'{work}/combined.csv',
           '--split_type', 'index_predetermined',
           '--index_predetermined_file', f'{work}/split_indices.pkl',
           '--log_directory', log, '--exclude_bn_bias',
           '--lr', str(CFG['lr']), '--hidden_size', str(CFG['hidden_size']),
           '--depth', str(CFG['depth']),
           '--num_readout_layers', str(CFG['num_readout_layers']),
           '--dropout', str(CFG['dropout']), '--aggr', CFG['aggr'],
           '--max_epochs', str(CFG['max_epochs']),
           '--train_batch_size', str(CFG['train_batch_size']),
           '--weight_decay', str(CFG['weight_decay']),
           '--ig_baseline_ratio', str(CFG['ig_baseline_ratio']),
           '--adv', 'flag', '--adv_m', str(CFG['adv_m']),
           '--lr_strategy', 'warmup_cosine_step', '--metric', 'val_ap',
           '--gpus', '1', '--num_workers', '0']
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'    FAILED rc={res.returncode}: {res.stderr[-400:]}')
        return None

    ckpts = sorted(glob.glob(f'{log}/*/fold_0/version_0/checkpoints/*.ckpt'),
                   key=lambda p: p.split('gneprop_output/')[1])
    if not ckpts:
        print('    no checkpoint'); return None

    from gneprop_pyg import GNEprop
    from data import MolDatasetOD, MoleculeSubset
    full = MolDatasetOD.load_csv_dataset(f'{work}/combined.csv')
    test_ds = MoleculeSubset(full, list(range(n1 + n2, n1 + n2 + n3)))
    model = GNEprop.load_from_checkpoint(ckpts[-1]); model.eval()
    with torch.no_grad():
        s = model.predict_data(test_ds, gpus=1, batch_size=200, num_workers=0).flatten()
    yt = np.asarray(test_ds.y, dtype=int).flatten()
    sm = np.asarray(test_ds.smiles)
    del model; torch.cuda.empty_cache()
    shutil.rmtree(log, ignore_errors=True)

    return dict(scores=s, labels=yt, smiles=sm)


def main():
    print('Leakage-free GNEprop config (selected on validation, trial 23):')
    print(' ', CFG)
    fold_filter = os.environ.get('LC_FOLDS')
    keep = set(int(x) for x in fold_filter.split(',')) if fold_filter else None
    for split, path in SPLITS.items():
        folds = pickle.load(open(path, 'rb'))
        for fd in folds:
            k = fd['fold']
            if keep is not None and k not in keep:
                continue
            if done(split, k):
                print(f'  {split} fold {k}: already done'); continue
            print(f'  {split} fold {k} ...', flush=True)
            r = run_fold(split, k, fd)
            if r is None:
                continue
            s, yt = r['scores'], r['labels']
            row = dict(fold=k, split=split, model='GNEprop (strong-tuned)',
                       auprc=average_precision_score(yt, s),
                       auroc=roc_auc_score(yt, s),
                       f1=f1_score(yt, (s >= 0.5).astype(int), zero_division=0),
                       bedroc=bedroc(yt, s),
                       n_test=len(yt), n_test_actives=int(yt.sum()))
            pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            pd.DataFrame(dict(split=split, fold=k, smiles=r['smiles'],
                              y_true=yt, y_score=s)).to_csv(
                PRED_OUT, mode='a', header=not os.path.exists(PRED_OUT), index=False)
            print(f'    AUPRC {row["auprc"]:.4f}  BEDROC {row["bedroc"]:.4f}', flush=True)

    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        print('\n=== leakage-free tuned GNEprop (mean +/- SEM) ===')
        for sp in ['Random', 'Scaffold', 'Scaffold-Cluster']:
            g = d[d.split == sp]
            if len(g):
                print(f'  {sp:17s} AUPRC {g.auprc.mean():.3f}+/-{g.auprc.sem():.3f}  '
                      f'(n={len(g)})  BEDROC {g.bedroc.mean():.3f}')




if __name__ == '__main__':
    main()
