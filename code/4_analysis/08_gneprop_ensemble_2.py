
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

SPLIT_FILE = 'splits/scaffold_cluster_splits.pkl'
GNEPROP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gneprop_pyg.py')
SEED0_PREDS = 'results/gneprop_strong_tuned_predictions_MERGED.csv'
_SUFFIX = os.environ.get('LC_OUT_SUFFIX', '')
_FOLDS = os.environ.get('LC_FOLDS')
_KEEP = set(int(x) for x in _FOLDS.split(',')) if _FOLDS else None
OUT = f'results/gneprop_ensemble2_per_fold{_SUFFIX}.csv'
LOG_ROOT = 'gneprop_ens2_logs'

CFG = json.load(open('../gneprop/results/gneprop_best_by_validation.json'))['params']

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].tolist()
y = df['activity'].values
seed0 = pd.read_csv(SEED0_PREDS)
seed0 = seed0[seed0.split == 'Scaffold-Cluster']


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


def train_one(fold_dir, n1, n2, n3):
    log = f'{fold_dir}/out_seed1'
    cmd = [sys.executable, GNEPROP_SCRIPT,
           '--dataset_path', f'{fold_dir}/combined.csv',
           '--split_type', 'index_predetermined',
           '--index_predetermined_file', f'{fold_dir}/split_indices.pkl',
           '--log_directory', log, '--exclude_bn_bias',
           '--all_seed', '0', '--ensemble_incr_seed', '1',
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
        print(f'    FAILED: {res.stderr[-300:]}')
        return None, None
    ck = sorted(glob.glob(f'{log}/*/fold_0/version_0/checkpoints/*.ckpt'))
    if not ck:
        return None, None
    from gneprop_pyg import GNEprop
    from data import MolDatasetOD, MoleculeSubset
    full = MolDatasetOD.load_csv_dataset(f'{fold_dir}/combined.csv')
    test_ds = MoleculeSubset(full, list(range(n1 + n2, n1 + n2 + n3)))
    model = GNEprop.load_from_checkpoint(ck[-1]); model.eval()
    with torch.no_grad():
        s = model.predict_data(test_ds, gpus=1, batch_size=200, num_workers=0).flatten()
    sm = np.asarray(test_ds.smiles)
    del model; torch.cuda.empty_cache()
    shutil.rmtree(log, ignore_errors=True)
    return np.asarray(s, dtype=float), sm


def main():
    folds = pickle.load(open(SPLIT_FILE, 'rb'))
    done = set(pd.read_csv(OUT).fold) if os.path.exists(OUT) else set()

    for fd in folds:
        k = fd['fold']
        if (_KEEP is not None and k not in _KEEP) or k in done:
            continue
        fold_dir = f'{LOG_ROOT}/fold{k}'
        os.makedirs(fold_dir, exist_ok=True)
        tr, va, te = np.asarray(fd['train']), np.asarray(fd['val']), np.asarray(fd['test'])
        allidx = np.concatenate([tr, va, te])
        pd.DataFrame({'SMILES': [smiles[i] for i in allidx],
                      'Activity': y[allidx].astype(int)}).to_csv(f'{fold_dir}/combined.csv', index=False)
        n1, n2, n3 = len(tr), len(va), len(te)
        with open(f'{fold_dir}/split_indices.pkl', 'wb') as fh:
            pickle.dump([(np.arange(0, n1, dtype=np.int64),
                          np.arange(n1, n1 + n2, dtype=np.int64),
                          np.arange(n1 + n2, n1 + n2 + n3, dtype=np.int64))], fh)

        print(f'\n=== fold {k}: training seed 1 ===', flush=True)
        s1, sm1 = train_one(fold_dir, n1, n2, n3)
        if s1 is None:
            continue

        # align stored seed-0 predictions to the same compounds by SMILES
        p0 = seed0[seed0.fold == k].set_index('smiles')
        s0 = p0.loc[sm1, 'y_score'].values
        yt = p0.loc[sm1, 'y_true'].values.astype(int)

        single = average_precision_score(yt, s0)
        ens = np.mean([s0, s1], axis=0)
        row = dict(fold=k, split='Scaffold-Cluster', model='GNEprop (2-ens)',
                   auprc_single=single,
                   auprc_seed1=average_precision_score(yt, s1),
                   auprc=average_precision_score(yt, ens),
                   auroc=roc_auc_score(yt, ens),
                   f1=f1_score(yt, (ens >= 0.5).astype(int), zero_division=0),
                   bedroc=bedroc(yt, ens), n_models=2,
                   n_test=len(yt), n_test_actives=int(yt.sum()))
        pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        print(f'  fold {k}: seed0 {single:.4f}  seed1 {row["auprc_seed1"]:.4f}  '
              f'2-ens {row["auprc"]:.4f}', flush=True)
        shutil.rmtree(fold_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
