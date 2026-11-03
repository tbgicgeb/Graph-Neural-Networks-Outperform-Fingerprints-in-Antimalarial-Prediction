

import os
import sys
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gneprop_pyg import GNEprop
from data import MolDatasetOD, MoleculeSubset

SPLIT_DIRS = {'random': 'Random',
              'scaffold': 'Scaffold',
              'scaffold_cluster': 'Scaffold-Cluster'}
N_FOLDS = 8


def bedroc(y_true, y_score, alpha=20.0):
    """Identical to the implementation used for the other models."""
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


def newest_checkpoint(fold_dir):
    """Latest run directory for this fold wins (folds re-run during debugging)."""
    ckpts = glob.glob(os.path.join(
        fold_dir, 'gneprop_output', '*', 'fold_0', 'version_0',
        'checkpoints', '*.ckpt'))
    if not ckpts:
        return None
    
    return sorted(ckpts, key=lambda p: p.split('gneprop_output/')[1])[-1]


def test_indices(fold_dir):
    with open(os.path.join(fold_dir, 'split_indices.pkl'), 'rb') as fh:
        splits = pickle.load(fh)
    return splits[0][2]


def run_fold(fold_dir, batch_size, num_workers):
    ckpt = newest_checkpoint(fold_dir)
    if ckpt is None:
        return None, 'no checkpoint'

    full = MolDatasetOD.load_csv_dataset(os.path.join(fold_dir, 'combined.csv'))
    test_idx = test_indices(fold_dir)
    test_ds = MoleculeSubset(full, list(test_idx))

    model = GNEprop.load_from_checkpoint(ckpt)
    model.eval()
    with torch.no_grad():
        scores = model.predict_data(test_ds, gpus=1,
                                    batch_size=batch_size,
                                    num_workers=num_workers).flatten()

    y = np.asarray(test_ds.y, dtype=int).flatten()
    scores = np.asarray(scores, dtype=float).flatten()

    del model
    torch.cuda.empty_cache()

    return dict(
        auprc=average_precision_score(y, scores),
        auroc=roc_auc_score(y, scores),
        bedroc=bedroc(y, scores),
        n_test=len(y),
        n_test_actives=int(y.sum()),
        scores=scores,
        labels=y,
        smiles=np.asarray(test_ds.smiles),
        checkpoint=ckpt,
    ), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=['default', 'tuned'], required=True)
    ap.add_argument('--root', required=True,
                    help='gneprop_logs (default) or gneprop_tuned_eval_logs (tuned)')
    ap.add_argument('--reference', required=True,
                    help='original per-fold CSV, for cross-checking AUPRC')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--batch-size', type=int, default=200)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)

    rows, preds = [], []
    for split_dir, split_name in SPLIT_DIRS.items():
        for k in range(N_FOLDS):
            fold_dir = os.path.join(args.root, split_dir, f'fold{k}')
            if not os.path.isdir(fold_dir):
                print(f'  [skip] {split_name} fold {k}: directory missing')
                continue

            res, err = run_fold(fold_dir, args.batch_size, args.num_workers)
            if err:
                print(f'  [skip] {split_name} fold {k}: {err}')
                continue

            prev = ref[(ref['split'] == split_name) & (ref['fold'] == k)]
            prev_auprc = float(prev['auprc'].iloc[0]) if len(prev) else np.nan
            delta = res['auprc'] - prev_auprc

            flag = '' if abs(delta) < 0.02 else '   <-- MISMATCH'
            print(f'  {split_name:17s} fold {k}: '
                  f'AUPRC {res["auprc"]:.4f} (recorded {prev_auprc:.4f}, '
                  f'd={delta:+.4f})  BEDROC {res["bedroc"]:.4f}{flag}')

            rows.append(dict(fold=k, split=split_name,
                             model=f'GNEprop ({args.config})',
                             auprc=res['auprc'], auroc=res['auroc'],
                             bedroc=res['bedroc'],
                             auprc_recorded=prev_auprc, auprc_delta=delta,
                             n_test=res['n_test'],
                             n_test_actives=res['n_test_actives'],
                             checkpoint=res['checkpoint']))

            preds.append(pd.DataFrame(dict(
                fold=k, split=split_name, smiles=res['smiles'],
                y_true=res['labels'], y_score=res['scores'])))

    out = pd.DataFrame(rows)
    out.to_csv(f'{args.out_prefix}_metrics.csv', index=False)
    pd.concat(preds).to_csv(f'{args.out_prefix}_predictions.csv', index=False)

    print('\n' + '=' * 70)
    print(f'GNEprop ({args.config}) — recovered BEDROC')
    print('=' * 70)
    for split_name in SPLIT_DIRS.values():
        g = out[out['split'] == split_name]
        if not len(g):
            continue
        print(f'  {split_name:17s} '
              f'AUPRC {g.auprc.mean():.3f}±{g.auprc.sem():.3f}   '
              f'AUROC {g.auroc.mean():.3f}±{g.auroc.sem():.3f}   '
              f'BEDROC {g.bedroc.mean():.3f}±{g.bedroc.sem():.3f}')
    print(f'\nMax |AUPRC delta| vs recorded: {out.auprc_delta.abs().max():.4f}')
    print(f'Saved: {args.out_prefix}_metrics.csv')
    print(f'Saved: {args.out_prefix}_predictions.csv')


if __name__ == '__main__':
    main()
