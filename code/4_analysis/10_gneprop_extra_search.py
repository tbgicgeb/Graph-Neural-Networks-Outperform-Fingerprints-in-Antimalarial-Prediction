import os
import sys
import glob
import json
import pickle
import subprocess

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

GNEPROP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gneprop_pyg.py')
CFG = json.load(open('../gneprop/results/gneprop_best_by_validation.json'))['params']

VARIANTS = {
    'base':               [],
    'molfeat':            ['--use_mol_features', '--mol_features', 'rdkit_normalized'],
    'balanced':           ['--balanced_batch_training'],
    'autoloss':           ['--auto_loss_weight'],
    'molfeat_balanced':   ['--use_mol_features', '--mol_features', 'rdkit_normalized',
                           '--balanced_batch_training'],
    'molfeat_autoloss':   ['--use_mol_features', '--mol_features', 'rdkit_normalized',
                           '--auto_loss_weight'],
    'meanagg_molfeat':    ['--aggr', 'mean', '--use_mol_features', '--mol_features',
                           'rdkit_normalized'],
}

df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].tolist()
y = df['activity'].values


def prep_fold0():
    folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))
    fd = folds[0]
    d = 'hp_expand/f0'
    os.makedirs(d, exist_ok=True)
    tr, va, te = np.asarray(fd['train']), np.asarray(fd['val']), np.asarray(fd['test'])
    allidx = np.concatenate([tr, va, te])
    pd.DataFrame({'SMILES': [smiles[i] for i in allidx],
                  'Activity': y[allidx].astype(int)}).to_csv(f'{d}/combined.csv', index=False)
    n1, n2, n3 = len(tr), len(va), len(te)
    with open(f'{d}/split_indices.pkl', 'wb') as fh:
        pickle.dump([(np.arange(0, n1, dtype=np.int64),
                      np.arange(n1, n1 + n2, dtype=np.int64),
                      np.arange(n1 + n2, n1 + n2 + n3, dtype=np.int64))], fh)
    return d


def run(variant):
    d = prep_fold0()
    log = f'hp_expand/{variant}_log'
    # aggregation may be overridden by the variant; drop CFG aggr if so
    extra = VARIANTS[variant]
    aggr = CFG['aggr']
    if '--aggr' in extra:
        aggr = extra[extra.index('--aggr') + 1]
        extra = [a for i, a in enumerate(extra)
                 if i not in (extra.index('--aggr'), extra.index('--aggr') + 1)]
    cmd = [sys.executable, GNEPROP_SCRIPT,
           '--dataset_path', f'{d}/combined.csv',
           '--split_type', 'index_predetermined',
           '--index_predetermined_file', f'{d}/split_indices.pkl',
           '--log_directory', log, '--exclude_bn_bias', '--all_seed', '0',
           '--lr', str(CFG['lr']), '--hidden_size', str(CFG['hidden_size']),
           '--depth', str(CFG['depth']),
           '--num_readout_layers', str(CFG['num_readout_layers']),
           '--dropout', str(CFG['dropout']), '--aggr', aggr,
           '--max_epochs', str(CFG['max_epochs']),
           '--train_batch_size', str(CFG['train_batch_size']),
           '--weight_decay', str(CFG['weight_decay']),
           '--ig_baseline_ratio', str(CFG['ig_baseline_ratio']),
           '--adv', 'flag', '--adv_m', str(CFG['adv_m']),
           '--lr_strategy', 'warmup_cosine_step', '--metric', 'val_ap',
           '--gpus', '1', '--num_workers', '4'] + extra
    print(f'[{variant}] extra={extra} aggr={aggr}', flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'[{variant}] FAILED: {res.stderr[-400:]}', flush=True)
        return

    best_val = 0.0
    for ev in glob.glob(f'{log}/**/events.out.tfevents*', recursive=True):
        try:
            ea = EventAccumulator(ev); ea.Reload()
            for tag in ('best_val_ap', 'val_ap'):
                if tag in ea.Tags()['scalars']:
                    best_val = max(best_val, max(s.value for s in ea.Scalars(tag)))
        except Exception:
            pass
    pd.DataFrame([dict(variant=variant, best_val_ap=best_val)]).to_csv(
        f'results/hp_expand_{variant}.csv', index=False)
    print(f'[{variant}] fold-0 best_val_ap = {best_val:.4f}', flush=True)


if __name__ == '__main__':
    run(os.environ['VARIANT'])
