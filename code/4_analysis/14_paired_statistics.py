

import glob
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.width', 200)
SPLITS = ['Random', 'Scaffold', 'Scaffold-Cluster']
BASELINES = ['Random Forest', 'XGBoost', 'ChemProp (D-MPNN)']


def merge_shards(pattern):
    frames = [pd.read_csv(f) for f in glob.glob(pattern) if _nonempty(f)]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=['split', 'fold'], keep='last')


def _nonempty(f):
    try:
        return len(pd.read_csv(f)) > 0
    except Exception:
        return False



gne = merge_shards('results/gneprop_strong_tuned_per_fold*.csv')
gne.to_csv('results/gneprop_strong_tuned_MERGED.csv', index=False)
print('=' * 74)
print('LEAKAGE-FREE GNEprop (merged shards)')
print('=' * 74)
for sp in SPLITS:
    g = gne[gne.split == sp]
    status = 'COMPLETE' if len(g) == 8 else f'PARTIAL {len(g)}/8'
    if len(g):
        print(f'  {sp:17s} AUPRC {g.auprc.mean():.4f}+/-{g.auprc.sem():.4f}  '
              f'AUROC {g.auroc.mean():.3f}  F1 {g.f1.mean():.3f}  '
              f'BEDROC {g.bedroc.mean():.3f}   [{status}]')


master = pd.read_csv('results/MASTER_all_folds.csv')
keep = master[~((master.model == 'GNEprop') & (master.config == 'Tuned'))].copy()
newrows = gne.rename(columns={}).assign(model='GNEprop', config='Tuned')[
    ['fold', 'split', 'auprc', 'auroc', 'f1', 'bedroc', 'model', 'config']]
master2 = pd.concat([keep, newrows], ignore_index=True)
master2.to_csv('results/MASTER_all_folds_leakfree.csv', index=False)


def vec(df, model, config, split):
    g = df[(df.model == model) & (df.config == config) & (df.split == split)]
    return g.sort_values('fold').set_index('fold').auprc


def paired(a, b):
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return None
    x, y = a.loc[common].values, b.loc[common].values
    d = x - y
    return dict(n=len(common), mean=d.mean(), wins=int((d > 0).sum()),
                t=stats.ttest_rel(x, y).pvalue,
                w=stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0,
                dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) else np.nan)


print('\n' + '=' * 74)
print('TABLE 3  — tuned GNEprop row (leakage-free)')
print('=' * 74)
for sp in SPLITS:
    g = gne[gne.split == sp]
    d = vec(master, 'GNEprop', 'Default', sp)
    if len(g):
        delta = g.auprc.mean() - d.mean()
        print(f'  {sp:17s} {d.mean():.3f} -> {g.auprc.mean():.3f} +/- {g.auprc.sem():.3f}  '
              f'(d={delta:+.3f})  AUROC {g.auroc.mean():.3f}  F1 {g.f1.mean():.3f}  '
              f'BEDROC {g.bedroc.mean():.3f}')

print('\n' + '=' * 74)
print('TABLE 4  — OOD degradation, tuned GNEprop (leakage-free)')
print('=' * 74)
r = vec(master2, 'GNEprop', 'Tuned', 'Random')
s = vec(master2, 'GNEprop', 'Tuned', 'Scaffold-Cluster')
if len(r) and len(s):
    common = sorted(set(r.index) & set(s.index))
    a, b = r.loc[common].mean(), s.loc[common].mean()
    print(f'  GNEprop  {a:.3f} -> {b:.3f}   d={b-a:+.3f} ({100*(b-a)/a:+.1f}%)')

print('\n' + '=' * 74)
print('TABLE 5  — GNEprop vs each model, paired (leakage-free)')
print('=' * 74)
for sp in SPLITS:
    gn = vec(master2, 'GNEprop', 'Tuned', sp)
    for m in BASELINES:
        r = paired(gn, vec(master2, m, 'Tuned', sp))
        if r:
            verdict = ('GNEprop superior' if (r['t'] < 0.05 and r['w'] < 0.05)
                       else 'marginal' if r['t'] < 0.10 else 'not separable')
            print(f'  {sp:17s} vs {m:19s} {r["mean"]:+.4f}  {r["wins"]}/{r["n"]}  '
                  f't={r["t"]:.4f}  W={r["w"]:.4f}  dz={r["dz"]:.2f}  [{verdict}]')
    print()

print('=' * 74)
print('TABLE 6  — tuning effect for GNEprop (leakage-free tuned - default)')
print('=' * 74)
for sp in SPLITS:
    t = vec(master2, 'GNEprop', 'Tuned', sp)
    d = vec(master, 'GNEprop', 'Default', sp)
    common = sorted(set(t.index) & set(d.index))
    if len(common) >= 3:
        x, y = t.loc[common].values, d.loc[common].values
        dd = x - y
        print(f'  {sp:17s} {dd.mean():+.4f}  {int((dd>0).sum())}/{len(common)}  '
              f'p={stats.ttest_rel(x, y).pvalue:.4f}')

# ---- ChemProp OOD-aware ----
print('\n' + '=' * 74)
print('ChemProp OOD-aware re-tune (scaffold-cluster)')
print('=' * 74)
try:
    c = pd.read_csv('results/chemprop_ood_tuned_per_fold.csv')
    status = 'COMPLETE' if len(c) == 8 else f'PARTIAL {len(c)}/8'
    print(f'  AUPRC {c.auprc.mean():.4f} +/- {c.auprc.sem():.4f}   [{status}]')
    print(f'  default 0.5652 | in-distribution-tuned 0.5604')
    cd = c.set_index('fold').auprc
    r = paired(cd, vec(master, 'ChemProp (D-MPNN)', 'Default', 'Scaffold-Cluster'))
    if r:
        print(f'  vs default: {r["mean"]:+.4f}  {r["wins"]}/{r["n"]}  t={r["t"]:.4f}')
except FileNotFoundError:
    print('  not available')

# ---- merge prediction ----
preds = [pd.read_csv(f) for f in glob.glob('results/gneprop_strong_tuned_predictions*.csv')
         if _nonempty(f)]
if preds:
    allp = pd.concat(preds, ignore_index=True).drop_duplicates(
        subset=['split', 'fold', 'smiles'], keep='last')
    allp.to_csv('results/gneprop_strong_tuned_predictions_MERGED.csv', index=False)
    print(f'\nMerged predictions: {len(allp)} rows -> '
          f'results/gneprop_strong_tuned_predictions_MERGED.csv')
print('\nSaved results/MASTER_all_folds_leakfree.csv')
