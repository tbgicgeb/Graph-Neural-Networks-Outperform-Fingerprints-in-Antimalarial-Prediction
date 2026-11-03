

import numpy as np
import pandas as pd

SPLIT_ORDER = ['Random', 'Scaffold', 'Scaffold-Cluster']
MODEL_ORDER = ['Random Forest', 'XGBoost', 'ChemProp (D-MPNN)', 'GNEprop']

SOURCES = [
    ('Random Forest',     'Default', 'results/rf_results_per_fold.csv'),
    ('XGBoost',           'Default', 'results/xgb_results_per_fold.csv'),
    ('ChemProp (D-MPNN)', 'Default', 'results/chemprop_results_per_fold.csv'),
    ('GNEprop',           'Default', 'results/gneprop_scratch_results_per_fold.csv'),
    ('Random Forest',     'Tuned',   'results/tuned_rf_results_per_fold.csv'),
    ('XGBoost',           'Tuned',   'results/tuned_xgb_results_per_fold.csv'),
    ('ChemProp (D-MPNN)', 'Tuned',   'results/tuned_chemprop_results_per_fold.csv'),
    ('GNEprop',           'Tuned',   'results/tuned_gneprop_results_per_fold.csv'),
]


GNEPROP_BEDROC = {
    'Default': 'results/gneprop_default_bedroc_metrics.csv',
    'Tuned':   'results/gneprop_tuned_bedroc_metrics.csv',
}

frames = []
for model, config, path in SOURCES:
    d = pd.read_csv(path)
    d = d[['fold', 'split', 'auprc', 'auroc', 'f1', 'bedroc']].copy()
    d['model'] = model
    d['config'] = config

    if model == 'GNEprop':
        bd = pd.read_csv(GNEPROP_BEDROC[config])[['fold', 'split', 'bedroc']]
        d = d.drop(columns='bedroc').merge(bd, on=['fold', 'split'], how='left')

    frames.append(d)

allf = pd.concat(frames, ignore_index=True)
allf.to_csv('results/MASTER_all_folds.csv', index=False)

rows = []
for (model, config, split), g in allf.groupby(['model', 'config', 'split']):
    r = {'model': model, 'config': config, 'split': split, 'n_folds': len(g)}
    for m in ['auprc', 'auroc', 'f1', 'bedroc']:
        r[f'{m}_mean'] = g[m].mean()
        r[f'{m}_sem'] = g[m].sem()
    rows.append(r)

summary = pd.DataFrame(rows)
summary['split'] = pd.Categorical(summary['split'], SPLIT_ORDER, ordered=True)
summary['model'] = pd.Categorical(summary['model'], MODEL_ORDER, ordered=True)
summary = summary.sort_values(['model', 'config', 'split']).reset_index(drop=True)
summary.to_csv('results/MASTER_summary.csv', index=False)

pd.set_option('display.width', 250)
print('=' * 100)
print('MASTER SUMMARY (mean ± SEM, 8-fold CV)')
print('=' * 100)
for _, r in summary.iterrows():
    print(f"{r['model']:19s} {r['config']:8s} {r['split']:17s} "
          f"AUPRC {r.auprc_mean:.3f}±{r.auprc_sem:.3f}  "
          f"AUROC {r.auroc_mean:.3f}±{r.auroc_sem:.3f}  "
          f"F1 {r.f1_mean:.3f}±{r.f1_sem:.3f}  "
          f"BEDROC {r.bedroc_mean:.3f}±{r.bedroc_sem:.3f}")

print('\n' + '=' * 100)
print('OOD DEGRADATION (Random -> Scaffold-Cluster)')
print('=' * 100)
deg = []
for (model, config), g in summary.groupby(['model', 'config'], observed=True):
    g = g.set_index('split')
    a, b = g.loc['Random', 'auprc_mean'], g.loc['Scaffold-Cluster', 'auprc_mean']
    deg.append({'model': model, 'config': config, 'random': a,
                'scaffold_cluster': b, 'delta': b - a, 'pct': 100 * (b - a) / a})
    print(f'{model:19s} {config:8s} {a:.3f} -> {b:.3f}   '
          f'delta {b-a:+.3f} ({100*(b-a)/a:+.1f}%)')
pd.DataFrame(deg).to_csv('results/MASTER_degradation.csv', index=False)

print('\nSaved: results/MASTER_all_folds.csv')
print('Saved: results/MASTER_summary.csv')
print('Saved: results/MASTER_degradation.csv')
