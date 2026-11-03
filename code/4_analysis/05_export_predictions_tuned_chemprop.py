
import os, gc, pickle, warnings
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import average_precision_score
from chemprop import data as cpdata, nn as cpnn, models as cpmodels, featurizers
warnings.filterwarnings('ignore')
SEED = 42
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values; y = df['activity'].values.astype(int)
cfg = pd.read_csv('results/chemprop_strong_tuned_per_fold.csv')
cfg = cfg[cfg.split == 'Scaffold-Cluster'].set_index('fold')
folds = pickle.load(open('splits/scaffold_cluster_splits.pkl', 'rb'))
AGG = {'sum': cpnn.SumAggregation, 'mean': cpnn.MeanAggregation, 'norm': cpnn.NormAggregation}


def ds(idx, lab=True):
    return cpdata.MoleculeDataset([cpdata.MoleculeDatapoint.from_smi(
        smiles[i], np.array([float(y[i])] if lab else [0.0])) for i in idx], featurizer=feat)


def run(k, fd):
    c = cfg.loc[k]
    tr, va, te = np.asarray(fd['train']), np.asarray(fd['val']), np.asarray(fd['test'])
    bs = int(c.best_batch_size)
    trl = cpdata.build_dataloader(ds(tr), batch_size=bs, num_workers=0, shuffle=True)
    val = cpdata.build_dataloader(ds(va), batch_size=bs, num_workers=0, shuffle=False)
    tel = cpdata.build_dataloader(ds(te, False), batch_size=bs, num_workers=0, shuffle=False)
    pl.seed_everything(SEED, workers=True)
    mp = cpnn.BondMessagePassing(d_h=int(c.best_hidden_size), depth=int(c.best_depth),
                                 dropout=float(c.best_dropout))
    ffn = cpnn.BinaryClassificationFFN(input_dim=int(c.best_hidden_size),
                                       hidden_dim=int(c.best_ffn_hidden),
                                       n_layers=int(c.best_ffn_layers), dropout=float(c.best_dropout))
    model = cpmodels.MPNN(message_passing=mp, agg=AGG[c.best_agg](), predictor=ffn,
                          batch_norm=bool(c.best_batch_norm),
                          metrics=[cpnn.metrics.BinaryAUPRC()])
    ck = ModelCheckpoint(dirpath=f'cp_enr_tmp/f{k}', monitor='val/prc', mode='max', save_top_k=1)
    es = EarlyStopping(monitor='val/prc', patience=8, mode='max')
    tr_ = pl.Trainer(max_epochs=int(c.best_max_epochs), accelerator='gpu', devices=1, logger=False,
                     enable_progress_bar=False, enable_model_summary=False, callbacks=[ck, es])
    tr_.fit(model, trl, val)
    if ck.best_model_path and os.path.exists(ck.best_model_path):
        model = cpmodels.MPNN.load_from_checkpoint(ck.best_model_path)
    model.eval()
    pt = pl.Trainer(accelerator='gpu', devices=1, logger=False, enable_progress_bar=False,
                    enable_model_summary=False)
    try:
        raw = pt.predict(model, tel, weights_only=False)
    except TypeError:
        raw = pt.predict(model, tel)
    p = np.clip(torch.cat(raw).squeeze(-1).numpy(), 0, 1)
    del model, tr_, pt; torch.cuda.empty_cache(); gc.collect()
    return te, p


rows = []
for k, fd in enumerate(folds):
    te, p = run(k, fd)
    print(f'  fold {k}: ChemProp AUPRC {average_precision_score(y[te], p):.3f}', flush=True)
    for idx, sc in zip(te, p):
        rows.append(dict(model='ChemProp (D-MPNN)', split='Scaffold-Cluster', fold=k,
                         y_true=int(y[idx]), y_score=float(sc)))
    pd.DataFrame(rows).to_csv('results/chemprop_tuned_predictions_sc.csv', index=False)
print('saved results/chemprop_tuned_predictions_sc.csv')
