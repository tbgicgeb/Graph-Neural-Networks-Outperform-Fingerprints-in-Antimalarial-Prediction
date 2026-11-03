"""
ChemProp (strong config) on the Random and Scaffold splits, 8 folds
each, full data — to complete the tuned/best-effort robustness
"""
import os, gc, pickle, warnings
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import average_precision_score, roc_auc_score
from chemprop import data as cpdata, nn as cpnn, models as cpmodels, featurizers

warnings.filterwarnings('ignore')
SEED = 42
OUT = 'results/chemprop_tuned_other_splits.csv'
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
df = pd.read_csv('malaria_dataset_clean.csv')
smiles = df['smiles'].values
y = df['activity'].values.astype(int)
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
    ck = ModelCheckpoint(dirpath='chemprop_os_tmp', monitor='val/prc', mode='max',
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
    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); done = set(zip(d.split, d.fold)); rows = d.to_dict('records')
    for split_name, pkl in [('Random', 'splits/random_splits.pkl'),
                            ('Scaffold', 'splits/scaffold_splits.pkl')]:
        folds = pickle.load(open(pkl, 'rb'))
        for k, fd in enumerate(folds):
            if (split_name, k) in done:
                continue
            tr, va, te = np.asarray(fd['train']), np.asarray(fd['val']), np.asarray(fd['test'])
            p = train_predict(tr, va, te)
            ap = average_precision_score(y[te], p)
            au = roc_auc_score(y[te], p)
            print(f'  {split_name:9s} fold {k}  AUPRC {ap:.4f}  AUROC {au:.4f}', flush=True)
            rows.append(dict(model='ChemProp (strong-tuned)', split=split_name, fold=k,
                             auprc=ap, auroc=au, n_test=len(te)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print('done; saved', OUT)


if __name__ == '__main__':
    main()
