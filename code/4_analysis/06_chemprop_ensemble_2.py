import pandas as pd, numpy as np, pickle, glob, torch, warnings
warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from sklearn.metrics import average_precision_score
from chemprop import data as cpdata, models as cpmodels, featurizers
feat=featurizers.SimpleMoleculeMolGraphFeaturizer()
df=pd.read_csv('malaria_dataset_clean.csv'); smiles=df['smiles'].values; y=df['activity'].values.astype(int)
folds=pickle.load(open('splits/scaffold_cluster_splits.pkl','rb'))
def predict(ck,te):
    ds=cpdata.MoleculeDataset([cpdata.MoleculeDatapoint.from_smi(smiles[i],np.array([0.0])) for i in te],featurizer=feat)
    dl=cpdata.build_dataloader(ds,batch_size=64,num_workers=0,shuffle=False)
    m=cpmodels.MPNN.load_from_checkpoint(ck); m.eval()
    tr=pl.Trainer(accelerator='gpu',devices=1,logger=False,enable_progress_bar=False,enable_model_summary=False)
    try: raw=tr.predict(m,dl,weights_only=False)
    except TypeError: raw=tr.predict(m,dl)
    del m,tr; torch.cuda.empty_cache()
    return np.clip(torch.cat(raw).squeeze(-1).numpy(),0,1)
rows=[]
for fd in folds:
    k=fd['fold']; te=np.asarray(fd['test'])
    mem=sorted(glob.glob(f'chemprop_strong_tmp_f{k}/f{k}/final*/best.ckpt'))
    if len(mem)<2: print(f'fold {k}: <2 members'); continue
    p=[predict(m,te) for m in mem[:2]]
    rows.append(dict(fold=k, auprc_2ens=average_precision_score(y[te], np.mean(p,axis=0))))
    print(f'  fold {k}: ChemProp 2-ens {rows[-1]["auprc_2ens"]:.4f}', flush=True)
r=pd.DataFrame(rows); r.to_csv('results/chemprop_2ensemble.csv',index=False)
print(f'ChemProp 2-model ensemble mean: {r.auprc_2ens.mean():.4f} +/- {r.auprc_2ens.sem():.4f}')
