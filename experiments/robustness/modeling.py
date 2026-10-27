"""Train-only transforms, bounded model fits and portable saved predictors."""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.signal import butter, sosfilt, sosfilt_zi, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"baseline/src"))
from csi.features import remove_impulses, window_features
from experiments.models import make_model, parameter_count
from experiments.final_motion.pipeline import preprocess as motion_preprocess

CONFIGS = {
    "threshold_abs":dict(kind="threshold",prep="clean_abs",pca=0),
    "logreg_abs":dict(kind="logreg",prep="clean_abs",pca=0),
    "lightgbm_abs":dict(kind="lightgbm",prep="clean_abs",pca=0),
    "logreg_scaled":dict(kind="logreg",prep="clean_lp_scaled",pca=0),
    "logreg_standardized":dict(kind="logreg",prep="clean_lp_scaled",pca=0,standardize=True),
    "lightgbm_scaled":dict(kind="lightgbm",prep="clean_lp_scaled",pca=0),
    "logreg_pca8":dict(kind="logreg",prep="clean_lp_scaled",pca=8),
    "logreg_pca16":dict(kind="logreg",prep="clean_lp_scaled",pca=16),
    "tcn":dict(kind="tcn",prep="hampel_lp20",pca=0),
    "tcn_pca8":dict(kind="tcn",prep="hampel_lp20",pca=8),
    "tcn_pca16":dict(kind="tcn",prep="hampel_lp20",pca=16),
}
BASE_NAMES=["amp_mean","amp_std_t","var1s","cv1s","amp_range","pc1_band_0_2",
            "pc1_band_2_10","pc1_band_10_40","pc1_logpower"]
SCALED_NAMES=["log_var1s","log_std","log_range","pc1_band_0_2","pc1_band_2_10","pc1_band_10_40","log_pc1_power"]


def prepare(x, name):
    if name=="hampel_lp20":
        return motion_preprocess(x,name)
    out=np.empty_like(x,dtype=np.float32)
    sos=butter(4,20,fs=100,output="sos")
    for i,w in enumerate(x):
        a=remove_impulses(w.T)[0].T
        if name=="clean_lp_scaled":
            zi=sosfilt_zi(sos)[:,None,:]*a[None,:,:1]
            filtered,_=sosfilt(sos,a,axis=-1,zi=zi)
            a=(filtered-filtered.mean(-1,keepdims=True))/max(float(filtered.mean()),1e-12)
        out[i]=a
    return out


def fit_projection(x,k):
    mean=x.mean((0,2),dtype=np.float64)[None,:,None].astype(np.float32)
    std=x.std((0,2),dtype=np.float64).clip(1e-4)[None,:,None].astype(np.float32)
    saved=dict(mean=mean,std=std,components=None,explained_variance_ratio=None)
    if k:
        # All train time samples, covariance accumulation in float64.
        covariance=np.zeros((x.shape[1],x.shape[1]),np.float64)
        count=0
        for batch in np.array_split(x,max(1,int(np.ceil(len(x)/64)))):
            v=((batch-mean)/std).transpose(0,2,1).reshape(-1,x.shape[1]).astype(np.float64)
            covariance+=v.T@v;count+=len(v)
        covariance/=max(count-1,1)
        vals,vecs=np.linalg.eigh(covariance)
        order=np.argsort(vals)[::-1]
        comp=vecs[:,order[:k]].T
        comp*=np.where(comp[np.arange(k),np.abs(comp).argmax(1)]>=0,1,-1)[:,None]
        saved.update(components=comp.astype(np.float32),explained_variance_ratio=(vals[order]/vals.sum()).tolist())
    return saved


def project(x,s):
    z=(x-s["mean"])/s["std"]
    return z if s["components"] is None else np.einsum("kc,nct->nkt",s["components"],z,optimize=True).astype(np.float32)


def features(x,absolute=False):
    out=[]
    for w in x:
        a=w.T
        # Eigh of a small covariance is equivalent to the right singular vector
        # used in baseline.first_pc; the sign drops out of Welch power.
        centered=a-a.mean(0,keepdims=True)
        _,v=np.linalg.eigh(centered.T@centered)
        pc=centered@v[:,-1]
        f,p=welch(pc,fs=100,nperseg=128)
        total=float(p.sum())+1e-12
        sub=a.reshape(4,100,a.shape[1])
        var=float(sub.var(1).mean());std=float(a.std(0).mean());ran=float(np.ptp(a,axis=0).mean())
        bands=[float(p[(f>=lo)&(f<hi)].sum()/total) for lo,hi in [(0,2),(2,10),(10,40)]]
        if absolute:
            mean=float(a.mean());cv=float((sub.var(1)/np.maximum(sub.mean(1),1e-6)**2).mean())
            row=[np.log10(v+1e-9) for v in [mean,std,var,cv,ran]]+bands+[np.log10(total)]
        else:
            row=[np.log10(v+1e-12) for v in [var,std,ran]]+bands+[np.log10(total)]
        out.append(row)
    return np.asarray(out,dtype=np.float64)


def metrics(y,p,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    tpr=float(tp/(tp+fn)) if tp+fn else None
    fpr=float(fp/(tn+fp)) if tn+fp else None
    both=len(np.unique(y))==2
    return dict(n=len(y),positive=int(y.sum()),negative=int((y==0).sum()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        accuracy=float((tn+tp)/len(y)),balanced_accuracy=(tpr+1-fpr)/2 if both else None,
        recall=tpr,fpr=fpr,precision=float(tp/(tp+fp)) if tp+fp else 0.,
        f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        roc_auc=float(roc_auc_score(y,p)) if both else None,
        average_precision=float(average_precision_score(y,p)) if both else None)


def network(channels):
    model=make_model("tcn")
    if channels!=52:
        model.project=nn.Conv1d(channels,32,1)
    return model


def train(name,seed,x,y,groups,path,curve_path,validation=None,existing=None):
    """x and y contain TRAIN only; validation is never used for stopping/selection."""
    path=Path(path)
    if path.exists():
        saved=joblib.load(path)
        assert saved["name"]==name and saved["seed"]==seed and saved["train_groups"]==sorted(set(groups))
        return saved
    cfg=CONFIGS[name]
    saved=dict(name=name,seed=seed,config=cfg,threshold=.5,train_groups=sorted(set(groups)),
               train_n=len(y),source_unit="native amplitude units",epochs=20 if cfg["kind"]=="tcn" else None)
    start=time.perf_counter()
    xp=prepare(x,cfg["prep"])
    if cfg["kind"]=="tcn":
        if existing is not None:
            original=torch.load(existing,map_location="cpu",weights_only=True)
            assert sorted(original["train_recordings"])==saved["train_groups"]
            saved.update(model=original["model"],projection=dict(mean=original["mean"].numpy(),std=original["std"].numpy(),components=None,
                explained_variance_ratio=None),parameters=original["parameters"],
                reused_checkpoint=str(Path(existing).relative_to(ROOT)),train_seconds=original["train_seconds"])
            joblib.dump(saved,path,compress=3)
            torch.save(original,path.with_suffix(".pt"))
            return saved
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        projection=fit_projection(xp,cfg["pca"])
        tx=project(xp,projection)
        device="cuda" if torch.cuda.is_available() else "cpu"
        tx=torch.from_numpy(tx).to(device);ty=torch.from_numpy(y.astype(np.float32)).to(device)
        model=network(tx.shape[1]).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=.00001)
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y==0).sum()/(y==1).sum()],device=device))
        scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
        curves=[]
        vx=vy=None
        if validation is not None:
            raw,vy=validation;vx=torch.from_numpy(project(prepare(raw,cfg["prep"]),projection)).to(device)
        for epoch in range(1,21):
            model.train();loss_total=0.
            for ids in torch.randperm(len(tx),device=device).split(64):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device,dtype=torch.float16,enabled=device=="cuda"):
                    loss=criterion(model(tx[ids]).flatten(),ty[ids])
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();loss_total+=float(loss.detach())*len(ids)
            row=dict(epoch=epoch,train_loss=loss_total/len(tx))
            if vx is not None:
                p=network_predict(model,vx)
                row.update(validation_balanced_accuracy=metrics(vy,p)["balanced_accuracy"])
            curves.append(row);scheduler.step()
        saved.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},projection=projection,
                     parameters=parameter_count(model),optimizer="AdamW lr=.001 wd=.0001; cosine20; clip1; pos_weight=n0/n1")
        pd.DataFrame(curves).to_csv(curve_path,index=False)
        del model,tx,ty,vx,optimizer
        if device=="cuda":torch.cuda.synchronize();torch.cuda.empty_cache()
    else:
        projection=fit_projection(xp,cfg["pca"]) if cfg["pca"] or cfg.get("standardize") else None
        f=features(project(xp,projection) if projection is not None else xp,cfg["prep"]=="clean_abs")
        if cfg["kind"]=="logreg":
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=2000)).fit(f,y)
        elif cfg["kind"]=="lightgbm":
            model=LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=15,min_child_samples=10,
                subsample=.8,subsample_freq=1,colsample_bytree=.8,class_weight="balanced",random_state=seed,verbose=-1,n_jobs=1).fit(f,y)
        else:
            v=f[:,2];unique=np.unique(v)
            candidates=np.r_[unique[0]-1,(unique[:-1]+unique[1:])/2,unique[-1]+1]
            ba=[.5*((v[y==1]>c).mean()+(v[y==0]<=c).mean()) for c in candidates]
            cut=float(candidates[int(np.argmax(ba))]);model=dict(cut=cut,feature=2)
        saved.update(model=model,projection=projection,feature_names=BASE_NAMES if cfg["prep"]=="clean_abs" else SCALED_NAMES)
    saved["fit_wall_seconds"]=time.perf_counter()-start
    joblib.dump(saved,path,compress=3)
    if cfg["kind"]=="tcn":
        # A self-contained .pt includes weights and every fitted transform.
        portable=copy.deepcopy(saved)
        portable["projection"]={k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in saved["projection"].items()}
        torch.save(portable,path.with_suffix(".pt"))
    return saved


@torch.inference_mode()
def network_predict(model,x):
    model.eval()
    return torch.cat([model(b).flatten().sigmoid() for b in x.split(64)]).cpu().numpy()


def predict(saved,x,device=None):
    cfg=saved["config"]
    xp=prepare(x,cfg["prep"])
    if saved["projection"] is not None:xp=project(xp,saved["projection"])
    if cfg["kind"]=="tcn":
        device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        model=network(xp.shape[1]).to(device);model.load_state_dict(saved["model"])
        p=network_predict(model,torch.from_numpy(xp).to(device));del model
        return p
    f=features(xp,cfg["prep"]=="clean_abs")
    if cfg["kind"]=="threshold":
        # Monotone score with an exact 0.5 decision at the train-only cut.
        return 1/(1+np.exp(-(f[:,saved["model"]["feature"]]-saved["model"]["cut"])))
    return saved["model"].predict_proba(f)[:,1]


def bootstrap(y,p,clusters,repeats=2000):
    """Cluster bootstrap retaining both classes; returns percentile intervals."""
    clusters=np.asarray(clusters);unique=np.unique(clusters);rng=np.random.default_rng(20261028)
    counts=[]
    for g in unique:
        mask=clusters==g
        counts.append(confusion_matrix(y[mask],p[mask]>=.5,labels=[0,1]).ravel())
    counts=np.asarray(counts)
    sums=counts[rng.integers(len(unique),size=(repeats,len(unique)))].sum(1)
    tn,fp,fn,tp=sums.T
    keep=((tn+fp)>0)&((tp+fn)>0)
    values=.5*(tn[keep]/(tn[keep]+fp[keep])+tp[keep]/(tp[keep]+fn[keep]))
    return dict(ba_lo=float(np.quantile(values,.025)),ba_hi=float(np.quantile(values,.975)),
                bootstrap_valid=len(values),clusters=len(unique),
                negative_clusters=len(np.unique(clusters[y==0])),positive_clusters=len(np.unique(clusters[y==1]))) if len(values) else {}
