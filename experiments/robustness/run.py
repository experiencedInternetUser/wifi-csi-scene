"""Bounded source comparison and locked external robustness experiments.

python -m experiments.robustness.run source
python -m experiments.robustness.run external
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .datasets import CACHE, OUT, ROOT, write_json
from .modeling import CONFIGS, bootstrap, metrics, predict, train

SEEDS=(17,42,2026)
DEV=("LOS/BQ","LOS/PIFA","NLOS/BQ")


def setup():
    for name in ["checkpoints","curves","predictions"]:(OUT/name).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


def fit(name,seed,stage,x,m,tr,va=None,reuse=False):
    assert len(np.unique(m.iloc[tr].target))==2
    if va is not None:assert not set(m.iloc[tr].group_id)&set(m.iloc[va].group_id)
    run_id=f"{stage}__{name}__s{seed}"
    existing=None
    if reuse and name=="tcn":
        old=f"{stage}__hampel_lp20__tcn__s{seed}"
        if stage.startswith("cv_"):old=old.replace("cv_","cv_",1)
        existing=ROOT/"experiments/motion_results/checkpoints"/(old+".pt")
        assert existing.exists(),existing
    saved=train(name,seed,x[tr],m.target.to_numpy()[tr],m.group_id.to_numpy()[tr],
        OUT/"checkpoints"/(run_id+".joblib"),OUT/"curves"/(run_id+".csv"),
        validation=None if va is None else (x[va],m.target.to_numpy()[va]),existing=existing)
    write_json(OUT/"checkpoints"/(run_id+".json"),dict(run_id=run_id,config=CONFIGS[name],seed=seed,
        train_indices=tr.tolist(),validation_indices=[] if va is None else va.tolist(),
        train_groups=sorted(set(m.iloc[tr].group_id)),
        validation_groups=[] if va is None else sorted(set(m.iloc[va].group_id)),
        parameters=saved.get("parameters"),feature_names=saved.get("feature_names"),
        explained_variance_ratio=None if saved["projection"] is None else saved["projection"].get("explained_variance_ratio"),
        fit_seconds=saved.get("fit_wall_seconds"),reused=saved.get("reused_checkpoint")))
    return run_id,saved


def evaluate(run_id,saved,x,m,ids,stage,details=True):
    output=OUT/"predictions"/(run_id+"__"+stage+".csv")
    if output.exists():
        previous=pd.read_csv(output)
        assert previous["index"].tolist()==ids.tolist()
        p=previous.probability.to_numpy()
    else:
        p=predict(saved,x[ids])
        frame=m.iloc[ids].copy();frame["index"]=ids;frame["probability"]=p;frame["prediction"]=(p>=.5).astype(int)
        frame.to_csv(output,index=False)
    mm=m.iloc[ids];y=mm.target.to_numpy();rows=[]
    masks={"ALL":np.ones(len(ids),bool)}
    if details:
        masks.update({str(d):mm.domain.to_numpy()==d for d in sorted(mm.domain.unique())})
    for domain,mask in masks.items():
        row=dict(run_id=run_id,name=saved["name"],seed=saved["seed"],stage=stage,domain=domain,
                 train_n=saved["train_n"],**metrics(y[mask],p[mask]))
        if saved["seed"]==42 and len(np.unique(y[mask]))==2:
            clusters=mm.experiment.to_numpy()[mask] if "experiment" in mm else mm.group_id.to_numpy()[mask]
            row.update(bootstrap(y[mask],p[mask],clusters))
        rows.append(row)
    return rows


def source():
    setup()
    meta=pd.read_csv(ROOT/"experiments/motion_results/windows.csv")
    meta["experiment"]=meta.group_id
    x=np.load(ROOT/"experiments/motion_results/amplitudes.npy",mmap_mode="r")
    assert x.shape==(len(meta),52,400) and len(meta)==3052
    dev=(meta.dataset=="Wallhack")&meta.domain.isin(DEV)
    cv=[]
    for held in DEV:
        tr=np.flatnonzero(dev & (meta.domain!=held));va=np.flatnonzero(dev & (meta.domain==held))
        stage="cv_"+held.replace("/","-")
        for name,cfg in CONFIGS.items():
            seeds=SEEDS if cfg["kind"] in ("tcn","lightgbm") else (42,)
            for seed in seeds:
                rid,saved=fit(name,seed,stage,x,meta,tr,va,reuse=True)
                cv.extend(evaluate(rid,saved,x,meta,va,"source_cv",details=False))
                pd.DataFrame(cv).to_csv(OUT/"source_cv.csv",index=False)
                print(rid,'BA',cv[-1]["balanced_accuracy"],flush=True)
    summary=pd.DataFrame(cv).groupby("name").agg(mean_ba=("balanced_accuracy","mean"),
        std_ba=("balanced_accuracy","std"),min_ba=("balanced_accuracy","min"),fits=("balanced_accuracy","size")).reset_index()
    summary.to_csv(OUT/"source_selection.csv",index=False)
    # The PCA option remains a separate finalist, even if it loses to no PCA.
    def best(names):return summary[summary.name.isin(names)].sort_values(["mean_ba","name"],ascending=[False,True]).iloc[0]["name"]
    chosen=dict(classic=best(["logreg_abs","lightgbm_abs","logreg_scaled","lightgbm_scaled"]),
        lightweight_pca=best(["logreg_pca8","logreg_pca16"]),tcn_pca=best(["tcn_pca8","tcn_pca16"]))
    finalists=list(dict.fromkeys(["tcn",chosen["classic"],"logreg_scaled",chosen["lightweight_pca"],chosen["tcn_pca"],"threshold_abs"]))
    write_json(OUT/"locked_selection.json",dict(**chosen,finalists=finalists,
        selection="highest mean development-domain BA; PCA candidates kept for ablation; no target-domain tuning",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        baseline_commit=subprocess.check_output(["git","rev-parse","origin/baseline"],text=True).strip()))
    results=[]
    for stage,tr,tests in [
        ("domain_holdout",np.flatnonzero(dev),[("wallhack_holdout",np.flatnonzero((meta.dataset=="Wallhack")&~dev)),("3do_holdout",np.flatnonzero(meta.dataset=="3DO"))]),
        ("all_wallhack",np.flatnonzero(meta.dataset=="Wallhack"),[("3do_final",np.flatnonzero(meta.dataset=="3DO"))])]:
        for name in finalists:
            seeds=SEEDS if CONFIGS[name]["kind"] in ("tcn","lightgbm") else (42,)
            for seed in seeds:
                rid,saved=fit(name,seed,stage,x,meta,tr,reuse=True)
                for test,ids in tests:results.extend(evaluate(rid,saved,x,meta,ids,test))
                pd.DataFrame(results).to_csv(OUT/"source_tests.csv",index=False)
                print(rid,'source tests complete',flush=True)


def external(datasets=("days","opera"),retrain=True):
    setup()
    selection=json.loads((OUT/"locked_selection.json").read_text(encoding="utf-8"))
    names=selection["finalists"]
    zero=[]
    for dataset in datasets:
        m=pd.read_csv(OUT/(dataset+"_windows.csv"));x=np.load(CACHE/(dataset+".npy"),mmap_mode="r")
        for name in names:
            seeds=SEEDS if CONFIGS[name]["kind"] in ("tcn","lightgbm") else (42,)
            for seed in seeds:
                rid=f"all_wallhack__{name}__s{seed}"
                saved=joblib.load(OUT/"checkpoints"/(rid+".joblib"))
                zero.extend(evaluate(rid,saved,x,m,np.arange(len(m)),dataset+"_zero_shot"))
                pd.DataFrame(zero).to_csv(OUT/"external_zero_shot.csv",index=False)
                print(rid,dataset,'zero-shot complete',flush=True)
    if not retrain:return
    m=pd.read_csv(OUT/"opera_windows.csv");x=np.load(CACHE/"opera.npy",mmap_mode="r")
    # exp019 is an independent background run; exp020..027 are validation
    # activity experiments. Room2 is entirely excluded from train/validation.
    exp=m.experiment.str.extract(r"(\d+)")[0].astype(int)
    train_mask=(m.room==1)&(exp<=18)
    val_mask=(m.room==1)&(exp>=19)&(exp<=27)
    test_mask=m.room==2
    tr,va,te=map(np.flatnonzero,(train_mask,val_mask,test_mask))
    assert len(tr)>0 and len(va)>0 and len(te)>0
    assert not set(m.iloc[tr].experiment)&set(m.iloc[te].experiment)
    assert not set(m.iloc[tr].experiment)&set(m.iloc[va].experiment)
    assert not set(m.iloc[va].experiment)&set(m.iloc[te].experiment)
    write_json(OUT/"opera_split.json",dict(train_indices=tr.tolist(),validation_indices=va.tolist(),test_indices=te.tolist(),
        train_experiments=sorted(set(m.iloc[tr].experiment)),validation_experiments=sorted(set(m.iloc[va].experiment)),
        test_experiments=sorted(set(m.iloc[te].experiment)),
        train_domains=sorted(set(m.iloc[tr].domain)),test_domains=sorted(set(m.iloc[te].domain)),
        rule="train exp001..018 Room1; validation exp019..027 Room1; test all Room2; frozen 20 epochs"))
    results=[]
    for name in names:
        seeds=SEEDS if CONFIGS[name]["kind"] in ("tcn","lightgbm") else (42,)
        for seed in seeds:
            rid,saved=fit(name,seed,"opera_room1",x,m,tr,va)
            results.extend(evaluate(rid,saved,x,m,va,"opera_validation"))
            results.extend(evaluate(rid,saved,x,m,te,"opera_room2_test"))
            pd.DataFrame(results).to_csv(OUT/"opera_retrained.csv",index=False)
            print(rid,'Room2 test complete',flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["source","external","days_zero"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=="days_zero":external(("days",),False)
        else:globals()[args.stage]()
