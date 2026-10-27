"""Save audit summaries, paired comparisons and inference resource measurements."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from .datasets import CACHE, OUT, ROOT, write_json
from .modeling import CONFIGS, features, network, prepare, project


def benchmark():
    x=np.load(ROOT/"experiments/motion_results/amplitudes.npy",mmap_mode="r")[:1].copy()
    selection=json.loads((OUT/"locked_selection.json").read_text(encoding="utf-8"))
    rows=[]
    for name in selection["finalists"]:
        path=OUT/"checkpoints"/f"all_wallhack__{name}__s42.joblib"
        saved=joblib.load(path);cfg=saved["config"]
        model=None
        if cfg["kind"]=="tcn":
            model=network(cfg["pca"] or 52).eval();model.load_state_dict(saved["model"])
        times=[]
        for i in range(60):
            start=time.perf_counter()
            z=prepare(x,cfg["prep"])
            if saved["projection"] is not None:z=project(z,saved["projection"])
            with torch.inference_mode():
                if model is not None:model(torch.from_numpy(z)).sigmoid()
                else:
                    f=features(z,cfg["prep"]=="clean_abs")
                    if cfg["kind"]!="threshold":saved["model"].predict_proba(f)
                    else:f[:,2]>saved["model"]["cut"]
            if i>=10:times.append((time.perf_counter()-start)*1000)
        rows.append(dict(name=name,cpu_p50_ms=float(np.median(times)),cpu_p95_ms=float(np.quantile(times,.95)),
            parameters=saved["parameters"] if "parameters" in saved else (1 if cfg["kind"]=="threshold" else len(saved["feature_names"])+1),
            classifier_bytes=path.stat().st_size,pt_bytes=path.with_suffix(".pt").stat().st_size if path.with_suffix(".pt").exists() else None,
            threads=1,batch=1,repeats=50,input_shape="1,52,400",device="CPU",cpu=platform.processor(),
            includes="window preprocessing + trained projection + features + classifier; loaded model; no disk/parsing/resampling"))
    pd.DataFrame(rows).to_csv(OUT/"benchmark.csv",index=False)


def signal_summary():
    rows=[]
    for kind in ("source","days","opera"):
        if kind=="source":
            m=pd.read_csv(ROOT/"experiments/motion_results/windows.csv")
            x=np.load(ROOT/"experiments/motion_results/amplitudes.npy",mmap_mode="r")
        else:
            m=pd.read_csv(OUT/(kind+"_windows.csv"));x=np.load(CACHE/(kind+".npy"),mmap_mode="r")
        for (dataset,domain,target),g in m.groupby(["dataset","domain","target"]):
            ids=g.index.to_numpy();chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(64,len(ids))).round().astype(int))]
            sample=x[chosen]
            norm=(sample-sample.mean(-1,keepdims=True))/np.maximum(sample.mean((1,2),keepdims=True),1e-12)
            f,p=welch(norm,fs=100,nperseg=400,axis=-1)
            total=p.sum(-1).mean(1)
            rows.append(dict(dataset=dataset,domain=domain,target=int(target),windows=len(ids),diagnostic_windows=len(chosen),
                median_amplitude=float(np.median(sample.mean((1,2)))),
                median_relative_std=float(np.median(norm.std(-1).mean(1))),
                median_power_above20=float(np.median(p[...,f>20].sum(-1).mean(1)/np.maximum(total,1e-30)))))
    pd.DataFrame(rows).to_csv(OUT/"signal_summary.csv",index=False)


def inventory_summary():
    rows=[]
    for kind in ("days","opera"):
        m=pd.read_csv(OUT/(kind+"_windows.csv"))
        for domain,g in m.groupby("domain"):
            rows.append(dict(dataset=kind,domain=domain,recordings=g.recording.nunique(),
                experiments=g.experiment.nunique(),negative=int((g.target==0).sum()),positive=int(g.target.sum()),
                background_recordings=g[g.target==0].recording.nunique(),positive_recordings=g[g.target==1].recording.nunique()))
    pd.DataFrame(rows).to_csv(OUT/"cohorts.csv",index=False)
    r=pd.read_csv(OUT/"opera_recordings.csv")
    activity={}
    for s in r.labels:
        for key,value in json.loads(s).items():activity[key]=activity.get(key,0)+value
    pd.DataFrame([dict(activity=k,packets=v) for k,v in sorted(activity.items())]).to_csv(OUT/"opera_activity_inventory.csv",index=False)


def audit_and_matched_tests():
    from .modeling import metrics
    d=pd.read_csv(OUT/"days_recordings.csv");r=pd.read_csv(OUT/"opera_recordings.csv")
    conf=json.loads((ROOT/"experiments/robustness/sources/presence_detection_cnn/day_conf.json").read_text())
    cross={}
    for day,g in d.groupby("day"):
        actual=g.groupby("category").size().to_dict()
        expected={k:v for k,v in conf[day].items() if k not in ("location","mixed_truth")}
        cross[day]=dict(actual=actual,expected=expected,
            counts_match=all(actual.get(k,0)==expected.get(k,0) for k in set(actual)|set(expected)))
    write_json(OUT/"days_label_crosscheck.json",cross)
    people=set()
    for s in r.people:people.update(json.loads(s))
    write_json(OUT/"audit_summary.json",dict(days_archives=d.archive.nunique(),days_recordings=len(d),
        days_frames=int(d.frames.sum()),days_selected_valid_packets=int(d.valid_packets.sum()),
        days_zero_byte_records=d.loc[d.bytes.eq(0),"member"].tolist(),days_mixed_records=int(d.mixed.sum()),
        days_bad_header_resync_steps=int(d.invalid_headers.sum()),opera_files=len(r),opera_rows=int(r.packets.sum()),
        opera_csi_columns=sorted(r.csi_columns.unique().tolist()),opera_selected_records=int((r.windows>0).sum()),
        opera_person_id_labels=sorted(people),opera_duplicate_ms=int(r.duplicate_ms.sum()),
        notes=["MAT tables have no packet bandwidth flags",
               "Published six participants vs seven distinct individual person_id strings: identities not merged",
               "Room2 changes both environment and participant composition"]))
    rows=[]
    selection=json.loads((OUT/"locked_selection.json").read_text(encoding="utf-8"))
    for training in ("all_wallhack","opera_room1"):
        for name in selection["finalists"]:
            for seed in ((17,42,2026) if CONFIGS[name]["kind"] in ("tcn","lightgbm") else (42,)):
                suffix="opera_zero_shot" if training=="all_wallhack" else "opera_room2_test"
                p=pd.read_csv(OUT/f"predictions/{training}__{name}__s{seed}__{suffix}.csv")
                p=p[p.room==2]
                rows.append(dict(training=training,name=name,seed=seed,**metrics(p.target.to_numpy(),p.probability.to_numpy())))
    pd.DataFrame(rows).to_csv(OUT/"opera_matched_room2.csv",index=False)
    p=pd.read_csv(OUT/"predictions/opera_room1__tcn__s42__opera_room2_test.csv")
    p.query("target==0 and prediction==1")[["recording","experiment","receiver","start_s","end_s","probability"]].to_csv(OUT/"opera_background_errors.csv",index=False)


def environment():
    import scipy,sklearn,lightgbm,numba,h5py
    packages={"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,
              "torch":str(torch.__version__),"lightgbm":lightgbm.__version__,"numba":numba.__version__,"h5py":h5py.__version__}
    packages.update(python=sys.version,platform=platform.platform(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,cuda=torch.version.cuda)
    write_json(OUT/"environment.json",packages)


def artifacts():
    rows=[]
    files=list((ROOT/"experiments/robustness").rglob("*"))+list((ROOT/"research/robustness_figures").glob("*.png"))+[ROOT/"research/robustness_report.md"]
    for p in sorted(files):
        if not p.is_file() or "__pycache__" in p.parts or p.name=="artifact_manifest.csv":continue
        rows.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(OUT/"artifact_manifest.csv",index=False)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):benchmark();signal_summary();inventory_summary();audit_and_matched_tests();environment()
    artifacts()
