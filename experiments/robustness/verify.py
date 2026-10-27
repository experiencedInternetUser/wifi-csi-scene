"""Independent checks of binary decoding, feature equivalence and split integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .datasets import HEADER, OUT, ROOT, CACHE, ath_decode, write_json
from .modeling import BASE_NAMES, features, fit_projection, prepare, predict


def parser_check():
    rng=np.random.default_rng(234)
    iq=rng.integers(-511,512,size=(56,3,3,2),dtype=np.int16)
    iq[iq==0]=1
    vals=iq.reshape(-1)
    bits=((vals.astype(np.int32)&1023)[:,None]>>np.arange(10))&1
    payload=np.packbits(bits.ravel(),bitorder="little").tobytes()
    samples=[]
    for extra in (b"",b"testpayload"*4):
        header=HEADER.pack(25+len(payload)+len(extra),123456789,len(payload),2437,0,0,144,0,56,3,3,40,30,30,30,len(extra))
        # Also include a legitimate frame without CSI and zero antenna counts.
        nocsi=HEADER.pack(25,123456780,0,2437,0,0,144,0,0,0,0,40,30,30,30,0)
        blob=b"\x00\x00"+nocsi+header+payload+extra
        t,a,n,bad,pad,tail,_,_=ath_decode(np.frombuffer(blob,np.uint8))
        expected=np.hypot(iq[:,0,0,0].astype(float),iq[:,0,0,1].astype(float))
        assert n==2 and len(t)==1 and t[0]==123456789 and bad==0 and tail==0
        np.testing.assert_allclose(a[0],expected,rtol=1e-6)
        samples.append(dict(payload_bytes=len(extra),max_amplitude_error=float(np.max(np.abs(a[0]-expected)))))
    return samples


def feature_check():
    from csi.features import window_features
    x=np.load(ROOT/"experiments/motion_results/amplitudes.npy",mmap_mode="r")
    ids=np.linspace(0,len(x)-1,12).round().astype(int)
    clean=prepare(x[ids],"clean_abs")
    observed=features(clean,absolute=True)
    reference=[]
    for w in clean:
        f=window_features(w.T)
        row=[f[k] for k in BASE_NAMES]
        for j in range(5):row[j]=np.log10(row[j]+1e-9)
        reference.append(row)
    reference=np.asarray(reference)
    np.testing.assert_allclose(observed,reference,atol=3e-5,rtol=3e-5)
    s=fit_projection(prepare(x[ids[:6]],"hampel_lp20"),8)
    np.testing.assert_allclose(s["components"]@s["components"].T,np.eye(8),atol=1e-6)
    return dict(windows=len(ids),max_feature_error=float(np.abs(observed-reference).max()),
                pca_orthogonality_error=float(np.abs(s["components"]@s["components"].T-np.eye(8)).max()))


def final_checks():
    result={}
    for kind in ("days","opera"):
        m=pd.read_csv(OUT/(kind+"_windows.csv"));x=np.load(CACHE/(kind+".npy"),mmap_mode="r")
        assert x.shape==(len(m),52,400) and np.isfinite(x).all() and (x>=0).all()
        assert not m.duplicated(["recording","start_s"]).any()
        for _,g in m.groupby("recording"):
            starts=np.sort(g.start_s.to_numpy())
            assert (np.diff(starts)>=4-1e-6).all()
        counts=m.groupby(["recording","target"]).size()
        assert counts.max()<=24
        assert set(m.original_label)<=({"empty","motion","bedroomI","bedroomII","kitchen","living_room"} if kind=="days" else {"background","walk"})
        result[kind]=dict(windows=len(m),recordings=m.recording.nunique(),shape=list(x.shape),nonoverlap=True)
    split=json.loads((OUT/"opera_split.json").read_text(encoding="utf-8"))
    m=pd.read_csv(OUT/"opera_windows.csv")
    groups={k:set(m.iloc[split[k+"_indices"]].experiment) for k in ("train","validation","test")}
    assert not groups["train"]&groups["validation"] and not groups["train"]&groups["test"] and not groups["validation"]&groups["test"]
    assert set(m.iloc[split["test_indices"]].room)=={2}
    result["opera_group_disjoint"]=True
    # A saved model must reproduce its stored predictions after reload.
    x=np.load(CACHE/"opera.npy",mmap_mode="r")
    errors={}
    for path in sorted((OUT/"checkpoints").glob("opera_room1*__s42.joblib")):
        saved=joblib.load(path)
        rows=pd.read_csv(OUT/"predictions"/(path.stem+"__opera_room2_test.csv"))
        sample=rows.iloc[:5]
        p=predict(saved,x[sample["index"].to_numpy()])
        error=float(np.max(np.abs(p-sample.probability.to_numpy())))
        assert error<2e-5,(path,error)
        errors[path.stem]=error
    result["reload_probability_errors"]=errors
    # Every fitted transform has the same train groups recorded in its sidecar.
    checked=0
    for path in (OUT/"checkpoints").glob("*.joblib"):
        saved=joblib.load(path);side=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert saved["train_groups"]==side["train_groups"]
        assert not set(side["train_groups"])&set(side["validation_groups"])
        checked+=1
    result["checked_checkpoints"]=checked
    from experiments.final_motion.pipeline import MotionDetector
    original=MotionDetector(ROOT/"experiments/motion_results/deployment.pt")
    raw=np.load(ROOT/"experiments/motion_results/amplitudes.npy",mmap_mode="r")
    ids=np.linspace(0,len(raw)-1,12).round().astype(int)
    saved=joblib.load(OUT/"checkpoints/all_wallhack__tcn__s42.joblib")
    error=float(np.max(np.abs(original.predict(raw[ids])-predict(saved,raw[ids],device="cpu"))))
    assert error<2e-5,error
    result["original_deployment_probability_error"]=error
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--final",action="store_true")
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result=dict(binary_parser=parser_check(),features=feature_check())
        if args.final:result.update(final_checks())
    write_json(OUT/("verification.json" if args.final else "reader_verification.json"),result)
    print(json.dumps(result,indent=2))
