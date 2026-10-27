"""Bounded, timestamp-aware audit and preparation of the external CSI archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/robustness/results"
CACHE = ROOT / "tmp/robustness_cache"
sys.path.insert(0, str(ROOT / "tmp/robustness_deps"))
CAP = 24
HEADER = struct.Struct("<HQHHBBBBBBBBBBBH")
ATH_K = np.r_[-28:0, 1:29]
ESP_K = np.r_[-26:0, 1:27]
# 30 Intel groups: interpolation uses normalized ordinal positions in each half.
# Actual group indices are not stored in OPERAnet MAT tables; this approximation
# is explicit and is not a claim of measured values at 52 physical tones.
INTEL_POS = np.r_[np.linspace(-28, -1, 15), np.linspace(1, 28, 15)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@njit(cache=True)
def ath_decode(buf):
    nmax = len(buf) // 400 + 1
    amp = np.empty((nmax, 56), np.float32)
    times = np.empty(nmax, np.int64)
    channels = np.zeros(7000, np.int64)
    bws = np.zeros(256, np.int64)
    off = count = frames = bad = padding = 0
    while off + 27 <= len(buf):
        length = int(buf[off]) + (int(buf[off + 1]) << 8)
        if length == 0:
            off += 2
            padding += 2
            continue
        clen = int(buf[off+10]) + (int(buf[off+11]) << 8)
        channel = int(buf[off+12]) + (int(buf[off+13]) << 8)
        bw, tones, nr, nc = [int(buf[off+j]) for j in (17, 18, 19, 20)]
        end = off + length + 2
        if end > len(buf):
            break
        if not (2000 <= channel <= 6500 and (clen==0 or (tones in (56,114) and 1<=nr<=3 and 1<=nc<=3))):
            bad += 1
            off += 2
            continue
        frames += 1
        channels[channel] += 1
        bws[bw] += 1
        if channel==2437 and bw==0 and tones == 56 and clen == nr*nc*tones*20//8 and clen > 0:
            tm = np.int64(0)
            for j in range(8):
                tm |= np.int64(buf[off+2+j]) << (8*j)
            valid = True
            for k in range(56):
                vals = np.zeros(2, np.int32)
                for q in range(2):
                    bit = (2*k*nr*nc+q)*10
                    pos = off+27+bit//8
                    word = int(buf[pos]) | (int(buf[pos+1]) << 8)
                    if pos+2 < off+27+clen:
                        word |= int(buf[pos+2]) << 16
                    v = (word >> (bit % 8)) & 1023
                    vals[q] = v-1024 if v >= 512 else v
                a = np.sqrt(float(vals[0])**2 + float(vals[1])**2)
                amp[count, k] = a
                if a < 1:
                    valid = False
            if valid:
                times[count] = tm
                count += 1
        off = end
    return times[:count], amp[:count], frames, bad, padding, len(buf)-off, channels, bws


def unwrap_us(t):
    out = t.copy()
    if len(t) > 1:
        wraps = np.r_[0, np.cumsum(np.diff(t) < -(2**31))]
        out += wraps * 2**32
    return out


def make_windows(t, a, label, *, trim=5., cap=CAP):
    """Return bounded windows from one constant-label run, in native channel count."""
    t = np.asarray(t, np.float64)
    if len(t) < 2:
        return [], [], {}
    t = t - t[0]
    dt = np.diff(t)
    diagnostic = dict(packets=len(t), duration_s=float(t[-1]),
                      nonpositive_dt=int((dt <= 0).sum()), gaps_gt_50ms=int((dt > .05+1e-9).sum()),
                      median_dt_ms=float(np.median(dt[dt > 0])*1000) if (dt > 0).any() else 0.)
    bounds = np.r_[0, np.flatnonzero((dt <= 0) | (dt > .05+1e-9))+1, len(t)]
    all_windows, starts = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi-lo < 2:
            continue
        tt, aa = t[lo:hi], a[lo:hi]
        keep = (tt >= trim) & (tt <= t[-1]-trim)
        tt, aa = tt[keep], aa[keep]
        if len(tt) < 2 or tt[-1]-tt[0] < 3.99:
            continue
        median_dt = np.median(np.diff(tt))
        # A common 1 kHz intermediate grid prevents aliasing before 100 Hz
        # decimation. The 40 Hz filter is applied at the actual intermediate fs.
        fs = 1000. if median_dt < .005 else 100.
        grid = tt[0] + np.arange(int(np.floor((tt[-1]-tt[0])*fs))+1)/fs
        v = np.stack([np.interp(grid, tt, aa[:, c]) for c in range(aa.shape[1])], 1)
        if fs > 100:
            v = sosfiltfilt(butter(6, 40, fs=fs, output="sos"), v, axis=0)[::10]
            v = np.maximum(v, 0.)
        for first in range(0, len(v)-399, 400):
            all_windows.append(v[first:first+400].T.astype(np.float32))
            starts.append(float(tt[0]+first/100))
    diagnostic["eligible_windows"] = len(starts)
    ids = np.unique(np.linspace(0, len(starts)-1, min(cap, len(starts))).round().astype(int)) if starts else []
    return [all_windows[i] for i in ids], [starts[i] for i in ids], diagnostic


def days():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    conf = json.loads((Path(__file__).parent / "sources/presence_detection_cnn/day_conf.json").read_text())
    rows, records, inventory, arrays = [], [], [], []
    for path in sorted((ROOT/"datasets").glob("day*.zip"), key=lambda p: int(p.stem[3:])):
        day = path.stem
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                if not member.filename.endswith(".data"):
                    continue
                name = Path(member.filename).stem
                category = re.sub(r"\d+$", "", name)
                location = conf[day]["location"]
                inv = dict(archive=path.name, member=member.filename, bytes=member.file_size, crc32=member.CRC,
                           day=day, domain=location, category=category, mixed=category == "mixed")
                inventory.append(inv)
                buf = np.frombuffer(z.read(member), np.uint8)
                t, a, frames, bad, padding, tail, chans, bws = ath_decode(buf)
                del buf
                rec = {**inv, "frames": frames, "valid_packets": len(t), "invalid_headers": bad,
                       "zero_padding_bytes": padding, "trailing_bytes": tail,
                       "channels_MHz": json.dumps(np.flatnonzero(chans).tolist()),
                       "bandwidth_codes": json.dumps(np.flatnonzero(bws).tolist())}
                if category == "mixed":
                    rec["windows"] = 0
                    rec["exclusion"] = "mixed run: interval boundaries not used in primary test"
                    records.append(rec)
                    continue
                target = int(category != "empty")
                t = unwrap_us(t)
                wins, starts, diag = make_windows(t.astype(float)/1e6, a, target)
                for w, start in zip(wins, starts):
                    arrays.append(w[np.isin(ATH_K, ESP_K)])
                    rows.append(dict(dataset="Days", domain=location, day=day, recording=member.filename,
                                     group_id=f"Days::{member.filename}", experiment=day+"/"+name,
                                     target=target, original_label=category, start_s=start, end_s=start+3.99,
                                     antenna="tx1rx1", band="2.4/20", adapter="56_to_52_drop_outer2"))
                records.append({**rec, **diag, "windows":len(wins), "exclusion":""})
                print(day, name, len(t), 'packets,',len(wins),'windows',flush=True)
        pd.DataFrame(records).to_csv(OUT/"days_recordings.csv",index=False)
    x=np.stack(arrays)
    np.save(CACHE/"days.npy", x)
    pd.DataFrame(rows).to_csv(OUT/"days_windows.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"days_inventory.csv",index=False)
    write_json(OUT/"days_preparation.json",dict(shape=list(x.shape), cap_per_recording=CAP,
        input="Atheros signed 10-bit imag then real, Fortran rx/tx/tone ordering; tx1rx1",
        all_recordings=len(records), selected_recordings=len(set(r["recording"] for r in rows)),
        missing_days=[17,18,19], cache_sha256=hashlib.sha256((CACHE/"days.npy").read_bytes()).hexdigest()))


def matlab_strings(dataset):
    """Decode MATLAB MCOS string storage, validating lengths and dimensions."""
    a = dataset[()].ravel()
    assert a.dtype == np.uint64 and int(a[0]) == 1 and int(a[1]) == 2
    n = int(a[2])*int(a[3])
    lens = a[4:4+n].astype(np.int64)
    data = a[4+n:].tobytes().decode("utf-16le")
    assert lens.sum() <= len(data)
    edges = np.r_[0,np.cumsum(lens)]
    return np.asarray([data[edges[i]:edges[i+1]] for i in range(n)])


@njit(cache=True)
def compact_chars(buf, addresses, base):
    """Read HDF5 v1 compact UTF16 char objects without opening 1M objects."""
    out=np.zeros((len(addresses),32),np.uint32)
    for i,address in enumerate(addresses):
        start=int(address)+base
        if buf[start]!=1:
            raise ValueError("Unsupported object header version")
        size=int(buf[start+8])+(int(buf[start+9])<<8)+(int(buf[start+10])<<16)+(int(buf[start+11])<<24)
        pos=start+16;found=False
        while pos<start+16+size:
            typ=int(buf[pos])+(int(buf[pos+1])<<8)
            length=int(buf[pos+2])+(int(buf[pos+3])<<8)
            if typ==8 and buf[pos+8]==3 and buf[pos+9]==0:
                nbytes=int(buf[pos+10])+(int(buf[pos+11])<<8)
                if nbytes>64 or nbytes%2:
                    raise ValueError("Unexpected compact char length")
                for j in range(nbytes//2):
                    out[i,j]=int(buf[pos+12+2*j])+(int(buf[pos+13+2*j])<<8)
                found=True;break
            pos+=8+length
        if not found:
            raise ValueError("Compact HDF5 string layout not found")
    return out


def cell_strings(f,d):
    import h5py
    raw=np.empty(d.shape,dtype="V8")
    d.id.read(h5py.h5s.ALL,h5py.h5s.ALL,raw,mtype=h5py.h5t.STD_REF_OBJ)
    addresses=raw.view("<u8").ravel()
    with open(f.filename,"rb") as source,mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        buf=np.frombuffer(mapping,np.uint8)
        values=compact_chars(buf,addresses,int(f.userblock_size)).view("<U32").ravel().copy()
        del buf
    # Compare against the public HDF5 reader for evenly spaced source objects.
    for idx in np.unique(np.linspace(0,len(values)-1,min(64,len(values))).round().astype(int)):
        rr=d[0,idx] if d.shape[0]==1 else d[idx,0]
        expected="".join(map(chr,f[rr][()].ravel()))
        assert expected==values[idx],(idx,expected,values[idx])
    return values


def opera_table(f):
    import h5py
    objects=[f[r] for r in f["#subsystem#/MCOS"][()].ravel()]
    cells=[o for o in objects if isinstance(o,h5py.Dataset) and o.dtype==h5py.ref_dtype and o.size>=270]
    names_cell=next(o for o in cells if f[o[()].ravel()[0]].dtype==np.uint16)
    names=["".join(map(chr,f[r][()].ravel())) for r in names_cell[()].ravel()]
    data_cell=next(o for o in cells if o.id!=names_cell.id and o.size==len(names))
    cols={n:f[r] for n,r in zip(names,data_cell[()].ravel())}
    # MCOS object references index objects, starting with object ID 1 (table).
    # Resolve each string column's object ID against its serialized uint64 data.
    serialized=[o for o in objects if isinstance(o,h5py.Dataset) and o.dtype==np.uint64 and o.size>4]
    stringcols={n:(cell_strings(f,cols[n]) if cols[n].dtype==h5py.ref_dtype else
                  matlab_strings(serialized[int(cols[n][()].ravel()[4])-2]))
                for n in ("timestamp","activity","exp_no","person_id","room_no")}
    n=len(stringcols["timestamp"])
    assert all(len(v)==n for v in stringcols.values())
    # Cross-check object resolution against semantic values, not just array sizes.
    assert set(stringcols["room_no"]) <= {"1","2"}
    assert all(s.startswith("exp_") for s in stringcols["exp_no"])
    return cols, stringcols


def opera():
    import h5py
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    records, rows, inventory, arrays = [],[],[],[]
    staging=ROOT/"tmp/robustness_probe"
    staging.mkdir(exist_ok=True)
    for path in sorted((ROOT/"datasets").glob("wificsi*.zip")):
        receiver=path.stem
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                if not member.filename.endswith(".mat"):
                    continue
                ident=Path(member.filename).stem
                inv=dict(archive=path.name, member=member.filename, bytes=member.file_size,crc32=member.CRC,receiver=receiver)
                inventory.append(inv)
                cache=CACHE/(ident+".npz")
                meta_path=CACHE/(ident+".json")
                if cache.exists() and meta_path.exists():
                    saved=json.loads(meta_path.read_text(encoding="utf-8"))
                    with np.load(cache) as d: arrays.extend(d["x"])
                    rows.extend(saved["rows"]);records.append(saved["record"])
                    continue
                local=staging/member.filename
                if not local.exists():
                    with z.open(member) as src,local.open("wb") as dst:
                        shutil.copyfileobj(src,dst,1024*1024*8)
                with h5py.File(local) as f:
                    cols,s=opera_table(f)
                    activities, counts=np.unique(s["activity"],return_counts=True)
                    rec={**inv,"packets":len(s["activity"]),"labels":json.dumps(dict(zip(activities.tolist(),counts.tolist()))),
                         "rooms":json.dumps(np.unique(s["room_no"]).tolist()),
                         "people":json.dumps(np.unique(s["person_id"]).tolist()),
                         "csi_columns":len([k for k in cols if re.match(r"tx\drx\d_sub\d+$",k)]),
                         "raw_band":"5GHz channel149 (paper); bandwidth not in MAT table"}
                    # Integer milliseconds are the source precision. Equal timestamps
                    # represent several packets, not clock resets.
                    ts=pd.to_timedelta(s["timestamp"]).total_seconds().to_numpy()
                    ts=unwrap_us(np.rint(ts*1000).astype(np.int64))*1e-3
                    rec["duration_s"]=float(ts[-1]-ts[0])
                    rec["duplicate_ms"]=int((np.diff(ts)==0).sum())
                    chosen=np.isin(s["activity"],["background","walk"])
                    a=None
                    local_arrays=[];local_rows=[]
                    if chosen.any():
                        a=np.stack([np.hypot(cols[f"tx1rx1_sub{c}"][()]["real"].ravel(),
                                            cols[f"tx1rx1_sub{c}"][()]["imag"].ravel()) for c in range(1,31)],1).astype(np.float32)
                        assert len(a)==len(ts)
                        cuts=np.r_[0,np.flatnonzero((s["activity"][1:]!=s["activity"][:-1]) |
                                  (s["room_no"][1:]!=s["room_no"][:-1]) | (np.diff(ts)<0))+1,len(ts)]
                        candidates={0:[],1:[]}
                        for lo,hi in zip(cuts[:-1],cuts[1:]):
                            label=s["activity"][lo]
                            if label not in ("background","walk"):
                                continue
                            tt=ts[lo:hi]; aa=a[lo:hi]
                            unique, first, ndup=np.unique(tt,return_index=True,return_counts=True)
                            av=np.add.reduceat(aa,first,axis=0)/ndup[:,None]
                            # Keep a 1 s guard on both sides of each annotated activity.
                            ws,starts,_=make_windows(unique,av,int(label=="walk"),trim=1.,cap=100000)
                            for w,start in zip(ws,starts):
                                w52=np.stack([np.interp(ESP_K,INTEL_POS,w[:,j]) for j in range(400)],1).astype(np.float32)
                                candidates[int(label=="walk")].append((w52,dict(dataset="OPERAnet",
                                    domain=f"Room{s['room_no'][lo]}/{receiver}",room=s["room_no"][lo],receiver=receiver,
                                    recording=member.filename, group_id=f"OPERAnet::{member.filename}",
                                    experiment=s["exp_no"][lo],person=s["person_id"][lo],target=int(label=="walk"),
                                    original_label=label,start_s=float(unique[0]+start-ts[0]),
                                    end_s=float(unique[0]+start-ts[0]+3.99),antenna="tx1rx1",band="5GHz/BW_unverified",
                                    adapter="30_to_52_ordinal_half_spectrum_interpolation")))
                        for target,items in candidates.items():
                            ids=np.unique(np.linspace(0,len(items)-1,min(CAP,len(items))).round().astype(int)) if items else []
                            for i in ids:
                                w,row=items[i];local_arrays.append(w);local_rows.append(row)
                    rec["windows"]=len(local_rows)
                    np.savez_compressed(cache,x=np.stack(local_arrays) if local_arrays else np.empty((0,52,400),np.float32))
                    write_json(meta_path,dict(rows=local_rows,record=rec))
                    arrays.extend(local_arrays);rows.extend(local_rows);records.append(rec)
                    print(ident,rec["labels"],len(local_rows),'windows',flush=True)
                pd.DataFrame(records).to_csv(OUT/"opera_recordings.csv",index=False)
    x=np.stack(arrays)
    np.save(CACHE/"opera.npy",x)
    pd.DataFrame(rows).to_csv(OUT/"opera_windows.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"opera_inventory.csv",index=False)
    write_json(OUT/"opera_preparation.json",dict(shape=list(x.shape),cap_per_recording_class=CAP,
        task="background vs walk; noactivity excluded", recordings=len(records),
        adapter="ordinal half-spectrum 30-to-52 interpolation; bandwidth assumption explicitly unverified",
        cache_sha256=hashlib.sha256((CACHE/"opera.npy").read_bytes()).hexdigest()))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("dataset",choices=["days","opera"])
    args=parser.parse_args();globals()[args.dataset]()
