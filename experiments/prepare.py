"""Подготовка CSI-окон из полных записей, подходящих для бинарной задачи.

Случайное разделение окон на выборки не выполняется.
"""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'experiments/results'
PUBLIC = ROOT / 'experiments/results'
SELECTED = np.r_[6:32, 33:59]


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(ROOT / 'research/analysis/window_features.csv')
    arrays, metadata, recordings = [], [], []
    for dataset, archive in [('Wallhack', 'wallhack1.8k.zip'), ('3DO', '3DO.zip')]:
        with zipfile.ZipFile(ROOT / 'datasets' / archive) as z:
            names = sorted(n for n in z.namelist() if n.endswith('.csv') and
                (('spectrograms' not in n) if dataset == 'Wallhack' else
                 (n.endswith('/csiposreg.csv') and n.split('/')[2].startswith(('b', 'w')))))
            for path in names:
                print('Подготовка записи:', path, flush=True)
                df = pd.read_csv(z.open(path))
                raw_count = len(df)
                valid = (df.data.str.count(',') + 1 == df['len']) & (df.rx_state == 0)
                df = df.loc[valid].reset_index(drop=True)
                raw = np.stack([np.fromstring(s.strip()[1:-1], sep=',', dtype=np.float32) for s in df.data])
                assert raw.shape[1] == 256
                amp = np.abs(raw[:, 2 * SELECTED + 1] + 1j * raw[:, 2 * SELECTED])
                ticks = df.local_timestamp.to_numpy(dtype=np.float64)
                wrap = np.r_[0, (np.diff(ticks) < -(2**31)).astype(int)]
                adjusted = ticks + np.cumsum(wrap) * (2**32)
                t = (adjusted - adjusted[0]) * 1e-6
                labels = df['class'].to_numpy(dtype=int)
                domain = '/'.join(path.split('/')[1:3]) if dataset == 'Wallhack' else path.split('/')[1]
                sequence = Path(path).stem if dataset == 'Wallhack' else path.split('/')[2]
                expected = 0 if sequence.startswith('b') else (2 if sequence.startswith('ww') else 1)
                assert set(labels) == {expected}, (path, set(labels), expected)
                cuts = np.r_[0, np.flatnonzero((np.diff(t) <= 0) | (np.diff(t) > .05) | (np.diff(labels) != 0)) + 1, len(t)]
                count_before = len(metadata)
                for start, end in zip(cuts[:-1], cuts[1:]):
                    if end-start < 400 or t[end-1]-t[start] < 3.99:
                        continue
                    local = t[start:end] - t[start]
                    grid = np.arange(0, local[-1]+1e-8, .01)
                    reg = np.stack([np.interp(grid, local, amp[start:end, k]) for k in range(52)], axis=1).astype(np.float32)
                    for s in range(0, len(reg)-400+1, 400):
                        arrays.append(reg[s:s+400].T)
                        metadata.append(dict(index=len(metadata), dataset=dataset, domain=domain, sequence=sequence,
                            recording=path, group_id=dataset+'::'+path, original_label=expected,
                            target=int(expected != 0), start_s=float(t[start]+s/100), end_s=float(t[start]+s/100+3.99)))
                recordings.append(dict(dataset=dataset, domain=domain, recording=path, sequence=sequence,
                    target=int(expected != 0), original_label=expected, raw_packets=raw_count, valid_packets=len(df),
                    discarded_packets=raw_count-len(df), clock_wraps=int(wrap.sum()), windows=len(metadata)-count_before,
                    source_crc32=z.getinfo(path).CRC, source_bytes=z.getinfo(path).file_size))
                prior = old[old.path == path]
                now = metadata[count_before:]
                assert len(prior) == len(now), (path, len(prior), len(now))
                assert np.allclose(prior.start_s.to_numpy(), [r['start_s'] for r in now], atol=1e-7)
    x = np.stack(arrays).astype(np.float32)
    meta = pd.DataFrame(metadata)
    rec = pd.DataFrame(recordings)
    assert x.shape == (len(meta), 52, 400) and np.isfinite(x).all()
    assert len(rec) == 67 and rec.target.value_counts().to_dict() == {1:54, 0:13}
    for _, g in meta.groupby('recording'):
        assert (g.start_s.diff().dropna() >= 4-1e-7).all()
    np.save(WORK/'amplitudes.npy', x)
    meta.to_csv(PUBLIC/'windows.csv', index=False)
    rec.to_csv(PUBLIC/'recordings.csv', index=False)
    info = dict(shape=list(x.shape), dtype=str(x.dtype), sample_rate_hz=100, window_seconds=4,
        window_stride_seconds=4, subcarriers=SELECTED.tolist(),
        task='0=empty/background; 1=walking or walking+arm-waving; 3DO sitting/lying recordings excluded',
        parser='real=data[2k+1]; imaginary=data[2k]', clock='unwrap uint32 microseconds',
        gap_policy='split when dt<=0 or dt>0.05s; interpolate amplitude only within continuous segment',
        validation='every start_s and recording count matches previous audited EDA manifest',
        md5={n:hashlib.md5((ROOT/'datasets'/n).read_bytes()).hexdigest() for n in ['3DO.zip','wallhack1.8k.zip']})
    (PUBLIC/'data_manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf8')
    print(f"Данные подготовлены: {len(rec)} записей, {len(meta)} окон. "
          f"Форма массива: {info['shape']}; тип данных: {info['dtype']}. "
          f"Частота: {info['sample_rate_hz']} Гц; длина и шаг окна: "
          f"{info['window_seconds']} и {info['window_stride_seconds']} с.", flush=True)
    print(f"Массивы сохранены в {WORK}, описание данных — в {PUBLIC / 'data_manifest.json'}.", flush=True)


if __name__ == '__main__':
    prepare()
