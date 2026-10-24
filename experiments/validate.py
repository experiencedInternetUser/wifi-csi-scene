"""Проверка сохранённых экспериментов: разбиения данных, веса и пересчёт метрик."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import make_model
from train_grid import verify_code
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'experiments/results'
WORK=ROOT/'experiments/results'


def validate():
    plan=json.loads((PUBLIC/'plan.json').read_text(encoding='utf8'))
    status=json.loads((PUBLIC/'status.json').read_text(encoding='utf8'))
    runs=pd.read_csv(PUBLIC/'runs.csv')
    results=pd.read_csv(PUBLIC/'results.csv')
    meta=pd.read_csv(PUBLIC/'windows.csv')
    recordings=pd.read_csv(PUBLIC/'recordings.csv')
    assert status['state']=='complete' and status['completed']==plan['expected_fits']==256
    expected={f"{s['id']}__{p}__{a}__s42" for s,p,a in itertools.product(plan['scenarios'],plan['preprocessors'],plan['architectures'])}
    assert len(runs)==256 and set(runs.run_id)==expected and runs.run_id.is_unique
    assert set(runs.status)=={'complete'}
    assert len(results)==plan['expected_domain_metric_rows']==656
    assert not results.duplicated(['run_id','target_domain']).any()
    assert len(meta)==3045 and len(recordings)==67 and meta['index'].is_unique
    assert set(meta.original_label)=={0,1,2}
    assert (meta.target==(meta.original_label!=0).astype(int)).all()
    assert meta.loc[meta.dataset=='3DO','sequence'].str.startswith(('b','w')).all()
    assert not meta.duplicated(['recording','start_s']).any()
    assert len(set(zip(recordings.source_crc32,recordings.source_bytes)))==67
    for _,g in meta.groupby('recording'):
        assert (g.start_s.diff().dropna()>=4-1e-7).all()
    verify_code(plan)
    scaler_checked=set(); metric_count=0; checkpoint_count=0; max_scaler_error=0.
    for _,run in runs.iterrows():
        s=next(s for s in plan['scenarios'] if s['id']==run.scenario)
        train=(meta.dataset==s['train_dataset'])&meta.domain.isin(s['train_domains'])
        test=(meta.dataset==s['test_dataset'])&meta.domain.isin(s['test_domains'])
        assert not set(meta.loc[train,'group_id'])&set(meta.loc[test,'group_id'])
        if s['train_dataset']==s['test_dataset']:
            assert not set(s['train_domains'])&set(s['test_domains'])
        assert set(meta.loc[train,'recording'])==set(s['train_recordings'])
        assert set(meta.loc[test,'recording'])==set(s['test_recordings'])
        pred=np.load(WORK/'predictions'/(run.run_id+'.npz'))
        indices=pred['indices']; truth=pred['truth']; prob=pred['probability']
        assert np.array_equal(indices,np.flatnonzero(test))
        assert np.array_equal(truth,meta.loc[test,'target'].to_numpy())
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        checkpoint=torch.load(ROOT/run.checkpoint,map_location='cpu',weights_only=False)
        checkpoint_count+=1
        assert checkpoint['epochs']==20 and checkpoint['seed']==42 and checkpoint['threshold']==.5
        assert checkpoint['scenario']==s and checkpoint['runtime_hashes']==plan['runtime_hashes']
        assert len(checkpoint['losses'])==20 and np.isfinite(checkpoint['losses']).all()
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        pos=meta.loc[train,'target'].sum();neg=int(train.sum())-pos
        assert np.isclose(checkpoint['pos_weight'],neg/pos,rtol=1e-6)
        scaler_key=run.scenario,run.preprocessing
        if scaler_key not in scaler_checked:
            transformed=np.load(WORK/(run.preprocessing+'.npy'),mmap_mode='r')
            source=np.asarray(transformed[np.flatnonzero(train)])
            mean=source.mean((0,2),dtype=np.float64).reshape(1,52,1)
            std=source.std((0,2),dtype=np.float64).clip(1e-4).reshape(1,52,1)
            saved_mean=checkpoint['mean'].numpy();saved_std=checkpoint['std'].numpy()
            assert np.allclose(mean,saved_mean,rtol=1e-4,atol=1e-5),scaler_key
            assert np.allclose(std,saved_std,rtol=1e-4,atol=1e-5),scaler_key
            max_scaler_error=max(max_scaler_error,float(np.max(np.abs(mean-saved_mean))),float(np.max(np.abs(std-saved_std))))
            scaler_checked.add(scaler_key)
        target_domains=meta.loc[test,'domain'].to_numpy()
        actual=results[results.run_id==run.run_id]
        expected_domains=set(s['test_domains'])|({'ALL'} if len(s['test_domains'])>1 else set())
        assert set(actual.target_domain)==expected_domains
        for _,row in actual.iterrows():
            mask=np.ones(len(truth),dtype=bool) if row.target_domain=='ALL' else target_domains==row.target_domain
            y=truth[mask];p=prob[mask];binary=p>=.5
            tn,fp,fn,tp=confusion_matrix(y,binary,labels=[0,1]).ravel()
            assert [row.tn,row.fp,row.fn,row.tp]==[tn,fp,fn,tp]
            values=dict(accuracy=accuracy_score(y,binary),balanced_accuracy=balanced_accuracy_score(y,binary),
                precision=precision_score(y,binary,zero_division=0),recall=recall_score(y,binary,zero_division=0),
                f1=f1_score(y,binary,zero_division=0),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p))
            for name,value in values.items():
                assert np.isclose(row[name],value,atol=1e-10),(run.run_id,name)
            metric_count+=1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    replay_errors={}
    for architecture in plan['architectures']:
        run_id=f'wh_to_td__relative__{architecture}__s42'
        row=runs[runs.run_id==run_id].iloc[0]
        checkpoint=torch.load(ROOT/row.checkpoint,map_location='cpu',weights_only=False)
        pred=np.load(WORK/'predictions'/(run_id+'.npz'))
        transformed=np.load(WORK/'relative.npy',mmap_mode='r')
        data=torch.tensor(np.array(transformed[pred['indices']],copy=True),device='cuda')
        data=(data-checkpoint['mean'].cuda())/checkpoint['std'].cuda()
        model=make_model(architecture).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        replay=[]
        with torch.no_grad():
            for first in range(0,len(data),64):
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(data[first:first+64]).flatten()
                replay.append(torch.sigmoid(logits.float()).cpu().numpy())
        probability=np.concatenate(replay)
        error=float(np.max(np.abs(probability-pred['probability'])))
        assert np.allclose(probability,pred['probability'],rtol=2e-3,atol=2e-4),(run_id,error)
        replay_errors[architecture]=error
        del model,data
    summary=dict(state='passed',fits=256,metric_rows_recomputed=metric_count,checkpoints_checked=checkpoint_count,
        scalers_independently_checked=len(scaler_checked),max_scaler_absolute_error=max_scaler_error,
        recordings=67,windows=3045,recording_splits_disjoint=True,domain_splits_disjoint=True,
        windows_nonoverlapping=True,binary_labels_verified=True,source_only_scalers_verified=True,
        runtime_hashes_verified=True,checkpoint_weights_finite=True,predictions_match_original_window_indices=True,
        restored_model_inference_max_absolute_probability_errors=replay_errors)
    (PUBLIC/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(f"Проверка пройдена: {summary['fits']} обучений, "
          f"{summary['checkpoints_checked']} файлов весов, "
          f"{summary['metric_rows_recomputed']} строк с пересчитанными метриками.")
    print(f"Проверено {summary['recordings']} записей и {summary['windows']} окон. "
          "Записи и домены обучения и теста не пересекаются; окна не перекрываются.")
    print("Бинарные метки, соответствие предсказаний исходным окнам, хеши кода "
          "и конечность значений весов проверены.")
    print(f"Параметры стандартизации проверены для "
          f"{summary['scalers_independently_checked']} сочетаний разбиения и предобработки. "
          f"Максимальная абсолютная ошибка: {summary['max_scaler_absolute_error']:.8g}.")
    print("Максимальная абсолютная ошибка вероятностей после загрузки весов:")
    for architecture, error in replay_errors.items():
        print(f"  {architecture}: {error:.8g}")
    print(f"Результат проверки сохранён: {PUBLIC / 'validation.json'}")
    return summary


if __name__=='__main__':
    validate()
