"""Обучение по сетке экспериментов с разделением по записям и доменам.

Каждый запуск длится 20 эпох с seed=42 и порогом 0,5. Валидация, ранняя
остановка и выбор модели по тесту не используются. Запуск можно продолжить
после прерывания. Веса и параметры стандартизации сохраняются, а массивы
данных во время обучения остаются на GPU.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from torch import nn
from models import MODELS, PREPROCESSORS, PREPROCESSOR_DESCRIPTIONS, make_model, parameter_count

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'experiments/results'
PUBLIC = ROOT/'experiments/results'
SEED, EPOCHS, BATCH = 42, 20, 64


def verify_code(plan=None):
    """Проверить версию кода после переноса файлов в общую папку экспериментов."""
    manifest=json.loads((PUBLIC/'code_manifest.json').read_text(encoding='utf8'))
    folder=Path(__file__).parent
    for name,expected in manifest['files'].items():
        actual=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if actual!=expected:
            raise RuntimeError(f'Файл {name} изменён после проверки результатов. Для новой версии нужна отдельная сетка экспериментов.')
    if plan is not None and plan['runtime_hashes']!=manifest['training_runtime_hashes']:
        raise RuntimeError('Исходные хеши обучения не соответствуют сохранённым результатам')


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)


def save_csv(path, rows):
    temp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def make_plan(meta):
    scenarios = []
    wh = sorted(meta.loc[meta.dataset=='Wallhack','domain'].unique())
    td = sorted(meta.loc[meta.dataset=='3DO','domain'].unique())
    def add(id, family, source_ds, source_domains, target_ds, target_domains):
        train = (meta.dataset==source_ds) & meta.domain.isin(source_domains)
        test = (meta.dataset==target_ds) & meta.domain.isin(target_domains)
        train_files = sorted(meta.loc[train,'recording'].unique())
        test_files = sorted(meta.loc[test,'recording'].unique())
        assert not set(train_files)&set(test_files)
        assert set(meta.loc[train,'target'])==set(meta.loc[test,'target'])=={0,1}
        scenarios.append(dict(id=id, family=family, train_dataset=source_ds, train_domains=list(source_domains),
            test_dataset=target_ds, test_domains=list(target_domains), train_recordings=train_files, test_recordings=test_files,
            train_windows=int(train.sum()), test_windows=int(test.sum()), validation='none_fixed_budget'))
    for domain in wh:
        key=domain.replace('/','-')
        add('wh_one_'+key,'Wallhack 1→3','Wallhack',[domain],'Wallhack',[d for d in wh if d!=domain])
        add('wh_lodo_'+key,'Wallhack 3→1','Wallhack',[d for d in wh if d!=domain],'Wallhack',[domain])
    for domain in td:
        add('td_one_'+domain,'3DO 1→2','3DO',[domain],'3DO',[d for d in td if d!=domain])
        add('td_lodo_'+domain,'3DO 2→1','3DO',[d for d in td if d!=domain],'3DO',[domain])
    add('wh_to_td','Wallhack→3DO','Wallhack',wh,'3DO',td)
    add('td_to_wh','3DO→Wallhack','3DO',td,'Wallhack',wh)
    plan = dict(seed=SEED, epochs=EPOCHS, batch_size=BATCH, threshold=.5,
        architectures=list(MODELS), preprocessors=list(PREPROCESSORS),
        preprocessing_descriptions=PREPROCESSOR_DESCRIPTIONS, scenarios=scenarios,
        expected_fits=len(scenarios)*len(MODELS)*len(PREPROCESSORS),
        expected_domain_metric_rows=sum(len(s['test_domains'])+(len(s['test_domains'])>1) for s in scenarios)*len(MODELS)*len(PREPROCESSORS),
        optimizer='AdamW', learning_rates={k:(.0003 if k=='transformer' else .001) for k in MODELS},
        weight_decay=.0001, schedule='cosine to 1 percent initial LR, fixed20epochs', dropout=.2,
        loss='BCEWithLogitsLoss(pos_weight=n_negative/n_positive) from source only', gradient_clip_norm=1.,
        normalization='per-subcarrier mean/std over TRAIN windows and time only; std floor1e-4',
        checkpoint='last epoch only; no validation, early stopping or test tuning',
        target='background/empty0 versus walking or walking+arm-waving1; 3DO b/w recordings only',
        sample_rate_hz=100, window_samples=400, stride_samples=400,
        seed_limit='one seed exploratory grid; no confidence intervals and no final independent model-selection test',
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        precision='CUDA autocast float16 + GradScaler; TF32 allowed; seed set, bitwise determinism not guaranteed',
        runtime_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('models.py')]})
    verify_code()
    saved_path=PUBLIC/'plan.json'
    if saved_path.exists():
        saved=json.loads(saved_path.read_text(encoding='utf8'))
        verify_code(saved)
        scientific=lambda value:{key:item for key,item in value.items() if key!='runtime_hashes'}
        if scientific(plan)!=scientific(saved):
            raise RuntimeError('Параметры, данные или окружение отличаются от сохранённого плана экспериментов')
        plan['runtime_hashes']=saved['runtime_hashes']
    return plan


def metrics(prob, truth):
    pred=prob>=.5
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return dict(n_windows=len(truth), n_positive=int(truth.sum()), n_negative=int((truth==0).sum()),
        tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),accuracy=float(accuracy_score(truth,pred)),
        balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
        precision=float(precision_score(truth,pred,zero_division=0)),recall=float(recall_score(truth,pred,zero_division=0)),
        f1=float(f1_score(truth,pred,zero_division=0)), roc_auc=float(roc_auc_score(truth,prob)),
        average_precision=float(average_precision_score(truth,prob)), threshold=.5)


def main(only_run=None):
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Для запуска полной сетки экспериментов на этом компьютере нужен GPU с CUDA'
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    meta=pd.read_csv(PUBLIC/'windows.csv')
    x=np.load(WORK/'amplitudes.npy',mmap_mode='r')
    assert x.shape==(len(meta),52,400) and len(meta)==3045
    y_all=torch.as_tensor(meta.target.to_numpy(),dtype=torch.float32,device='cuda')
    plan=make_plan(meta)
    write_json(PUBLIC/'plan.json',plan)
    for directory in ['checkpoints','predictions','curves']:
        (WORK/directory).mkdir(exist_ok=True)
    runs=pd.read_csv(PUBLIC/'runs.csv').to_dict('records') if (PUBLIC/'runs.csv').exists() else []
    results=pd.read_csv(PUBLIC/'results.csv').to_dict('records') if (PUBLIC/'results.csv').exists() else []
    completed={r['run_id'] for r in runs if r['status']=='complete'}
    started=datetime.now(timezone.utc).isoformat()
    total=plan['expected_fits']
    write_json(PUBLIC/'status.json',dict(state='running',completed=len(completed),total=total,started_utc=started))
    for prep_name, prep in PREPROCESSORS.items():
        cache=WORK/(prep_name+'.npy')
        if cache.exists():
            transformed=np.load(cache,mmap_mode='r')
        else:
            transformed=prep(x)
            np.save(cache,transformed)
        assert transformed.shape==x.shape and np.isfinite(transformed).all()
        inputs=torch.from_numpy(np.array(transformed,copy=True)).cuda()
        del transformed
        for scenario in plan['scenarios']:
            train_mask=(meta.dataset==scenario['train_dataset'])&meta.domain.isin(scenario['train_domains'])
            test_mask=(meta.dataset==scenario['test_dataset'])&meta.domain.isin(scenario['test_domains'])
            train_idx=np.flatnonzero(train_mask)
            test_idx=np.flatnonzero(test_mask)
            assert not set(meta.loc[train_mask,'group_id'])&set(meta.loc[test_mask,'group_id'])
            ti=torch.tensor(train_idx,device='cuda')
            si=torch.tensor(test_idx,device='cuda')
            raw_train=inputs[ti]
            mean=raw_train.mean((0,2),keepdim=True)
            std=raw_train.std((0,2),unbiased=False,keepdim=True).clamp_min(1e-4)
            train_x=(raw_train-mean)/std
            test_x=(inputs[si]-mean)/std
            del raw_train
            train_y=y_all[ti]
            truth=meta.loc[test_mask,'target'].to_numpy(dtype=int)
            positives=int(train_y.sum().item()); negatives=len(train_y)-positives
            pos_weight=torch.tensor([negatives/positives],device='cuda')
            for architecture in MODELS:
                run_id=f"{scenario['id']}__{prep_name}__{architecture}__s{SEED}"
                if run_id in completed or (only_run and only_run!=run_id):
                    continue
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
                model=make_model(architecture).cuda()
                lr=plan['learning_rates'][architecture]
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=plan['weight_decay'])
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=lr*.01)
                loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                scaler=torch.amp.GradScaler('cuda')
                losses=[]
                torch.cuda.synchronize(); begin=time.perf_counter()
                for epoch in range(EPOCHS):
                    model.train()
                    order=torch.randperm(len(train_y),device='cuda')
                    epoch_loss=torch.zeros((),device='cuda')
                    for left in range(0,len(order),BATCH):
                        ids=order[left:left+BATCH]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits=model(train_x[ids]).flatten()
                            loss=loss_fn(logits,train_y[ids])
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        scaler.step(optimizer);scaler.update()
                        epoch_loss+=loss.detach()*len(ids)
                    losses.append(float((epoch_loss/len(train_y)).item()))
                    assert math.isfinite(losses[-1]),run_id
                    scheduler.step()
                torch.cuda.synchronize(); training_seconds=time.perf_counter()-begin
                model.eval(); probabilities=[]
                with torch.no_grad():
                    for left in range(0,len(test_x),BATCH):
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits=model(test_x[left:left+BATCH]).flatten()
                        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                probability=np.concatenate(probabilities)
                assert np.isfinite(probability).all()
                checkpoint=WORK/'checkpoints'/(run_id+'.pt')
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=architecture, preprocessing=prep_name, mean=mean.cpu(),std=std.cpu(),
                    sample_rate_hz=100,window_samples=400,run_id=run_id,seed=SEED,epochs=EPOCHS,
                    threshold=.5,scenario=scenario, optimizer='AdamW', lr=lr, weight_decay=.0001,
                    pos_weight=float(pos_weight.item()), losses=losses,runtime_hashes=plan['runtime_hashes']),checkpoint)
                np.savez_compressed(WORK/'predictions'/(run_id+'.npz'), indices=test_idx,truth=truth,probability=probability)
                pd.DataFrame(dict(epoch=np.arange(1,EPOCHS+1),train_loss=losses)).to_csv(WORK/'curves'/(run_id+'.csv'),index=False)
                domains=meta.loc[test_mask,'domain'].to_numpy()
                targets=scenario['test_domains']+(['ALL'] if len(scenario['test_domains'])>1 else [])
                for target in targets:
                    select=np.ones(len(truth),dtype=bool) if target=='ALL' else domains==target
                    results.append(dict(run_id=run_id,scenario=scenario['id'],family=scenario['family'],
                        preprocessing=prep_name,architecture=architecture,seed=SEED,target_domain=target,
                        test_dataset=scenario['test_dataset'],**metrics(probability[select],truth[select])))
                run=dict(run_id=run_id,scenario=scenario['id'],family=scenario['family'],preprocessing=prep_name,
                    architecture=architecture,seed=SEED,epochs=EPOCHS,train_dataset=scenario['train_dataset'],
                    train_domains=' + '.join(scenario['train_domains']),test_dataset=scenario['test_dataset'],
                    test_domains=' + '.join(scenario['test_domains']),train_recordings=len(scenario['train_recordings']),
                    test_recordings=len(scenario['test_recordings']),train_windows=len(train_idx),test_windows=len(test_idx),
                    n_parameters=parameter_count(model),train_seconds=training_seconds,loss_first=losses[0],loss_last=losses[-1],
                    checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\','/'),status='complete')
                runs.append(run);completed.add(run_id)
                save_csv(PUBLIC/'results.csv',results)
                save_csv(PUBLIC/'runs.csv',runs)
                write_json(PUBLIC/'status.json',dict(state='running',completed=len(completed),total=total,
                    last_run=run_id,started_utc=started,updated_utc=datetime.now(timezone.utc).isoformat()))
                print(f"Завершено {len(completed)}/{total}: {run_id}; "
                      f"время — {training_seconds:.2f} с; "
                      f"функция потерь — с {losses[0]:.4f} до {losses[-1]:.4f}",flush=True)
                del model,optimizer,scheduler,scaler
            del train_x,test_x,train_y,ti,si,mean,std
        del inputs
        gc.collect();torch.cuda.empty_cache()
    complete=len(completed)==total
    write_json(PUBLIC/'status.json',dict(state='complete' if complete else 'partial',completed=len(completed),total=total,
        started_utc=started,updated_utc=datetime.now(timezone.utc).isoformat()))
    print(f'Обучение завершено. Выполнено экспериментов: {len(completed)} из {total}.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Обучение моделей CSI по сетке экспериментов.')
    parser.add_argument('--only-run',help='Идентификатор единственного эксперимента для запуска.')
    main(parser.parse_args().only_run)
