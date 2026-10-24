
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import train_grid


def replace_retry(temp, target):
    for attempt in range(40):
        try:
            temp.replace(target)
            return
        except PermissionError:
            if attempt==39:
                raise
            time.sleep(.05+min(attempt*.02,.35))


def main(check=False):
    folder=train_grid.PUBLIC
    meta=pd.read_csv(folder/'windows.csv')
    if (folder/'plan.json').exists():
        existing=json.loads((folder/'plan.json').read_text(encoding='utf8'))
        assert train_grid.make_plan(meta)==existing,'Продолжить обучение нельзя: план экспериментов или код изменился'
    else:
        raise RuntimeError('Не найден сохранённый план экспериментов для проверки совместимости')
    if check:
        print('План экспериментов и совместимость исходников проверены. Обучение не запускалось.')
        return
    prior_status=json.loads((folder/'status.json').read_text(encoding='utf8')) if (folder/'status.json').exists() else {}
    initial_started=prior_status.get('started_utc')
    def write_json(path,value):
        if path.name=='status.json' and initial_started:
            value['started_utc']=initial_started
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        replace_retry(temp,path)
    def save_csv(path,rows):
        temp=path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(temp,index=False)
        replace_retry(temp,path)
    train_grid.write_json=write_json
    train_grid.save_csv=save_csv
    train_grid.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Продолжение обучения по сохранённому плану экспериментов.')
    parser.add_argument('--check',action='store_true',help='Проверить план и исходники без запуска обучения или записи файлов.')
    main(check=parser.parse_args().check)
