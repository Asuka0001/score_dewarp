"""Durable full-training queue with a single accelerator job and fixed follow-up."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=ROOT/'outputs/diverse_training_20261115';out.mkdir(exist_ok=True,parents=True)
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT)
    state={'status':'running','pid':os.getpid(),'planned_epochs':120,'completed':[]}
    def status(**changes):
        state.update(changes,updated_at=datetime.now(timezone.utc).isoformat())
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'status.json')
    def run(name,command,result):
        if result.exists():state['completed'].append(name);status();return
        status(current=name,command=command)
        with (out/(name+'.log')).open('a') as log:
            subprocess.run(command,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if not result.exists():raise ValueError(f'Missing result for {name}')
        state['completed'].append(name);status()
    try:
        status()
        command=[PYTHON,'-u','scripts/train_diverse_geometry.py','--cache','data/prepared/diverse_cache_clean_20261115',
                 '--out','outputs/diverse_training_20261115/model','--epochs','120','--device','mps']
        if args.resume and (out/'model/last.pt').exists():command+=['--resume']
        run('train',command,out/'model/report.json')
        checkpoint='outputs/diverse_training_20261115/model/best.pt'
        for dataset in ('deepscores_full_v3','doc3d_pilot'):
            dest=out/('retention_'+dataset)
            run('retention_'+dataset,[PYTHON,'scripts/evaluate_geometry.py','--data',f'data/prepared/{dataset}/val',
                '--checkpoint',checkpoint,'--out',str(dest),'--resume'],dest/'summary.json')
        run('cpms',[PYTHON,'scripts/evaluate_cpms.py','--out',str(out/'cpms'),
            '--checkpoint','network_original=outputs/auxiliary_20261115/original_kind/validated.pt',
            '--checkpoint','network_robust='+checkpoint],out/'cpms/evaluation.json')
        run('cpms_fields',[PYTHON,'scripts/diagnose_cpms_fields.py','--experiment',str(out/'cpms')],out/'cpms/field_diagnostic.json')
        status(status='complete',current=None)
        subprocess.run([PYTHON,'scripts/report_diverse_training.py'],check=True,cwd=ROOT)
    except BaseException as exc:
        status(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
