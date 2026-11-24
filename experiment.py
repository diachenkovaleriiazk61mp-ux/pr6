import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
from common import save_json
from prepare import prepare


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','prepare','speed','quality','recovery','report'),default='all')
    p.add_argument('--resume',type=Path)
    p.add_argument('--physical-cores',type=int)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    directory=a.resume or Path('smoke-results' if a.smoke else 'results')/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory=directory.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'machine.json').exists():
        machine=json.loads((directory/'machine.json').read_text(encoding='utf-8'))
        if bool(machine.get('smoke'))!=a.smoke:
            raise RuntimeError('Cannot mix smoke and full measurements')
    else:
        machine=prepare(directory,a.physical_cores)
        machine['smoke']=a.smoke
        save_json(directory/'machine.json',machine)
    threads=machine['n_th']
    env=os.environ.copy()
    env.update(USE_LIBUV='0',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    (directory/'logs').mkdir(exist_ok=True)

    def launch(name,task,mode='ddp',world=2,batch=192,lr=.02,nosync=False,failure=False):
        target=directory/('equivalence.json' if task=='check' else name+'.json')
        if target.exists():
            print(f'{name}: already saved',flush=True)
            return
        out=directory/'recovery' if failure else directory
        if failure and (out/'failure.json').exists():
            raise RuntimeError('Incomplete previous failure run; use a new result directory')
        args=['train.py','--task',task,'--mode',mode,'--name',name,
              '--out',str(out),'--threads',str(threads),'--batch',str(batch),'--lr',str(lr)]
        if a.smoke:
            args+=['--warmup','1','--steps','3','--epochs','1']
        if nosync:
            args+=['--no-sync']
        if failure:
            args+=['--failure']
        if mode=='plain':
            cmd=[sys.executable,'-u']+args
        else:
            cmd=[sys.executable,'-u','launch.py',
                 '--nnodes=1',f'--nproc_per_node={world}',
                 '--rdzv-backend=static','--master-addr=127.0.0.1',
                 f'--master-port={port()}','--node-rank=0','--monitor-interval=0.2',
                 f'--max-restarts={1 if failure else 0}']+args
        print(f'Running {name}...',flush=True)
        started=perf_counter()
        logpath=directory/'logs'/f'{name}.log'
        with logpath.open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line,end='',flush=True)
            code=proc.wait()
        if code:
            save_json(directory/'status.json',dict(state='failed',stage=name,exit_code=code))
            raise RuntimeError(f'{name} failed; see {logpath}')
        expected=out/(name+'.json') if failure else target
        if not expected.exists():
            raise RuntimeError(f'Missing result: {expected}')
        save_json(directory/f'{name}-launch.json',dict(command=cmd,elapsed=perf_counter()-started,
                                                     exit_code=code))
        print(f'{name}: saved',flush=True)

    if a.stage in ('all','speed'):
        launch('plain-B','bench','plain',1)
        launch('equivalence','check','manual')
        launch('manual-P2','bench','manual')
        for world in (1,2):
            launch(f'strong-P{world}','bench',world=world)
            launch(f'strong-P{world}-ns','bench',world=world,nosync=True)
        launch('weak-P1','bench',world=1,batch=96)
    if a.stage in ('all','quality'):
        launch('quality-plain-B','quality','plain',1)
        launch('quality-ddp-B','quality')
        launch('quality-small-B','quality','plain',1,batch=96)
        launch('quality-linear','quality',lr=.04)
    if a.stage in ('all','recovery'):
        if not (directory/'quality-ddp-B.json').exists():
            raise RuntimeError('Run quality stage in this directory before recovery')
        if not (directory/'recovery'/'quality-recovery.json').exists():
            launch('quality-recovery','quality',failure=True)
    if a.stage in ('all','report'):
        from report import build
        build(directory)
        save_json(directory/'status.json',dict(state='complete',smoke=a.smoke))
    else:
        save_json(directory/'status.json',dict(state='stage_complete',stage=a.stage,smoke=a.smoke))
    print(f'Saved: {directory}',flush=True)


if __name__=='__main__':
    main()
