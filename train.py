import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, roc_auc_score
from common import Net, VARIANT, load_split, save_json

LOSS = nn.CrossEntropyLoss()


def unwrap(model):
    return model.module if isinstance(model,DDP) else model


def synchronize(model, world):
    begin=perf_counter()
    end=begin
    for p in model.parameters():
        dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        end=perf_counter()
        p.grad.div_(world)
    return end-begin


def optimizer(model, lr):
    return torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)


def initialize(mode, rank):
    torch.manual_seed(0)
    net=Net()
    if mode=='manual':
        for p in net.parameters():
            dist.broadcast(p.data,src=0)
    elif mode=='ddp':
        net=DDP(net)
    return net


def loader(dataset, rank, world, batch):
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,
                               shuffle=True,seed=0,drop_last=False)
    dl=DataLoader(dataset,batch_size=batch,sampler=sampler,drop_last=True,
                  num_workers=0,generator=torch.Generator().manual_seed(123))
    return dl,sampler


def forever(dl,sampler):
    epoch=0
    while True:
        sampler.set_epoch(epoch)
        yield from dl
        epoch+=1


def do_step(model,opt,batch,mode,world,nosync=False):
    x,y=batch
    context=model.no_sync() if nosync else nullcontext()
    with context:
        opt.zero_grad(set_to_none=True)
        value=LOSS(model(x),y)
        value.backward()
        sync=synchronize(model,world) if mode=='manual' else 0.0
        opt.step()
    return value,sync


def benchmark(a,rank,world,train):
    model=initialize(a.mode,rank)
    opt=optimizer(model,.02)
    dl,sampler=loader(train,rank,world,a.batch//world)
    batches=forever(dl,sampler)
    for _ in range(a.warmup):
        do_step(model,opt,next(batches),a.mode,world,a.no_sync)
    if a.mode!='plain':
        dist.barrier()
    times,syncs=[],[]
    started=perf_counter()
    for _ in range(a.steps):
        batch=next(batches)
        begin=perf_counter()
        _,sync=do_step(model,opt,batch,a.mode,world,a.no_sync)
        times.append(perf_counter()-begin)
        syncs.append(sync)
    elapsed=perf_counter()-started
    record=dict(rank=rank,world=world,mode=a.mode,no_sync=a.no_sync,
        global_batch=a.batch,local_batch=a.batch//world,n_th=a.threads,
        warmup=a.warmup,steps=a.steps,step_seconds=times,sync_seconds=syncs,
        t_step=median(times),t_sync=median(syncs),elapsed=elapsed,
        X=a.steps*a.batch/elapsed)
    save_json(a.out/f'{a.name}-rank{rank}.json',record)
    if a.mode!='plain':
        dist.barrier()
    if rank==0:
        save_json(a.out/f'{a.name}.json',record)
        print(f'{a.name}: step={record["t_step"]*1000:.3f} ms; X={record["X"]:.2f}',flush=True)


def check_gradients(a,rank,world,train):
    model=initialize('manual',rank)
    x,y=train.tensors
    model.zero_grad(set_to_none=True)
    LOSS(model(x[:192]),y[:192]).backward()
    reference=[p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    LOSS(model(x[rank:192:world]),y[rank:192:world]).backward()
    synchronize(model,world)
    details=[]
    for (name,p),ref in zip(model.named_parameters(),reference):
        difference=(ref-p.grad).abs().max().item()
        scale=ref.abs().max().item()
        ratio=difference/scale if scale else (0. if difference==0 else None)
        if ratio is None or ratio>1e-4:
            raise RuntimeError(f'Gradient equivalence failed: {name}, {ratio}')
        details.append(dict(parameter=name,max_abs_difference=difference,
                            max_abs_reference=scale,relative_error=ratio))
    model=initialize('manual',rank)
    opt=optimizer(model,.02)
    dl,sampler=loader(train,rank,world,192//world)
    batches=forever(dl,sampler)
    for _ in range(50):
        do_step(model,opt,next(batches),'manual',world)
    checksum=torch.tensor([sum(p.detach().double().sum().item() for p in model.parameters())],dtype=torch.float64)
    checksums=[torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(checksums,checksum)
    sums=[t.item() for t in checksums]
    result=dict(world=world,global_batch=192,local_batch=192//world,steps=50,
        parameters=details,max_relative_error=max(d['relative_error'] for d in details),
        weight_sums=sums,weight_sum_spread=max(sums)-min(sums))
    if result['weight_sum_spread']>1e-9:
        raise RuntimeError('Parameters differ between ranks')
    if rank==0:
        save_json(a.out/'equivalence.json',result)
        print('Gradient equivalence and 50-step rank checks: OK',flush=True)


def evaluate(model,test):
    net=unwrap(model)
    net.eval()
    predictions,truth=[],[]
    with torch.no_grad():
        for x,y in DataLoader(test,batch_size=256,shuffle=False,num_workers=0):
            predictions.append(net(x).softmax(dim=1))
            truth.append(y)
    prob=torch.cat(predictions).numpy()
    y=torch.cat(truth).numpy()
    acc=float(accuracy_score(y,prob.argmax(axis=1)))
    auc=float(roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=list(range(7))))
    net.train()
    return acc,auc


def train_quality(a,rank,world,train,test):
    model=initialize(a.mode,rank)
    opt=optimizer(model,a.lr)
    dl,sampler=loader(train,rank,world,a.batch//world)
    step,start_epoch,skip=0,0,0
    history=[]
    ckpt=a.out/'checkpoint.pt'
    failure=a.out/'failure.json'
    restarting=int(os.environ.get('TORCHELASTIC_RESTART_COUNT','0'))>0
    if a.failure and restarting:
        if not ckpt.exists():
            raise RuntimeError('Restart requested without checkpoint')
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        unwrap(model).load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        step,start_epoch,skip=state['step'],state['epoch'],state['step_in_epoch']
        history=state['history'] if rank==0 else []
        torch.set_rng_state(state['torch_rng'])
        if rank==0:
            save_json(a.out/'recovery-start.json',dict(checkpoint_step=step,
                       epoch=start_epoch,step_in_epoch=skip,at=perf_counter()))
            print(f'Restored checkpoint at step {step}',flush=True)
    elif a.failure and ckpt.exists():
        raise RuntimeError('Refusing stale checkpoint in a new failure run')
    first_resumed_step=True
    for epoch in range(start_epoch,a.epochs):
        sampler.set_epoch(epoch)
        for i,batch in enumerate(dl):
            if epoch==start_epoch and i<skip:
                continue
            next_step=step+1
            if a.failure and next_step==25 and rank==world-1 and not failure.exists():
                checkpoint_info=json.loads((a.out/'checkpoint-event.json').read_text())
                save_json(failure,dict(rank=rank,step=25,last_completed_step=step,
                    checkpoint_step=checkpoint_info['step'],checkpoint_at=checkpoint_info['at'],
                    at=perf_counter(),epoch=epoch,step_in_epoch=i))
                os._exit(1)
            value,_=do_step(model,opt,batch,a.mode,world)
            mean_loss=value.detach().double()
            if a.mode!='plain':
                dist.all_reduce(mean_loss)
                mean_loss.div_(world)
            loss=mean_loss.item()
            if not math.isfinite(loss):
                if a.failure or a.lr == .02:
                    raise RuntimeError(f'Non-finite loss at step {next_step}: {loss}')
                if rank == 0:
                    previous = [row['loss'] for row in history if row['epoch'] == epoch]
                    save_json(a.out/f'{a.name}.json', dict(mode=a.mode, world=world,
                        global_batch=a.batch, local_batch=a.batch//world,
                        lr=a.lr, momentum=.9, epochs=a.epochs,
                        epochs_completed=epoch, optimizer_steps=next_step,
                        ACC=None, AUC=None, last_epoch_max_loss=max(previous) if previous else None,
                        losses=history, n_th=a.threads, diverged=True,
                        divergence_step=next_step))
                    print(f'{a.name}: divergence recorded at step {next_step}', flush=True)
                return
            step=next_step
            if rank==0:
                history.append(dict(step=step,epoch=epoch,step_in_epoch=i+1,loss=loss))
                if a.failure and restarting and first_resumed_step:
                    event=json.loads(failure.read_text())
                    finished=perf_counter()
                    save_json(a.out/'recovery.json',dict(
                        checkpoint_step=event['checkpoint_step'],failure_step=25,
                        repeated_steps=event['last_completed_step']-event['checkpoint_step'],
                        t_f=event['at'],t_c=event['checkpoint_at'],
                        first_resumed_step=step,first_resumed_finished=finished,
                        t_r=finished-event['at'],
                        t_lost=(event['at']-event['checkpoint_at'])+(finished-event['at'])))
                    first_resumed_step=False
            if a.failure and step%10==0:
                if rank==0:
                    state=dict(model=unwrap(model).state_dict(),opt=opt.state_dict(),
                        epoch=epoch,step_in_epoch=i+1,step=step,history=history,
                        torch_rng=torch.get_rng_state())
                    torch.save(state,a.out/'checkpoint.tmp')
                    os.replace(a.out/'checkpoint.tmp',ckpt)
                    save_json(a.out/'checkpoint-event.json',dict(step=step,at=perf_counter()))
                dist.barrier()
        skip=0
        if rank==0:
            print(f'{a.name}: epoch {epoch+1}/{a.epochs}, step={step}, loss={loss:.6f}',flush=True)
    if a.mode!='plain':
        dist.barrier()
    if rank==0:
        acc,auc=evaluate(model,test)
        last=[p['loss'] for p in history if p['epoch']==a.epochs-1]
        save_json(a.out/f'{a.name}.json',dict(
            mode=a.mode,world=world,global_batch=a.batch,local_batch=a.batch//world,
            lr=a.lr,momentum=.9,epochs=a.epochs,optimizer_steps=step,
            ACC=acc,AUC=auc,last_epoch_max_loss=max(last),losses=history,
            n_th=a.threads,test_size=len(test),
            weight_sum=sum(p.detach().double().sum().item() for p in unwrap(model).parameters())))
        print(f'{a.name}: ACC={acc:.6f}, AUC={auc:.6f}',flush=True)
    if a.mode!='plain':
        dist.barrier()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=('bench','quality','check'),required=True)
    p.add_argument('--mode',choices=('plain','manual','ddp'),default='ddp')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--data',type=Path,default=Path('data/dermamnist.npz'))
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--batch',type=int,default=192)
    p.add_argument('--lr',type=float,default=.02)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--no-sync',action='store_true')
    p.add_argument('--failure',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1')) if a.mode!='plain' else 1
    assert a.threads>=1 and a.batch%world==0
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode!='plain':
        os.environ['USE_LIBUV']='0'
        timeout = timedelta(seconds=120)
        store = dist.TCPStore(
            '127.0.0.1', int(os.environ['MASTER_PORT']),
            world_size=None, is_master=False, timeout=timeout,
            use_libuv=False)
        restart = os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')
        store = dist.PrefixStore(f'training/{a.name}/restart-{restart}', store)
        dist.init_process_group(
            'gloo', store=store, rank=rank, world_size=world, timeout=timeout)
    train=load_split(a.data,'train',7000)
    assert len(train)==7000
    if a.task=='check':
        assert world==2 and a.mode=='manual'
        check_gradients(a,rank,world,train)
    elif a.task=='bench':
        benchmark(a,rank,world,train)
    else:
        test=load_split(a.data,'test')
        train_quality(a,rank,world,train,test)
    if a.mode!='plain':
        dist.destroy_process_group()


if __name__=='__main__':
    main()
