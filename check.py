import copy
import tempfile
from pathlib import Path
import torch
from common import Net, load_split, parameter_count
from train import LOSS, do_step, evaluate, loader, optimizer


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset=load_split('data/dermamnist.npz','train',7000)
    test=load_split('data/dermamnist.npz','test')
    assert len(dataset)==7000 and len(test)==2005
    assert dataset.tensors[0].shape==(7000,3,28,28)
    assert set(dataset.tensors[1].tolist())==set(range(7))
    assert dataset.tensors[0].min()>=-1 and dataset.tensors[0].max()<=1
    torch.manual_seed(0)
    net=Net()
    assert sum(p.numel() for p in net.parameters())==parameter_count()==824263
    x,y=dataset.tensors
    LOSS(net(x[:192]),y[:192]).backward()
    reference=[p.grad.clone() for p in net.parameters()]
    partial=[]
    for rank in range(2):
        net.zero_grad(set_to_none=True)
        LOSS(net(x[rank:192:2]),y[rank:192:2]).backward()
        partial.append([p.grad.clone() for p in net.parameters()])
    max_error=0
    for i,ref in enumerate(reference):
        averaged=(partial[0][i]+partial[1][i])/2
        error=(ref-averaged).abs().max().item()/max(ref.abs().max().item(),1e-30)
        max_error=max(max_error,error)
    assert max_error<1e-4
    torch.manual_seed(0)
    net=Net()
    opt=optimizer(net,.02)
    dl,sampler=loader(dataset,0,1,192)
    sampler.set_epoch(0)
    baseline=[]
    checkpoint=None
    for i,batch in enumerate(dl):
        value,_=do_step(net,opt,batch,'plain',1)
        baseline.append(value.item())
        if i+1==20:
            checkpoint=copy.deepcopy(dict(model=net.state_dict(),opt=opt.state_dict(),step=20))
    final={k:v.clone() for k,v in net.state_dict().items()}
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'checkpoint.pt'
        torch.save(checkpoint,path)
        state=torch.load(path,map_location='cpu',weights_only=False)
    restored=Net()
    restored.load_state_dict(state['model'])
    resumed_opt=optimizer(restored,.02)
    resumed_opt.load_state_dict(state['opt'])
    dl,sampler=loader(dataset,0,1,192)
    sampler.set_epoch(0)
    for i,batch in enumerate(dl):
        if i<state['step']:
            continue
        value,_=do_step(restored,resumed_opt,batch,'plain',1)
        assert value.item()==baseline[i]
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in final.items())
    acc,auc=evaluate(restored,test)
    assert 0<=acc<=1 and 0<=auc<=1
    print(f'Local checks OK: parameters=824263; gradient error={max_error:.3g}; '
          f'checkpoint/SGD exact; test ACC={acc:.4f}, AUC={auc:.4f}')


if __name__=='__main__':
    main()
