import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import urllib.request
import psutil
import torch
from common import VARIANT, Net, file_md5, parameter_count, save_json


def prepare(directory, physical_cores=None):
    directory=Path(directory)
    cores=physical_cores or psutil.cpu_count(logical=False)
    if not cores:
        raise RuntimeError('Cannot detect physical cores; pass --physical-cores.')
    assert cores>=1
    data=Path('data')
    data.mkdir(exist_ok=True)
    path=data/'dermamnist.npz'
    expected='0744692d530f8e62ec473284d019b0c7'
    if not path.exists() or file_md5(path)!=expected:
        print('Downloading official DermaMNIST (about 20 MB)...',flush=True)
        temp=path.with_suffix('.download')
        urllib.request.urlretrieve('https://zenodo.org/records/10519652/files/dermamnist.npz?download=1',temp)
        if file_md5(temp)!=expected:
            raise RuntimeError('Dataset checksum mismatch')
        temp.replace(path)
    torch.manual_seed(0)
    params=sum(p.numel() for p in Net().parameters())
    assert params==parameter_count()
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        raise RuntimeError('This PyTorch build has no gloo backend')
    metadata=dict(variant=VARIANT, processor=platform.processor(),
        physical_cores=cores, logical_cores=psutil.cpu_count(logical=True),
        memory_GiB=psutil.virtual_memory().total/1024**3, os=platform.platform(),
        python=sys.version, torch=torch.__version__, n_th=max(1,cores//2),
        n_params=params, gradient_bytes=4*params, dataset_md5=expected,
        data_path=str(path.resolve()), packages={name:importlib.metadata.version(name)
                     for name in ('torch','numpy','scikit-learn','psutil','matplotlib')})
    save_json(directory/'machine.json',metadata)
    print(f'Dataset OK; parameters={params}; threads per process={metadata["n_th"]}',flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/prepared')
    p.add_argument('--physical-cores',type=int)
    a=p.parse_args()
    prepare(a.out,a.physical_cores)
