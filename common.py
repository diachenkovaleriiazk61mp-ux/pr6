import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

VARIANT = dict(variant=7, dataset='dermamnist', K=7, C=3, c1=32, c2=64,
               h=256, P_max=2, B=192, b=96, K_c=10, s_f=25, N=7000,
               epochs=3, lr=.02, linear_lr=.04, momentum=.9, warmup=10, steps=50)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_split(path, split, n=None):
    with np.load(path) as d:
        x = d[split + '_images'][:n].copy()
        y = d[split + '_labels'][:n].reshape(-1).copy()
    x = x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = (x / 255 - .5) / .5
    return TensorDataset(x, torch.from_numpy(y.astype(np.int64)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(7*7*64, 256),
                                  nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        return self.head(self.features(x))


def parameter_count():
    return 9*3*32+32+9*32*64+64+49*64*256+256+256*7+7


def file_md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
