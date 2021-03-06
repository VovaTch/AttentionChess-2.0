import pathlib
import os
import shutil
import yaml
from pathlib import Path
from itertools import repeat
import numpy as np

import torch
import pandas as pd
import chess
    


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.safe_load(handle)

def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=1, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def delete_last_folder(config_dict):
    """
    Removes last log + model folders, this is useful for inference + test runs.
    """
    model_name = config_dict['name']
    latest_dir_log = max(pathlib.Path('saved/log/' + model_name + '/').glob('*/'), key=os.path.getmtime)
    latest_dir_models = max(pathlib.Path('saved/models/' + model_name + '/').glob('*/'), key=os.path.getmtime)
    
    shutil.rmtree(str(latest_dir_log) + '/')
    shutil.rmtree(str(latest_dir_models) + '/')
    
    print('Removed the latest checkpoint folder')

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
def num_param(model: torch.nn.Module):
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    return params

def extract_device(model: torch.nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    for parameter in model_parameters:
        device = parameter.device
        break
    return device

def is_game_end(board: chess.Board):
    """Checks if the game ends."""
    if board.is_checkmate():
        result_const = -1 if board.turn else 1
        return True, result_const
    elif board.is_stalemate() or board.is_repetition() or \
            board.is_seventyfive_moves() or board.is_insufficient_material():
        return True, 0
    return False, 0