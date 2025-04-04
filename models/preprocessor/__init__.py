import torch.nn as nn

from .basic import *
from .ava import *
from .utils import *

def model_entry(config):
    return globals()[config['type']](**config['kwargs'])


class Preprocessor(nn.Module):
    def __init__(self, config):
        super(Preprocessor, self).__init__()
        self.module = model_entry(config)

    def forward(self, data):
        return self.module(data)
