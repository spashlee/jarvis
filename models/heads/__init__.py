import torch.nn as nn

from .slowfast import *
from .vit import *

from .encoder.matcher import build_matcher
from .encoder.criterion import SetCriterion


def model_entry(config):
    return globals()[config['type']](**config['kwargs'])


class Head(nn.Module):
    def __init__(self, config):
        super(Head, self).__init__()
        self.module = model_entry(config)

    def forward(self, x):
        return self.module(x)