import torch
import torch.nn as nn

from .slowfast import *
from .slowfast_dc5 import *
from .vision_transformer import *
from utils import load_pretrain


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        
        self.config = config
        self.module = model_entry(config)
        if config.get('pretrain', None) is not None:
            load_pretrain(config.pretrain, self.module)
           
        if not config.get('learnable', True):
            self.module.requires_grad_(False)

    # data: clips
    # returns: features
    def forward(self, data):
        features = self.module(data['clips'])
        return {'features': features}