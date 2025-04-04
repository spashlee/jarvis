import math

import torch
import torch.nn as nn

from .encoder import build_encoder

__all__ = ['vit']


class ViTHead(nn.Module):
    def __init__(self, config):
        super(ViTHead, self).__init__()
        
        self.encoder = build_encoder(config)
        self.temporal_resolution = config['temporal_resolution']

    # data: feats, rois
    # returns: outputs
    def forward(self, data):
        feats = data['features']   
        B, C, T, H, W = feats.shape

        feats = nn.AdaptiveAvgPool3d((self.temporal_resolution, H, W))(feats)
        mask = torch.zeros((B, H, W), dtype=torch.bool).cuda(torch.cuda.current_device())
            
        outputs = self.encoder(feats, mask, data['roi_features'], data['region_info'], data['roi_masks'])
        return {'outputs': outputs}

def vit(**kwargs):
    model = ViTHead(**kwargs)
    return model