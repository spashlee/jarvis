import math

import torch
import torch.nn as nn

from .encoder import build_encoder

__all__ = ['slowfast_2d', 'slowfast_3d']


class SlowFast2DHead(nn.Module):
    def __init__(self, config):
        super(SlowFast2DHead, self).__init__()
        
        self.encoder = build_encoder(config)

    # data: feats, rois
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']        

        # temporal average pooling
        B, D, T, H, W = feats[0].shape
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((1, H, W))(f).view(-1, f.shape[1], H, W) for f in feats]
        feats = torch.cat(feats, dim=1)
            
        sizes_before_padding = data['sizes_before_padding']
        mask = torch.ones((B, H, W), dtype=torch.bool).cuda(torch.cuda.current_device())
        for i in range(B):
            eff_h, eff_w = math.ceil(H * sizes_before_padding[i][1]), math.ceil(W * sizes_before_padding[i][0])
            mask[i, :eff_h, :eff_w] = False

        outputs = self.encoder(feats, mask, data['roi_features'], data['region_info'])
        return {'outputs': outputs}


class SlowFast3DHead(nn.Module):
    def __init__(self, config):
        super(SlowFast3DHead, self).__init__()
        
        self.encoder = build_encoder(config)
        self.temporal_resolution = config['temporal_resolution']

    # data: feats, rois
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']   
        
        # temporal average pooling
        B, D, T, H, W = feats[0].shape
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((self.temporal_resolution, H, W))(f) for f in feats]
        feats = torch.cat(feats, dim=1)
            
        sizes_before_padding = data['sizes_before_padding']
        mask = torch.ones((B, H, W), dtype=torch.bool).cuda(torch.cuda.current_device())
        for i in range(B):
            eff_h, eff_w = math.ceil(H * sizes_before_padding[i][1]), math.ceil(W * sizes_before_padding[i][0])
            mask[i, :eff_h, :eff_w] = False

        outputs = self.encoder(feats, mask, data['roi_features'], data['region_info'])
        return {'outputs': outputs}


def slowfast_2d(**kwargs):
    model = SlowFast2DHead(**kwargs)
    return model

def slowfast_3d(**kwargs):
    model = SlowFast3DHead(**kwargs)
    return model