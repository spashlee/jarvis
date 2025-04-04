# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, threshold
import math

from .utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .position_encoding import build_position_encoding
from .matcher import build_matcher
from .transformer import build_transformer

class Encoder(nn.Module):
    """ This is the Encoder module that performs object detection """
    def __init__(self, position_encoding, transformer, in_channels, num_classes, temporal_resolution):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Encoder can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.hidden_dim
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.roi_embed = nn.Linear(hidden_dim, hidden_dim)
        self.roi_pos_embed = nn.Linear(6, hidden_dim)

        self.input_proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)

        self.position_encoding = position_encoding
        self.temporal_resolution = temporal_resolution

        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.roi_embed.weight, gain=1)
        nn.init.constant_(self.roi_embed.bias, 0)
        nn.init.xavier_uniform_(self.roi_pos_embed.weight, gain=1)
        nn.init.constant_(self.roi_pos_embed.bias, 0)

    def forward(self, feat, masks, roi_feat, region_info, roi_masks=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
        """
        vis_embed = self.input_proj(feat)
        spatial_embed, temporal_embed = self.position_encoding(feat, masks, self.temporal_resolution)

        new_masks = []
        new_pos_embed = []
        for idx in range(self.temporal_resolution):
            pos_embed = spatial_embed + temporal_embed[idx][None, :, None, None]
            new_masks.append(masks)
            new_pos_embed.append(pos_embed)
        masks = torch.stack(new_masks, dim=2).flatten(1)
        pos_embed = torch.stack(new_pos_embed, dim=2)

        # flatten NxCxHxW to HWxNxC
        vis_embed = vis_embed.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        vis_embed = vis_embed + pos_embed

        roi_embed = self.roi_embed(roi_feat)
        roi_pos_embed = self.roi_pos_embed(region_info)
        roi_embed = roi_embed + roi_pos_embed
        
        bs, roi_len = roi_embed.shape[:2]        
        input_embed = torch.cat((roi_embed, vis_embed), dim=1)

        if roi_masks is None:
            roi_masks = torch.zeros((bs, roi_len), dtype=torch.bool, device=vis_embed.device)
        masks = torch.cat((roi_masks, masks), dim=1)
        
        assert masks is not None
        hs = self.transformer(input_embed, masks)

        outputs_class = self.class_embed(hs[:, :roi_len])
        out = {'action_pred_logits': outputs_class}
        return out


def build_encoder(config):
    position_encoding = build_position_encoding(**config['position_encoding'])
    transformer = build_transformer(**config['transformer'])

    model = Encoder(
        position_encoding,
        transformer,
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        temporal_resolution=config['temporal_resolution'],
    )
    return model