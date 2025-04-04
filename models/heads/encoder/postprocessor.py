import torch
import torch.nn as nn

from .utils import box_ops

class ActionPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
        """
        out_logits, out_bbox = outputs['action_pred_logits'], outputs['pred_boxes']
        
        if 'action_pred_prob' in outputs:
            prob = outputs['action_pred_prob']
        else:
            prob = out_logits.sigmoid()

        if outputs['pred_boxes_mask'] is not None:
            out_bbox_mask = outputs['pred_boxes_mask']
            prob = prob * (~out_bbox_mask).float().unsqueeze(-1)

        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor')
        labels = topk_indexes % out_logits.shape[2]

        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class ActorPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
        """
        out_scores, out_bbox = outputs['pred_scores'], outputs['pred_boxes']

        prob = out_scores
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='floor')
        labels = topk_indexes % prob.shape[2]

        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


def set_postprocessor():
    postprocessor = {'action': ActionPostProcess(), 'actor': ActorPostProcess()}
    return postprocessor