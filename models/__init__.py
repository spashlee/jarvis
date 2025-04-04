import torch
import torch.nn as nn

from .backbones import Backbone
from .heads import Head
from .heads.encoder import set_criterion, set_postprocessor

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer


class JARViS(nn.Module):
    def __init__(self, config):
        super(JARViS, self).__init__()
        self.config = config

        self.backbone = Backbone(config.backbone)
        self.head = Head(config.head)

        self.criterion = set_criterion(config.head)
        self.postprocessor = set_postprocessor()

    def make_region_info(self, outputs):
        pred_boxes = outputs['pred_boxes']

        pred_region_info = []
        for boxes in pred_boxes:
            height = boxes[:, [3]] - boxes[:, [1]]
            width = boxes[:, [2]] - boxes[:, [0]]
            
            region_info = torch.cat((boxes, height, width), dim=1)
            pred_region_info.append(region_info)
        pred_region_info = torch.stack(pred_region_info)
        return pred_region_info

    def forward(self, data):
        region_info = self.make_region_info(data)

        i_b = {'clips': data['clips']}
        o_b = self.backbone(i_b)

        i_h = {'features': o_b['features'], 'sizes_before_padding': data['sizes_before_padding'],
               'roi_features': data['roi_features'], 'region_info': region_info,
               'roi_masks': data['pred_boxes_mask']}
        o_h = self.head(i_h)
        o_h['outputs']['pred_scores'] = data['pred_scores']
        o_h['outputs']['pred_boxes_mask'] = data['pred_boxes_mask']
        o_h['outputs']['pred_boxes'] = data['pred_boxes'] 
        return {'outputs': o_h['outputs'], 'criterion': self.criterion, 'postprocessor': self.postprocessor}

    def train(self, mode=True):
        super(JARViS, self).train(mode)
        if mode and self.config.get('freeze_bn', False):
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)


class Human_Detector(nn.Module):
    def __init__(self, config):
        super(Human_Detector, self).__init__()
        self.config = self.make_cfg(config.config_file)

        self.detector = DefaultTrainer.build_model(self.config)
        if self.config.MODEL.WEIGHTS is not None:
            DetectionCheckpointer(self.detector, save_dir=self.config.OUTPUT_DIR).resume_or_load(
                self.config.MODEL.WEIGHTS, resume=False)
    
    def make_cfg(self, config_file):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.freeze()
        return cfg

    def forward(self, data, inference=False):
        det_inputs = [{'image': image, 'height': image.shape[1], 'width': image.shape[2]} for image in data['images']]
        # train mode 
        if not inference:
            for i in range(len(det_inputs)):
                height, width = det_inputs[i]['image'].shape[1:]
                target = Instances((height, width))
                gt_boxes = data['targets'][i]['gt_boxes'] * torch.tensor([width, height, width, height], dtype=torch.float)
                target.gt_boxes = Boxes(gt_boxes)
                target.gt_classes = data['targets'][i]['gt_classes']          
                det_inputs[i]['instances'] = target
                       
            losses = self.detector(det_inputs)
            return {'loss_dict': losses}            
        # eval mode
        else:
            outputs = self.detector.inference(det_inputs, do_postprocess=False)
            if 'gt_boxes' in data:
                detected_instances = []
                for i in range(len(det_inputs)):
                    height, width = det_inputs[i]['image'].shape[1:]
                    target = Instances((height, width))
                    proposal_boxes = data['gt_boxes'][i] * torch.tensor([width, height, width, height], dtype=torch.float).cuda()
                    target.proposal_boxes = Boxes(proposal_boxes)
                    detected_instances.append(target)
        
                gt_roi_features = self.detector.inference(det_inputs, detected_instances, do_postprocess=False)
            else:
                gt_roi_features = None
            return {'outputs': outputs, 'gt_roi_features': gt_roi_features}