import torch
import torch.nn as nn

from .utils import bbox_jitter, get_bbox_after_aug

__all__ = ['ava']

class AVANeck(nn.Module):
    def __init__(self, aug_threshold=0., bbox_jitter=None, num_classes=60, multi_class=True, autocast=False):
        super(AVANeck, self).__init__()
        
        # threshold on preserved ratio of bboxes after cropping augmentation
        self.aug_threshold = aug_threshold
        # config for bbox jittering
        self.bbox_jitter = bbox_jitter

        self.num_classes = num_classes
        self.multi_class = multi_class

        if autocast:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    # data: aug_info, labels, filenames, mid_times
    # returns: num_rois, rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, bboxes, bbox_ids
    def forward(self, data):
        total_labels = []
        total_bboxes = []
        ori_bbox = []
        sizes_before_padding, filenames, mid_times = [], [], []

        for idx in range(len(data['aug_info'])):
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']

            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])

            label_set = []
            bbox_set = []
            for label in data['labels'][idx]:                
                # no bbox jittering during evaluation
                bbox_list = [label['bounding_box']]

                for b in bbox_list:
                    bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)

                    if bbox is None:
                        continue
                    if self.multi_class:                        
                        filenames.append(data['filenames'][idx])
                        mid_times.append(data['mid_times'][idx])
                        # bboxes.append(label['bounding_box'])
                        bbox_set.append(torch.tensor(bbox, dtype=self.dtype))
                        ori_bbox.append(b)

                        ret = torch.zeros(self.num_classes+1)
                        ret.put_(torch.LongTensor(label['label']), 
                                torch.ones(len(label['label'])))
                        label_set.append(ret[:self.num_classes].to(dtype=self.dtype))
                    else:
                        for class_idx in label['label']:
                            
                            filenames.append(data['filenames'][idx])
                            mid_times.append(data['mid_times'][idx])
                            # bboxes.append(label['bounding_box'])
                            bbox_set.append(torch.tensor(bbox, dtype=self.dtype))
                            
                            ret = torch.zeros(self.num_classes)
                            if not class_idx >= 60:
                                ret[class_idx] = 1.

                            label_set.append(ret.to(dtype=self.dtype))

            if len(bbox_set) == 0:
                bbox_set = torch.tensor(bbox_set, dtype=self.dtype).cuda()                
                label_set = torch.tensor(label_set, dtype=self.dtype).cuda()
            else:
                bbox_set = torch.stack(bbox_set, dim=0).cuda()
                label_set = torch.stack(label_set, dim=0).cuda()

            total_labels.append(label_set)
            total_bboxes.append(bbox_set)

        ori_bbox = torch.tensor(ori_bbox, dtype=self.dtype).cuda()
        targets = [{'labels': l, 'boxes': b} for l, b in zip(total_labels, total_bboxes)]
        return {'targets': targets,
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': ori_bbox}

    
def ava(**kwargs):
    model = AVANeck(**kwargs)
    return model
