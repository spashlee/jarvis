from PIL import Image
import os
import pickle5 as pickle
import numpy as np
import io

import torch
import torch.nn.functional as F
import torch.utils.data as data

from .spatial_transforms import *

# bounding box order: [left, top, right, bottom]
# size order: [width, height]
def get_aug_info(init_size, params):
    size = init_size
    bbox = [0.0, 0.0, 1.0, 1.0]
    flip = False
    
    for t in params:
        if t is None:
            continue
            
        if t['transform'] == 'ColorJitter':
            continue

        if t['transform'] == 'RandomHorizontalFlip':
            if t['flip']:
                flip = not flip
            continue
        
        if t['transform'] == 'Scale':
            if isinstance(t['size'], int):
                w, h = size
                if (w <= h and w == t['size']) or (h <= w and h == t['size']):
                    continue
                if w < h:
                    ow = t['size']
                    oh = int(t['size'] * h / w)
                    size = [ow, oh]
                else:
                    oh = t['size']
                    ow = int(t['size'] * w / h)
                    size = [ow, oh]
            else:
                size = t['size']
            continue
            
        if t['transform'] == 'CenterCrop':
            w, h = size
            size = t['size']
            
            x1 = int(round((w - size[0]) / 2.))
            y1 = int(round((h - size[1]) / 2.))
            x2 = x1 + size[0]
            y2 = y1 + size[1]
            
        elif t['transform'] == 'CornerCrop':
            w, h = size
            size = [t['size']] * 2

            if t['crop_position'] == 'c':
                th, tw = (t['size'], t['size'])
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif t['crop_position'] == 'tl':
                x1 = 0
                y1 = 0
                x2 = t['size']
                y2 = t['size']
            elif t['crop_position'] == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = t['size']
            elif t['crop_position'] == 'bl':
                x1 = 0
                y1 = h - t['size']
                x2 = t['size']
                y2 = h
            elif t['crop_position'] == 'br':
                x1 = w - t['size']
                y1 = h - t['size']
                x2 = w
                y2 = h
            
        elif t['transform'] == 'ScaleJitteringRandomCrop':
            min_length = min(size[0], size[1])
            jitter_rate = float(t['scale']) / min_length
            
            w = int(jitter_rate * size[0])
            h = int(jitter_rate * size[1])
            size = [t['size']] * 2
            
            x1 = t['pos_x'] * (w - t['size'])
            y1 = t['pos_y'] * (h - t['size'])
            x2 = x1 + t['size']
            y2 = y1 + t['size']
            
        dl = float(x1) / w * (bbox[2] - bbox[0])
        dt = float(y1) / h * (bbox[3] - bbox[1])
        dr = float(x2) / w * (bbox[2] - bbox[0])
        db = float(y2) / h * (bbox[3] - bbox[1])
        
        if flip:
            bbox = [bbox[2] - dr, bbox[1] + dt, bbox[2] - dl, bbox[1] + db]
        else:
            bbox = [bbox[0] + dl, bbox[1] + dt, bbox[0] + dr, bbox[1] + db]

    return {'init_size': init_size, 'crop_box': bbox, 'flip': flip}

def batch_pad(images, alignment=1, pad_value=0):
    max_img_h = max([_.size(-2) for _ in images])
    max_img_w = max([_.size(-1) for _ in images])
    target_h = int(np.ceil(max_img_h / alignment) * alignment)
    target_w = int(np.ceil(max_img_w / alignment) * alignment)
    padded_images, pad_ratios = [], []
    for image in images:
        src_h, src_w = image.size()[-2:]
        pad_size = (0, target_w - src_w, 0, target_h - src_h)
        padded_images.append(F.pad(image, pad_size, 'constant', pad_value).data)
        pad_ratios.append([target_w / src_w, target_h / src_h])
    return torch.stack(padded_images), pad_ratios


class UCFDataLoader(data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 **kwargs):
        super(UCFDataLoader, self).__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self._collate_fn, 
            pin_memory=pin_memory, 
            drop_last=drop_last,
            **kwargs
        )

    def _collate_fn(self, batch):
        clips = [_['clip'] for _ in batch]
        clips, pad_ratios = batch_pad(clips)
        aug_info = []
        for datum, pad_ratio in zip(batch, pad_ratios):
            datum['aug_info']['pad_ratio'] = pad_ratio
            aug_info.append(datum['aug_info'])

        det_clips = [_['det_clip'] for _ in batch]
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'det_clips': det_clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output

    
class UCF(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 mode='train',
                 clip_len=64):
        
        with open(annotation_path, 'rb') as f:
            self.dataset = pickle.load(f, encoding='latin1')

        self.root_path = root_path    
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.transform = Compose([ToTensor()])
        self.shape = (224, 224)

        labelmap = []
        for i, name in enumerate(self.dataset['labels']):
            labelmap.append({"id": i + 1, "name": name})
        self.idx_to_class = labelmap

        if mode == 'val' or mode == 'test':
            self.dataset_samples = self.dataset['test_videos'][0]
        elif mode == 'train':
            self.dataset_samples = self.dataset['train_videos'][0]

        self.n_frames = {}

        dataset = []
        for vid in self.dataset_samples:
            _, _, files = next(os.walk(os.path.join(self.root_path, vid)))
            self.n_frames[vid] = len(files)            
            for i in range(self.dataset['nframes'][vid]):
                data = {}
                data['video_name'] = vid
                data['frame_id'] = (i + 1)
                data['labels'] = self.load_annotation(vid, (i + 1))
                
                path = os.path.join(self.root_path, vid, '%05d.jpg'%(i + 1))
                if len(data['labels']) == 0:
                    continue
                elif not os.path.exists(path):
                    continue
                dataset.append(data)

        self.dataset = dataset
        self.clip_len = clip_len

    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            init_size = clip[0].size[:2]
            params = self.spatial_transform.randomize_parameters()
            aug_info = get_aug_info(init_size, params)
            
            clip = [self.spatial_transform(img) for img in clip]
        else:
            aug_info = None
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, aug_info

    def load_annotation(self, sample_id, start):
        # print('sample_id',sample_id)

        boxes, classes = [], []

        oh = self.dataset['resolution'][sample_id][0]
        ow = self.dataset['resolution'][sample_id][1]

        for ilabel, tubes in self.dataset['gttubes'][sample_id].items():
            # self.max_person = len(tubes) if self.max_person < len(tubes) else self.max_person
            # self.person_size = len(tubes)
            for t in tubes:
                box_ = t[(t[:, 0] == start), 0:5]

                if len(box_) > 0:
                    box = box_[0]
                    p_x1 = box[1] / ow
                    p_y1 = box[2] / oh
                    p_x2 = box[3] / ow
                    p_y2 = box[4] / oh
                    boxes.append(np.round([p_x1, p_y1, p_x2, p_y2], 4))
                    classes.append(np.clip(ilabel, 0, 24))
        
        targets = []
        for (box, label) in zip(boxes, classes):
            target = {}
            target['bounding_box'] = box
            target["label"] = [label]
            targets.append(target)

        return targets

    def __getitem__(self, index):
        data_dict = self.dataset[index]
        video_name = data_dict['video_name']
        path = os.path.join(self.root_path, video_name)
        mid_frame = data_dict['frame_id']
        target = data_dict['labels']
        frame_format = '%05d.jpg'

        file_count = self.n_frames[video_name]
        p_t = self.clip_len // 2

        # frame_indices = []
        # for i in reversed(range(self.clip_len)):
        #     # make it as a loop
        #     img_frame = mid_frame + p_t - i
        #     if img_frame < 1:
        #         img_frame = 1
        #     elif img_frame > file_count:
        #         img_frame = file_count
        #     frame_indices.append(img_frame)
        
        start = max(mid_frame - p_t, 1)
        end = min(mid_frame + p_t, file_count)
        frame_indices = list(range(start, end))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        else:
            # Use only one ket frame
            frame_indices = [mid_frame]

        clip = []
        for i in range(len(frame_indices)):
            image_path = os.path.join(path, frame_format%frame_indices[i])
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
            
            clip.append(img)
        clip, aug_info = self._spatial_transform(clip)

        frame_indices = []
        det_clip = []
        for i in reversed(range(p_t // 2)):
            # make it as a loop
            img_frame = mid_frame - i
            if img_frame < 1:
                img_frame = 1
            elif img_frame > file_count:
                img_frame = file_count
            frame_indices.append(img_frame)

            image_path = os.path.join(path, frame_format%img_frame)
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))

            img = img.resize(self.shape)

            if aug_info['flip']:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
            img = self.transform(img)
            det_clip.append(img)

        det_clip = torch.stack(det_clip, 0).permute(1, 0, 2, 3)

        return {'clip': clip, 'det_clip': det_clip, 'aug_info': aug_info, 'label': target, 
                'video_name': video_name, 'mid_time': mid_frame}

    def __len__(self):
        return len(self.dataset)
