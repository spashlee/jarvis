from PIL import Image
import os
import pickle5 as pickle
import numpy as np
import io
from typing import Optional, List

import torch
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor


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
                    value = int(t['size'] * h / w)
                    oh = 32 * round(value / 32)
                    size = [ow, oh]
                else:
                    oh = t['size']
                    value = int(t['size'] * w / h)
                    ow = 32 * round(value / 32)
                    size = [ow, oh]
            else:
                size = t['size']
            continue

        if t['transform'] == 'RandomCrop':
            w, h = size
            size = [t['size']] * 2
            
            x1 = t['pos_x'] * (w - t['size'])
            y1 = t['pos_y'] * (h - t['size'])
            x2 = x1 + t['size']
            y2 = y1 + t['size']

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

        elif t['transform'] == 'FullResolutionCrop': 
            w, h = size
            short_side = min(size)
            long_side = max(size)
            if (long_side / short_side) <= 2:
                num_crop = 3
            elif (long_side / short_side) <= 3:
                num_crop = 5
            else:
                num_crop = 7

            step_len = (long_side - short_side) / (num_crop - 1)

            if np.argmin(size) == 0:
                x1 = 0
                y1 = int(round(step_len * t['crop_idx']))
                x2 = t['size']
                y2 = y1 + t['size']
            elif np.argmin(size) == 1:
                x1 = int(round(step_len * t['crop_idx']))
                y1 = 0
                x2 = x1 + t['size']
                y2 = t['size']

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

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class AVADataLoader(data.DataLoader):
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
        super(AVADataLoader, self).__init__(
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
        
        images = [_['img'] for _ in batch]
        images = nested_tensor_from_tensor_list(images)        
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'images': images,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output

    
class AVA(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 det_spatial_transform=None):
        
        with open(annotation_path, 'rb') as f:
            self.data, self.idx_to_class = pickle.load(f)

        self.root_path = root_path    
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.det_spatial_transform = det_spatial_transform
        
    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            init_size = clip[0].size[:2]
            params = self.spatial_transform.randomize_parameters(init_size)
            aug_info = get_aug_info(init_size, params)
            
            clip = [self.spatial_transform(img) for img in clip]
        else:
            aug_info = None

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, aug_info

    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.data[index]['video'])
        frame_format = self.data[index]['format_str']
        start_frame = self.data[index]['start_frame']
        n_frames = self.data[index]['n_frames']
        key_frame = self.data[index]['mid_frame']
        mid_time = str(self.data[index]['time'])
        
        frame_indices = list(range(start_frame, start_frame + n_frames))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        else:
            # Use only one ket frame
            frame_indices = [key_frame]

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

        # Key frame image path
        image_path = os.path.join(path, frame_format%key_frame)
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
        except BaseException as e:
            raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
        key_frame_img = img

        if aug_info['flip']:
            key_frame_img = key_frame_img.transpose(Image.FLIP_LEFT_RIGHT)

        key_frame_img = self.det_spatial_transform(key_frame_img)

        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        return {'clip': clip, 'img': key_frame_img, 'aug_info': aug_info, 'label': target, 
                'video_name': video_name, 'mid_time': mid_time}

    def __len__(self):
        return len(self.data)

    
class AVAmulticropDataLoader(AVADataLoader):
    def _collate_fn(self, batch):
        set_clips, set_aug_info = [], []
        for idx in range(len(batch[0]['clip'])):
            clips, aug_info = [], []
            for i in range(len(batch[0]['clip'][idx])):
                clip, pad_ratios = batch_pad([_['clip'][idx][i] for _ in batch])
                clips.append(clip)
                cur_aug_info = []
                for datum, pad_ratio in zip(batch, pad_ratios):
                    print(datum['aug_info'])
                    datum['aug_info'][idx][i]['pad_ratio'] = pad_ratio
                    cur_aug_info.append(datum['aug_info'][idx][i])
                aug_info.append(cur_aug_info)
            set_clips.append(clips)
            set_aug_info.append(aug_info)

        clips_mask = [_['clips_mask'] for _ in batch] 
        images = [_['key_frame_image'] for _ in batch]
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': set_clips,
            'clips_mask': clips_mask,
            'key_frame_images': images,
            'aug_info': set_aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output
    
    
class AVAmulticrop(AVA):
    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            assert isinstance(self.spatial_transform, list)
                      
            init_size = clip[0].size[:2]
            clips, aug_info = [], []
            for st in self.spatial_transform:
                params = st.randomize_parameters()
                aug_info.append(get_aug_info(init_size, params))
            
                clips.append(torch.stack([st(img) for img in clip], 0).permute(1, 0, 2, 3))
        else:
            aug_info = [None]
            clips = [torch.stack(clip, 0).permute(1, 0, 2, 3)]
        return clips, aug_info


class AVAtempmulticrop(AVA):
    def __init__(self,
                 root_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 num_clips=5):

        self.num_clips = num_clips
        super(AVAtempmulticrop, self).__init__(root_path, annotation_path, spatial_transform, temporal_transform)

    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.data[index]['video'])
        frame_format = self.data[index]['format_str']
        start_frame = self.data[index]['start_frame']
        n_frames = self.data[index]['n_frames']
        key_frame = self.data[index]['mid_frame']
        mid_time = str(self.data[index]['time'])
        frame_rate = self.data[index]['frame_rate']

        clips = []
        aug_infos = []
        for idx in range(self.num_clips):
            clip_start_frame = int(start_frame + frame_rate * (idx - (self.num_clips // 2)))
            if clip_start_frame < 0 or (clip_start_frame + n_frames) > 27001:
                continue

            frame_indices = list(range(clip_start_frame, clip_start_frame + n_frames))
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            else:
                # Use only one ket frame
                frame_indices = [key_frame]

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
            clips.append(clip)
            aug_infos.append(aug_info)

        # Key frame image path
        image_path = os.path.join(path, frame_format%key_frame)
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
        except BaseException as e:
            raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
        key_frame_img = img

        if aug_info['flip']:
            key_frame_img = key_frame_img.transpose(Image.FLIP_LEFT_RIGHT)

        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        return {'clip': clips, 'key_frame_image': key_frame_img, 'aug_info': aug_infos, 'label': target, 
                'video_name': video_name, 'mid_time': mid_time}


class AVAFullResolutionCrop(AVA):

    def _spatial_transform(self, clip):
        init_size = clip[0].size[:2]

        params = self.spatial_transform.randomize_parameters()

        short_side = min(init_size)
        long_side = max(init_size)
        if (long_side / short_side) <= 2:
            num_crop = 3
        elif (long_side / short_side) <= 3:
            num_crop = 5
        else:
            num_crop = 7

        aug_info = []
        for crop_idx in range(num_crop):
            for param in params:
                if param is None:
                    continue
                if param['transform'] == 'FullResolutionCrop':
                    param['crop_idx'] = crop_idx

            aug_info.append(get_aug_info(init_size, params))
        
        clips = [torch.stack(self.spatial_transform(img), 0) for img in clip]

        clips = torch.stack(clips, 0).permute(1, 2, 0, 3, 4)
        return clips, aug_info

    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.data[index]['video'])
        frame_format = self.data[index]['format_str']
        start_frame = self.data[index]['start_frame']
        n_frames = self.data[index]['n_frames']
        key_frame = self.data[index]['mid_frame']
        mid_time = str(self.data[index]['time'])

        frame_indices = list(range(start_frame, start_frame + n_frames))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        else:
            # Use only one ket frame
            frame_indices = [key_frame]

        clip = []
        for i in range(len(frame_indices)):
            image_path = os.path.join(path, frame_format%frame_indices[i])
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
            
            clip.append(img)

        clips, aug_info = self._spatial_transform(clip)

        # Key frame image path
        image_path = os.path.join(path, frame_format%key_frame)
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
        except BaseException as e:
            raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
        key_frame_img = img

        if aug_info[0]['flip']:
            key_frame_img = key_frame_img.transpose(Image.FLIP_LEFT_RIGHT)

        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        return {'clip': clips, 'key_frame_image': key_frame_img, 'aug_info': aug_info, 'label': target, 
                'video_name': video_name, 'mid_time': mid_time}


class AVAscore(AVA):
    def __init__(self,
                 mode,
                 root_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 det_spatial_transform = None,
                 num_clips=5):
        
        self.mode = mode
        self.num_clips = num_clips
        super(AVAscore, self).__init__(root_path, annotation_path, spatial_transform, temporal_transform, det_spatial_transform)

    def __getitem__(self, index):
        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        path = os.path.join(self.root_path, 'AVA_dataset', video_name)
        frame_format = self.data[index]['format_str']
        start_frame = self.data[index]['start_frame']
        n_frames = self.data[index]['n_frames']
        key_frame = self.data[index]['mid_frame']
        mid_time = self.data[index]['time']

        video_path = os.path.join(self.root_path, 'ViT-B_F-RCNN_v2.2_35.3_longterm_11clips_{}'.format(self.mode), video_name)
        # if self.mode == 'eval':
        #     video_path = os.path.join(self.root_path, 'longterm_scores', 'ViT-H_AVA_v2.2_41.7_11clips_{}'.format(self.mode), video_name)
        # else:
        #     video_path = os.path.join(self.root_path, 'longterm_11clips_{}'.format(self.mode), video_name)

        clip_path = os.path.join(video_path, '{}.pkl'.format(mid_time))
        with open(clip_path, 'rb') as f: 
            clip_dict = pickle.load(f)
        pred_boxes = clip_dict['pred_boxes']
        pred_scores = clip_dict['pred_scores']

        clip_masks = []
        clips_scores = []
        if self.num_clips == 3:
            clip_list = [4,5,6]
        elif self.num_clips == 5:
            clip_list = [3,4,5,6,7]
        elif self.num_clips == 7:
            clip_list = [2,3,4,5,6,7,8]
        elif self.num_clips == 9:
            clip_list = [1,2,3,4,5,6,7,8,9]     
        elif self.num_clips == 11:
            # clip_list = [0,1,2,3,4,5,6,7,8,9,10]
            clip_list = [2,3,4,5,6,7,8,9,10,11,12]
        elif self.num_clips == 15:
            clip_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]            
        else:
            clip_list = [5]            

        for idx in clip_list:
            if str(idx) in clip_dict:
                clips_scores.append(clip_dict[str(idx)])
                clip_masks.append(0)
            else:
                clips_scores.append(clip_dict['7'])
                clip_masks.append(1)
        clips_scores = torch.tensor(clips_scores, dtype=torch.float)
        return {'clip_scores': clips_scores, 'clip_masks': clip_masks, 'pred_boxes': pred_boxes, 'pred_scores': pred_scores, 'label': target,
                'video_name': video_name, 'mid_time': str(mid_time), 'start_frame': str(start_frame)}

    def __len__(self):
        return len(self.data)


class AVAscoreDataLoader(AVADataLoader):
    def _collate_fn(self, batch):

        clip_scores = torch.stack([_['clip_scores'] for _ in batch])
        clip_masks = torch.tensor([_['clip_masks'] for _ in batch], dtype=torch.bool)
        pred_boxes = torch.tensor([_['pred_boxes'] for _ in batch], dtype=torch.float)
        pred_scores = torch.tensor([_['pred_scores'] for _ in batch], dtype=torch.float)

        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        start_frames = [_['start_frame'] for _ in batch]
        
        output = {
            'clip_scores': clip_scores,
            'clip_masks': clip_masks,
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times,
            'start_frames' : start_frames
        }
        return output


class AVAdemo(AVA):
    def __getitem__(self, index):
        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        path = os.path.join(self.root_path, video_name)
        frame_format = self.data[index]['format_str']
        key_frame = self.data[index]['mid_frame']
        frame_rate = self.data[index]['frame_rate']
        mid_time = self.data[index]['time']

        frame_indices = list(range(key_frame - int(frame_rate//2), key_frame + int(frame_rate//2)))
        clip = []
        for i in range(len(frame_indices)):
            image_path = os.path.join(path, frame_format%frame_indices[i])
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
            clip.append(img)

        return {'clip': clip, 'label': target, 'key_frame': key_frame,
                'video_name': video_name, 'mid_time': str(mid_time)}

    def __len__(self):
        return len(self.data)

class AVAdemoDataLoader(AVADataLoader):
    def _collate_fn(self, batch):
        clips = [_['clip'] for _ in batch]
        key_frames = [_['key_frame'] for _ in batch]
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'key_frames': key_frames,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output 