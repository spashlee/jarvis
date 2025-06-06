"""
Adapted code from:
    @inproceedings{hara3dcnns,
      author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
      title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={6546--6555},
      year={2018},
    }.
"""

import random
import numbers
import collections
import numpy as np
import torch
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Compose several transforms together.
    Args:
        transforms (list of ``Transform`` objects): List of transforms to compose.
    Example:
        >>> Compose([
        >>>     CenterCrop(10),
        >>>     ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be transformed.
        Returns:
            PIL.Image: Transformed image.
        """
        for t in self.transforms:
            if isinstance(img, list):
                img_list = []
                for img_ in img:
                    img_list.append(t(img_))
                img = img_list
            else:
                img = t(img)
        return img

    def randomize_parameters(self, size=None):
        param_info = []
        for t in self.transforms:
            param_info += t.randomize_parameters(size)
        return param_info

    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self.transforms) + ')'


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Convert a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range 
    [0.0, 255.0 / norm_value].
    Args:
        norm_value (float, optional): Normalization constant.
    """

    def __init__(self, norm_value=255.):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self, size=None):
        return [None]

    def __repr__(self):
        return '{self.__class__.__name__}(norm_value={self.norm_value})'.format(self=self)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std.
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self, size=None):
        return [None]

    def __repr__(self):
        return '{self.__class__.__name__}(mean={self.mean}, std={self.std})'.format(self=self)


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        resize (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width).
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
        max_ratio (float, optional): If not None, denotes maximum allowed aspect
            ratio after rescaling the input.
    """

    def __init__(self, resize, interpolation=Image.BILINEAR, max_ratio=None):
        assert isinstance(resize,
                          int) or (isinstance(resize, collections.Iterable) and
                                   len(resize) == 2)
        self.resize = resize
        self.interpolation = interpolation
        self.max_ratio = max_ratio

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        w, h = img.size
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                value = int(self.size * h / w)
                oh = 32 * round(value / 32)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                value = int(self.size * w / h)
                ow = 32 * round(value / 32)
                return img.resize((ow, oh), self.interpolation)
        else:
            if w == self.size[0] and h == self.size[1]:
                return img
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self, size=None):
        if isinstance(self.resize, int) and size and self.max_ratio:
            ratio = max(size[0] / size[1], size[1] / size[0])
            if ratio > self.max_ratio:
                if size[0] > size[1]:
                    value = int(self.resize * self.max_ratio)
                    value = 32 * round(value / 32)
                    resize = (value, self.resize)
                else:
                    value = int(self.resize * self.max_ratio)
                    value = 32 * round(value / 32)                    
                    resize = (self.resize, value)
            else:
                resize = self.resize
        else:
            resize = self.resize
        self.size = resize
        return [{'transform': 'Scale', 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(resize={self.resize}, interpolation={self.interpolation}, max_ratio={self.max_ratio})'.format(self=self)


class RandomScale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        resize (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width).
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
        max_ratio (float, optional): If not None, denotes maximum allowed aspect
            ratio after rescaling the input.
    """

    def __init__(self, resize, interpolation=Image.BILINEAR, max_ratio=None):
        self.resize = resize
        self.interpolation = interpolation
        self.max_ratio = max_ratio

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        w, h = img.size
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            if w == self.size[0] and h == self.size[1]:
                return img
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self, size=None):
        self.size = random.choice(self.resize)
        return [{'transform': 'Scale', 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(resize={self.resize}, interpolation={self.interpolation}, max_ratio={self.max_ratio})'.format(self=self)


class RandomCrop(object):
    """Ramdomly crop the given PIL.Image.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        x1 = self.tl_x * (w - self.size)
        y1 = self.tl_y * (h - self.size)
        x2 = x1 + self.size
        y2 = y1 + self.size
        return img.crop((x1, y1, x2, y2))

    def randomize_parameters(self, size=None):
        self.tl_x = random.random()
        self.tl_y = random.random()
        return [{'transform': 'RandomCrop', 'pos_x': self.tl_x,
                 'pos_y': self.tl_y, 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size})'.format(self=self)


class CenterCrop(object):
    """Crop the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self, size=None):
        return [{'transform': 'CenterCrop', 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size})'.format(self=self)


class CornerCrop(object):
    """Crop the given PIL.Image at some corner or the center.
    Args:
        size (int): Desired output size of the square crop.
        crop_position (str, optional): Designate the position to be cropped. 
            If is None, a random position will be selected from five choices.
    """

    def __init__(self, size, crop_position=None):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self, size=None):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]
        return [{'transform': 'CornerCrop', 'crop_position': self.crop_position,
                 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, crop_position={self.crop_position})'.format(self=self)


class MaxSizeCut(object):
    """Crop the given PIL.Image if the input image exceeds the maximum size.
    Args:
        size (int): Desired output size of the square crop.
        crop_position (str, optional): Designate the position to be cropped. 
            If is None, a random position will be selected from five choices.
    """

    def __init__(self, size):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        if w <= self.size and h <= self.size:
            return img

        if w > self.size:
            x1 = self.tl * (w - self.size)
            y1 = 0
            x2 = x1 + self.size
            y2 = h
        elif h > self.size:
            x1 = 0
            y1 = self.tl * (h - self.size)
            x2 = w
            y2 = y1 + self.size            

        return img.crop((x1, y1, x2, y2))

    def randomize_parameters(self, size=None):
        self.tl = random.random()
        return [{'transform': 'MaxSizeCut', 'pos': self.tl, 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size})'.format(self=self)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a given probability.
    Args:
        p (float, optional): Probability of flipping.
    """

    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self, size=None):
        p = random.random()
        if p < self.prob:
            self.flip = True
        else:
            self.flip = False
        return [{'transform': 'RandomHorizontalFlip', 'flip': self.flip}]

    def __repr__(self):
        return '{self.__class__.__name__}(prob={self.prob})'.format(self=self)

        
class ScaleJitteringRandomCrop(object):
    """Randomly rescale the given PIL.Image and then take a random crop.
    Args:
        min_scale (int): Minimum scale for random rescaling.
        max_scale (int): Maximum scale for random rescaling.
        size (int): Desired output size of the square crop.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    """

    def __init__(self, min_scale, max_scale, size, interpolation=Image.BILINEAR):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rescaled and cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        min_length = min(img.size[0], img.size[1])
        jitter_rate = float(self.scale) / min_length

        image_width = int(jitter_rate * img.size[0])
        image_height = int(jitter_rate * img.size[1])
        img = img.resize((image_width, image_height), self.interpolation)

        x1 = self.tl_x * (image_width - self.size)
        y1 = self.tl_y * (image_height - self.size)
        x2 = x1 + self.size
        y2 = y1 + self.size

        return img.crop((x1, y1, x2, y2))

    def randomize_parameters(self, size=None):
        self.scale = random.randint(self.min_scale, self.max_scale)
        self.tl_x = random.random()
        self.tl_y = random.random()
        return [{'transform': 'ScaleJitteringRandomCrop', 'pos_x': self.tl_x,
                 'pos_y': self.tl_y, 'scale': self.scale, 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(min_scale={self.min_scale}, max_scale={self.max_scale}, size={self.size}, interpolation={self.interpolation})'.format(self=self)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        """
        Perfrom a color jittering on the input images. The channels of images
        should be in order BGR.
        Args:
            images (tensor): images to perform color jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
            img_brightness (float): jitter ratio for brightness.
            img_contrast (float): jitter ratio for contrast.
            img_saturation (float): jitter ratio for saturation.
        Returns:
            images (tensor): the jittered images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        jitter = []
        if brightness != 0:
            jitter.append("brightness")
        if contrast != 0:
            jitter.append("contrast")
        if saturation != 0:
            jitter.append("saturation")
        self.jitter = jitter

    def blend(self, image1, image2, alpha):
        """
        Blend two images with a given weight alpha.
        Args:
            images1 (tensor): the first images to be blended, the dimension is
                `channel` x `height` x `width`.
            images2 (tensor): the second images to be blended, the dimension is
                `channel` x `height` x `width`.
            alpha (float): the blending weight.
        Returns:
            (tensor): blended images, the dimension is
                `channel` x `height` x `width`.
        """
        bound = 1.0 if image1.max() <= 1.0 else 255.0
        return (image1 * alpha + image2 * (1 - alpha)).clamp(0, bound)

    def grayscale(self, image):
        """
        Get the grayscale for the input images. The channels of images should be
        in order RGB.
        Args:
            images (tensor): the input images for getting grayscale. Dimension is
                `channel` x `height` x `width`.
        Returns:
            img_gray (tensor): blended images, the dimension is
                `channel` x `height` x `width`.
        """
        image_ = image.clone()
        # R -> 0.299, G -> 0.587, B -> 0.114.
        gray_channel = (
            0.299 * image_[0] + 0.587 * image_[1] + 0.114 * image_[2]
        )
        image_[0] = gray_channel
        image_[1] = gray_channel
        image_[2] = gray_channel
        return image_

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rescaled and cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if len(self.jitter) > 0:
            order = np.random.permutation(np.arange(len(self.jitter)))
            for idx in range(0, len(self.jitter)):
                if self.jitter[order[idx]] == "brightness":                  
                    img_bright = torch.zeros(img.shape)
                    img = self.blend(img, img_bright, self.brightness_scale)
                
                elif self.jitter[order[idx]] == "contrast":
                    img_gray = self.grayscale(img)
                    img_gray = torch.mean(img_gray, dim=(-3, -2, -1), keepdim=True)
                    img = self.blend(img, img_gray, self.contrast_scale)
               
                elif self.jitter[order[idx]] == "saturation":
                    img_gray = self.grayscale(img)
                    img = self.blend(img, img_gray, self.saturation_scale)
        return img

    def randomize_parameters(self, size=None):
        self.brightness_scale = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        self.contrast_scale = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        self.saturation_scale = 1.0 + np.random.uniform(-self.saturation, self.saturation)

        return [{'transform': 'ColorJitter', 'brightness_scale': self.brightness_scale,
                 'contrast_scale': self.contrast_scale, 'saturation_scale': self.saturation_scale}]

    def __repr__(self):
        return '{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, jitter={self.jitter})'.format(self=self)