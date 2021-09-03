import random

import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
import PIL


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, masks, im_info, flipped):
        for t in self.transforms:
            image, boxes, masks, im_info, flipped = t(image, boxes, masks, im_info, flipped)
        return image, boxes, masks, im_info, flipped

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, image, boxes, masks, im_info, flipped):
        origin_size = im_info[:2]
        size = self.get_size(origin_size)
        if image is not None:
            image = F.resize(image, (size[1], size[0]))

        ratios = [size[0] * 1.0 / origin_size[0], size[1] * 1.0 / origin_size[1]]
        if boxes is not None:
            boxes[:, [0, 2]] *= ratios[0]
            boxes[:, [1, 3]] *= ratios[1]
        im_info[0], im_info[1] = size
        im_info[2], im_info[3] = ratios
        return image, boxes, masks, im_info, flipped


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, masks, im_info, flipped):
        if random.random() < self.prob:
            w, h = im_info[:2]
            if image is not None:
                image = F.hflip(image)
            if boxes is not None:
                boxes[:, [0, 2]] = w - 1 - boxes[:, [2, 0]]
            if masks is not None:
                masks = torch.as_tensor(masks.numpy()[:, :, ::-1].tolist())
            flipped = not flipped
        return image, boxes, masks, im_info, flipped


class ToTensor(object):
    def __call__(self, image, boxes, masks, im_info, flipped):
        return F.to_tensor(image) if image is not None else image, boxes, masks, im_info, flipped


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, boxes, masks, im_info, flipped):
        if image is not None:
            if self.to_bgr255:
                image = image[[2, 1, 0]] * 255
            image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, masks, im_info, flipped


class FixPadding(object):
    def __init__(self, min_size, max_size, pad=0):
        self.min_size = min_size
        self.max_size = max_size
        self.pad = pad

    def __call__(self, image, boxes, masks, im_info, flipped):

        if image is not None:
            # padding to fixed size for determinacy
            c, h, w = image.shape
            if h <= w:
                h1 = self.min_size
                w1 = self.max_size
            else:
                h1 = self.max_size
                w1 = self.min_size
            padded_image = image.new_zeros((c, h1, w1)).fill_(self.pad)
            padded_image[:, :h, :w] = image
            image = padded_image

        return image, boxes, masks, im_info, flipped

class RandomAffine(object):

    def __init__(self, prob, translate, scale):
        '''
        prob: prob to apply affine;
        translate: (max_translate_on_x, max_translate_on_y);
        scale:(min_scale, max_scale)
        '''

        self.prob = prob
        assert isinstance(translate, tuple) and len(translate)==2
        self.translate = translate

        assert isinstance(scale, tuple) and len(scale)==2
        self.scale = scale
        
        self.degrees = (0, 0)
        self.shear = None
        self.resample = PIL.Image.BILINEAR
        self.fillcolor = 0
    
    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear
    def __call__(self, image, boxes, masks, im_info, flipped):
        prob = random.random()
        if prob < self.prob:
            #print('PIL size',image.size)
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
            image =  F.affine(image, *ret, resample=self.resample, fillcolor=self.fillcolor)
            
            angle, translations, scale, shear = ret

            im_info[2] *= scale
            im_info[3] *= scale

            #boxes *= scale
            w, h = image.size
            cx = w*0.5 + 0.5
            cy = h*0.5 +0.5
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - cx) * scale + cx + translations[0]
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - cy) * scale + cy + translations[1]
        #print('prob:', prob, 'self.prob:', self.prob)
        return image, boxes, masks, im_info, flipped

            

