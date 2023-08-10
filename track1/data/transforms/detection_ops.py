# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral

import traceback
import uuid
import random
import math
import numpy as np
import os
import copy
import logging
import cv2
from PIL import Image, ImageDraw
import pickle
from paddle.fluid.dataloader.collate import default_collate_fn

from modeling import bbox_utils

from ppdet.data.transform import RandomResize, RandomResizeCrop, Pad
import logging
logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                # op_cls = getattr(detection_ops, k)
                op_cls = eval(k)
                # print('v:', v)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


class ImageError(ValueError):
    pass


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence) and self.__class__.__name__ == 'Decode':  #只有deocde走这个通路
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


class Decode(BaseOperator):
    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        im = im.transpose((2, 0, 1))
        sample['image'] = im
        return sample


class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1], is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.is_scale = is_scale
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std

        sample['image'] = im
        return sample


class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                    width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(
                    sample['gt_keypoint'], width)

            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]

            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                         width)

            sample['flipped'] = True
            sample['image'] = im
        return sample


class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def apply(self, sample, context):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        height, width, _ = im.shape
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample


class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class RandomSelect(BaseOperator):
    """
    Randomly choose a transformation between transforms1 and transforms2,
    and the probability of choosing transforms1 is p.

    The code is based on https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py

    """

    def __init__(self, transforms1, transforms2, p=0.5):
        super(RandomSelect, self).__init__()
        self.transforms1 = Compose(transforms1)
        self.transforms2 = Compose(transforms2)
        self.save_mm = False
        for ts in transforms1: #如果transforms1含有马赛克
            if 'Mosaic' in ts:
                self.save_mm = True  #节省内存
        self.p = p

    def apply(self, sample, context=None):
        if random.random() < self.p:
            return self.transforms1(sample)
        else:
            if isinstance(sample, Sequence) and self.save_mm:
                sample = sample[0]
            return self.transforms2(sample)


class RandomShortSideResize(BaseOperator):
    def __init__(self,
                 short_side_sizes,
                 max_size=None,
                 interp=cv2.INTER_LINEAR,
                 random_interp=False):
        """
        Resize the image randomly according to the short side. If max_size is not None,
        the long side is scaled according to max_size. The whole process will be keep ratio.
        Args:
            short_side_sizes (list|tuple): Image target short side size.
            max_size (int): The size of the longest side of image after resize.
            interp (int): The interpolation method.
            random_interp (bool): Whether random select interpolation method.
        """
        super(RandomShortSideResize, self).__init__()

        assert isinstance(short_side_sizes,
                          Sequence), "short_side_sizes must be List or Tuple"

        self.short_side_sizes = short_side_sizes
        self.max_size = max_size
        self.interp = interp
        self.random_interp = random_interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

    def get_size_with_aspect_ratio(self, image_shape, size, max_size=None):
        h, w = image_shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def resize(self,
               sample,
               target_size,
               max_size=None,
               interp=cv2.INTER_LINEAR):
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        target_size = self.get_size_with_aspect_ratio(im.shape[:2], target_size,
                                                      max_size)
        im_scale_y, im_scale_x = target_size[1] / im.shape[0], target_size[
            0] / im.shape[1]

        sample['image'] = cv2.resize(im, target_size, interpolation=interp)
        sample['im_shape'] = np.asarray(target_size[::-1], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im.shape[:2],
                                                [im_scale_x, im_scale_y])
        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                target_size,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm, target_size, interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox.astype('float32')

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        target_size = random.choice(self.short_side_sizes)
        interp = random.choice(
            self.interps) if self.random_interp else self.interp

        return self.resize(sample, target_size, self.max_size, interp)


class RandomSizeCrop(BaseOperator):
    """
    Cut the image randomly according to `min_size` and `max_size`
    """

    def __init__(self, min_size, max_size):
        super(RandomSizeCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

        from paddle.vision.transforms.functional import crop as paddle_crop
        self.paddle_crop = paddle_crop

    @staticmethod
    def get_crop_params(img_shape, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img_shape (list|tuple): Image's height and width.
            output_size (list|tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (h, w)))

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        return i, j, th, tw

    def crop(self, sample, region):
        image_shape = sample['image'].shape[:2]
        sample['image'] = self.paddle_crop(sample['image'], *region)

        keep_index = None
        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], region)
            bbox = sample['gt_bbox'].reshape([-1, 2, 2])
            area = (bbox[:, 1, :] - bbox[:, 0, :]).prod(axis=1)
            keep_index = np.where(area > 0)[0]
            sample['gt_bbox'] = sample['gt_bbox'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 4], dtype=np.float32)
            sample['gt_class'] = sample['gt_class'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 1], dtype=np.float32)
            if 'gt_score' in sample:
                sample['gt_score'] = sample['gt_score'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)
            if 'is_crowd' in sample:
                sample['is_crowd'] = sample['is_crowd'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], region,
                                                image_shape)
            if keep_index is not None:
                sample['gt_poly'] = sample['gt_poly'][keep_index]
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            i, j, h, w = region
            sample['gt_segm'] = sample['gt_segm'][:, i:i + h, j:j + w]
            if keep_index is not None:
                sample['gt_segm'] = sample['gt_segm'][keep_index]

        return sample

    def apply_bbox(self, bbox, region):
        i, j, h, w = region
        region_size = np.asarray([w, h])
        crop_bbox = bbox - np.asarray([j, i, j, i])
        crop_bbox = np.minimum(crop_bbox.reshape([-1, 2, 2]), region_size)
        crop_bbox = crop_bbox.clip(min=0)
        return crop_bbox.reshape([-1, 4]).astype('float32')

    def apply_segm(self, segms, region, image_shape):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        i, j, h, w = region
        crop = [j, i, j + w, i + h]
        height, width = image_shape
        crop_segms = []
        for segm in segms:
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        h = random.randint(self.min_size,
                           min(sample['image'].shape[0], self.max_size))
        w = random.randint(self.min_size,
                           min(sample['image'].shape[1], self.max_size))

        region = self.get_crop_params(sample['image'].shape[:2], [h, w])
        return self.crop(sample, region)


class PadMaskBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
        return_pad_mask (bool): If `return_pad_mask = True`, return
            `pad_mask` for transformer.
    """

    def __init__(self, pad_to_stride=0, return_pad_mask=False):
        super(PadMaskBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.return_pad_mask = return_pad_mask

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm
            if self.return_pad_mask:
                padding_mask = np.zeros(
                    (max_shape[1], max_shape[2]), dtype=np.float32)
                padding_mask[:im_h, :im_w] = 1.
                data['pad_mask'] = padding_mask

            if 'gt_rbox2poly' in data and data['gt_rbox2poly'] is not None:
                # ploy to rbox
                polys = data['gt_rbox2poly']
                rbox = bbox_utils.poly2rbox(polys)
                data['gt_rbox'] = rbox

        return samples

class RandomScaledCrop(BaseOperator):
    """Resize image and bbox based on long side (with optional random scaling),
       then crop or pad image to target size.
    Args:
        target_dim (int): target size.
        scale_range (list): random scale range.
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self,
                 pad_value,
                 target_dim=512,
                 scale_range=[.1, 2.],
                 interp=cv2.INTER_LINEAR):
        super(RandomScaledCrop, self).__init__()
        self.target_dim = target_dim
        self.scale_range = scale_range
        self.interp = interp
        self.pad_value = pad_value

    def apply(self, sample, context=None):
        img = sample['image']
        h, w = img.shape[:2]
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim
        random_dim = int(dim * random_scale)
        dim_max = max(h, w)
        scale = random_dim / dim_max
        resize_w = int(w * scale)
        resize_h = int(h * scale)
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))

        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = self.pad_value * np.ones((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
            offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['image'] = canvas
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        scale_factor = sample['scale_factor']
        sample['scale_factor'] = np.asarray(
            [scale_factor[0] * scale, scale_factor[1] * scale],
            dtype=np.float32)

        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            shift_array = np.array([offset_x, offset_y] * 2, dtype=np.float32)
            boxes = sample['gt_bbox'] * scale_array - shift_array
            boxes = np.clip(boxes, 0, dim - 1)
            # filter boxes with no area
            area = np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)
            valid = (area > 1.).nonzero()[0]
            sample['gt_bbox'] = boxes[valid]
            sample['gt_class'] = sample['gt_class'][valid]

        return sample

class Pad_changeimshape(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad_changeimshape, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_segm(self, segms, offsets, im_size, size):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, h, w):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((h, w), 0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_keypoint(self, keypoints, offsets):
        n = len(keypoints[0]) // 2
        return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply(self, sample, context=None):
        im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            sample['image'] = im.astype(np.float32)
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['image'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            ## insert code 
            assert self.size[0] == self.size[1]
            origin_shape = sample['im_shape'] / sample['scale_factor']
            max_length = max(origin_shape)

            #update sample['im_shape']
            sample['im_shape'] = np.array(sample['image'].shape[:2], dtype=np.float32)

            #update scale
            scale = np.array(
                (sample['im_shape'][0] / max_length, sample['im_shape'][1] / max_length), 
                dtype=np.float32)
            sample['scale_factor'] = scale

            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets,
                                                im_size, size)

        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'],
                                                        offsets)
        return sample

# add new
class AutoAugment(BaseOperator):

    def __init__(self, autoaug_type="v1"):
        """
        Args:
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(AutoAugment, self).__init__()
        self.autoaug_type = autoaug_type

    def apply(self, sample, context=None):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        gt_bbox = sample['gt_bbox']
        im = sample['image']
        if len(gt_bbox) == 0:
            return sample

        # gt_boxes : [x1, y1, x2, y2]
        # norm_gt_boxes: [y1, x1, y2, x2]
        height, width, _ = im.shape
        norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
        norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
        norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
        norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
        norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)

        from data.transforms.autoaugment_utils import distort_image_with_autoaugment
        im, norm_gt_bbox = distort_image_with_autoaugment(
            im, norm_gt_bbox, self.autoaug_type)
        gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
        gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
        gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
        gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)

        sample['gt_bbox'] = gt_bbox
        sample['image'] = im
        sample['img_shape'] = im.shape
        sample['pad_shape'] = im.shape

        return sample

class Mosaic(BaseOperator):
    """ Mosaic operator for image and gt_bboxes
    The code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/mosaicdetection.py
    1. get mosaic coords
    2. clip bbox and get mosaic_labels
    3. random_affine augment
    4. Mixup augment as copypaste (optinal), not used in tiny/nano
    Args:
        prob (float): probability of using Mosaic, 1.0 as default
        input_dim (list[int]): input shape
        degrees (list[2]): the rotate range to apply, transform range is [min, max]
        translate (list[2]): the translate range to apply, transform range is [min, max]
        scale (list[2]): the scale range to apply, transform range is [min, max]
        shear (list[2]): the shear range to apply, transform range is [min, max]
        enable_mixup (bool): whether to enable Mixup or not
        mixup_prob (float): probability of using Mixup, 1.0 as default
        mixup_scale (list[int]): scale range of Mixup
        remove_outside_box (bool): whether remove outside boxes, False as
            default in COCO dataset, True in MOT dataset
    """

    def __init__(self,
                 prob=1.0,
                 input_dim=[1024, 1024],
                 degrees=[-10, 10],
                 translate=[-0.1, 0.1],
                 scale=[0.1, 2],
                 shear=[-2, 2],
                 enable_mixup=False,
                 mixup_prob=1.0,
                 mixup_scale=[0.5, 1.5],
                 remove_outside_box=False,
                 start_decrease_iter = None,
                 decrease_iter = None,
                 num_workers = 4,
                 base_iter = 0):
        super(Mosaic, self).__init__()
        self.prob = prob
        if isinstance(input_dim, Integral):
            input_dim = [input_dim, input_dim]
        self.input_dim = input_dim
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob
        self.mixup_scale = mixup_scale
        self.remove_outside_box = remove_outside_box

        # self.start_decrease_iter = start_decrease_iter
        # self.decrease_iter = decrease_iter
        # self.num_workers = num_workers
        # self.base_iter = base_iter

    def get_mosaic_coords(self, mosaic_idx, xc, yc, w, h, input_h, input_w):
        # (x1, y1, x2, y2) means coords in large image,
        # small_coords means coords in small image in mosaic aug.
        if mosaic_idx == 0:
            # top left
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coords = w - (x2 - x1), h - (y2 - y1), w, h
        elif mosaic_idx == 1:
            # top right
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coords = 0, h - (y2 - y1), min(w, x2 - x1), h
        elif mosaic_idx == 2:
            # bottom left
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coords = w - (x2 - x1), 0, w, min(y2 - y1, h)
        elif mosaic_idx == 3:
            # bottom right
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2,
                                                                   yc + h)
            small_coords = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        return (x1, y1, x2, y2), small_coords

    def random_affine_augment(self,
                              img,
                              labels=[],
                              input_dim=[640, 640],
                              degrees=[-10, 10],
                              scales=[0.1, 2],
                              shears=[-2, 2],
                              translates=[-0.1, 0.1]):
        # random rotation and scale
        degree = random.uniform(degrees[0], degrees[1])
        scale = random.uniform(scales[0], scales[1])
        assert scale > 0, "Argument scale should be positive."
        R = cv2.getRotationMatrix2D(angle=degree, center=(0, 0), scale=scale)
        M = np.ones([2, 3])

        # random shear
        shear = random.uniform(shears[0], shears[1])
        shear_x = math.tan(shear * math.pi / 180)
        shear_y = math.tan(shear * math.pi / 180)
        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]

        # random translation
        translate = random.uniform(translates[0], translates[1])
        translation_x = translate * input_dim[0]
        translation_y = translate * input_dim[1]
        M[0, 2] = translation_x
        M[1, 2] = translation_y

        # warpAffine
        img = cv2.warpAffine(
            img, M, dsize=tuple(input_dim), borderValue=(114, 114, 114))

        num_gts = len(labels)
        if num_gts > 0:
            # warp corner points
            corner_points = np.ones((4 * num_gts, 3))
            corner_points[:, :2] = labels[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                4 * num_gts, 2)  # x1y1, x2y2, x1y2, x2y1
            # apply affine transform
            corner_points = corner_points @M.T
            corner_points = corner_points.reshape(num_gts, 8)

            # create new boxes
            corner_xs = corner_points[:, 0::2]
            corner_ys = corner_points[:, 1::2]
            new_bboxes = np.concatenate((corner_xs.min(1), corner_ys.min(1),
                                         corner_xs.max(1), corner_ys.max(1)))
            new_bboxes = new_bboxes.reshape(4, num_gts).T

            # clip boxes
            new_bboxes[:, 0::2] = np.clip(new_bboxes[:, 0::2], 0, input_dim[0])
            new_bboxes[:, 1::2] = np.clip(new_bboxes[:, 1::2], 0, input_dim[1])
            labels[:, :4] = new_bboxes

        return img, labels

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(
            sample) == 5, "Mosaic needs 5 samples, 4 for mosaic and 1 for mixup."
        
        # if self.start_decrease_iter is not None:  #是否采用衰减策略
        #     current_iter = self.base_iter + sample[0]['curr_iter'] * self.num_workers #当前iter乘以self.num_workers
        #     if current_iter < self.start_decrease_iter:
        #         prob = self.prob
        #     elif current_iter > self.start_decrease_iter + self.decrease_iter:
        #         prob = 0 #停止
        #     else:
        #         prob = max(0,self.prob - self.prob * (current_iter-self.start_decrease_iter) / self.decrease_iter)  #逐渐衰减
        #     # print(current_iter, prob)  #当前iter以及prob
        #     if current_iter % 5000 == 0:
        #         print('current_iter {} in deteciton masaic and prob is {}'.format(current_iter, prob))
        # else:
        #     prob = self.prob
        if np.random.uniform(0., 1.) > self.prob:
            return sample[0]
       
        mosaic_gt_bbox, mosaic_gt_class, mosaic_is_crowd, mosaic_difficult = [], [], [], []
        input_h, input_w = self.input_dim
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)

        # 1. get mosaic coords
        for mosaic_idx, sp in enumerate(sample[:4]):
            img = sp['image']
            gt_bbox = sp['gt_bbox']
            h0, w0 = img.shape[:2]
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_LINEAR)
            # print(img.shape)
            (h, w, c) = img.shape[:3]
            # print(sp['image'].shape, (h,w,c), mosaic_img.shape)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (
                s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coords(
                    mosaic_idx, xc, yc, w, h, input_h, input_w)

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            # Normalized xywh to pixel xyxy format
            _gt_bbox = gt_bbox.copy()
            if len(gt_bbox) > 0:
                _gt_bbox[:, 0] = scale * gt_bbox[:, 0] + padw
                _gt_bbox[:, 1] = scale * gt_bbox[:, 1] + padh
                _gt_bbox[:, 2] = scale * gt_bbox[:, 2] + padw
                _gt_bbox[:, 3] = scale * gt_bbox[:, 3] + padh

            mosaic_gt_bbox.append(_gt_bbox)
            mosaic_gt_class.append(sp['gt_class'])
            if 'is_crowd' in sp:
                mosaic_is_crowd.append(sp['is_crowd'])
            if 'difficult' in sp:
                mosaic_difficult.append(sp['difficult'])

        # 2. clip bbox and get mosaic_labels([gt_bbox, gt_class, is_crowd])
        if len(mosaic_gt_bbox):
            mosaic_gt_bbox = np.concatenate(mosaic_gt_bbox, 0)
            mosaic_gt_class = np.concatenate(mosaic_gt_class, 0)
            if mosaic_is_crowd:
                mosaic_is_crowd = np.concatenate(mosaic_is_crowd, 0)
                mosaic_labels = np.concatenate([
                    mosaic_gt_bbox,
                    mosaic_gt_class.astype(mosaic_gt_bbox.dtype),
                    mosaic_is_crowd.astype(mosaic_gt_bbox.dtype)
                ], 1)
            elif mosaic_difficult:
                mosaic_difficult = np.concatenate(mosaic_difficult, 0)
                mosaic_labels = np.concatenate([
                    mosaic_gt_bbox,
                    mosaic_gt_class.astype(mosaic_gt_bbox.dtype),
                    mosaic_difficult.astype(mosaic_gt_bbox.dtype)
                ], 1)
            else:
                mosaic_labels = np.concatenate([
                    mosaic_gt_bbox, mosaic_gt_class.astype(mosaic_gt_bbox.dtype)
                ], 1)
            if self.remove_outside_box:
                # for MOT dataset
                flag1 = mosaic_gt_bbox[:, 0] < 2 * input_w
                flag2 = mosaic_gt_bbox[:, 2] > 0
                flag3 = mosaic_gt_bbox[:, 1] < 2 * input_h
                flag4 = mosaic_gt_bbox[:, 3] > 0
                flag_all = flag1 * flag2 * flag3 * flag4
                mosaic_labels = mosaic_labels[flag_all]
            else:
                mosaic_labels[:, 0] = np.clip(mosaic_labels[:, 0], 0,
                                              2 * input_w)
                mosaic_labels[:, 1] = np.clip(mosaic_labels[:, 1], 0,
                                              2 * input_h)
                mosaic_labels[:, 2] = np.clip(mosaic_labels[:, 2], 0,
                                              2 * input_w)
                mosaic_labels[:, 3] = np.clip(mosaic_labels[:, 3], 0,
                                              2 * input_h)
        else:
            mosaic_labels = np.zeros((1, 6))

        # 3. random_affine augment
        mosaic_img, mosaic_labels = self.random_affine_augment(
            mosaic_img,
            mosaic_labels,
            input_dim=self.input_dim,
            degrees=self.degrees,
            translates= self.translate,
            scales=self.scale,
            shears=self.shear)

        # 4. Mixup augment as copypaste, https://arxiv.org/abs/2012.07177
        # optinal, not used(enable_mixup=False) in tiny/nano
        if (self.enable_mixup and not len(mosaic_labels) == 0 and
                random.random() < self.mixup_prob):
            sample_mixup = sample[4]
            mixup_img = sample_mixup['image']
            if 'is_crowd' in sample_mixup:
                cp_labels = np.concatenate([
                    sample_mixup['gt_bbox'],
                    sample_mixup['gt_class'].astype(mosaic_labels.dtype),
                    sample_mixup['is_crowd'].astype(mosaic_labels.dtype)
                ], 1)
            elif 'difficult' in sample_mixup:
                cp_labels = np.concatenate([
                    sample_mixup['gt_bbox'],
                    sample_mixup['gt_class'].astype(mosaic_labels.dtype),
                    sample_mixup['difficult'].astype(mosaic_labels.dtype)
                ], 1)
            else:
                cp_labels = np.concatenate([
                    sample_mixup['gt_bbox'],
                    sample_mixup['gt_class'].astype(mosaic_labels.dtype)
                ], 1)
            mosaic_img, mosaic_labels = self.mixup_augment(
                mosaic_img, mosaic_labels, self.input_dim, cp_labels, mixup_img)

        sample0 = sample[0]
        # print(mosaic_img.shape)
        sample0['image'] = mosaic_img.astype(np.uint8)  # can not be float32
        # cv2.imwrite('111.png',sample0['image'])
        sample0['h'] = float(mosaic_img.shape[0])
        sample0['w'] = float(mosaic_img.shape[1])
        sample0['im_shape'][0] = sample0['h']
        sample0['im_shape'][1] = sample0['w']
        sample0['gt_bbox'] = mosaic_labels[:, :4].astype(mosaic_gt_bbox.dtype)
        sample0['gt_class'] = mosaic_labels[:, 4:5].astype(mosaic_gt_class.dtype)
        if 'is_crowd' in sample[0]:
            sample0['is_crowd'] = mosaic_labels[:, 5:6].astype(mosaic_gt_class.dtype)
        if 'difficult' in sample[0]:
            sample0['difficult'] = mosaic_labels[:, 5:6].astype(mosaic_gt_bbox.dtype)
        sample0['masaic'] = True

        # gt_boxes = sample0['gt_bbox']
        # img = sample0['image']
        # for i in range(len(gt_boxes)):
        #     box =list(map(int,gt_boxes[i]))
        #     cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]), (0,255,255), 2)
        return sample0

    def mixup_augment(self, origin_img, origin_labels, input_dim, cp_labels,
                      img):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        if len(img.shape) == 3:
            cp_img = np.ones(
                (input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0],
                             input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * cp_scale_ratio),
                  int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR)

        cp_img[:int(img.shape[0] * cp_scale_ratio), :int(img.shape[
            1] * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(cp_img, (int(cp_img.shape[1] * jit_factor),
                                     int(cp_img.shape[0] * jit_factor)))
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3),
            dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h, x_offset:
                                        x_offset + target_w]

        # adjust boxes
        cp_bboxes_origin_np = cp_labels[:, :4].copy()
        cp_bboxes_origin_np[:, 0::2] = np.clip(cp_bboxes_origin_np[:, 0::2] *
                                               cp_scale_ratio, 0, origin_w)
        cp_bboxes_origin_np[:, 1::2] = np.clip(cp_bboxes_origin_np[:, 1::2] *
                                               cp_scale_ratio, 0, origin_h)

        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        if self.remove_outside_box:
            # for MOT dataset
            cp_bboxes_transformed_np[:, 0::2] -= x_offset
            cp_bboxes_transformed_np[:, 1::2] -= y_offset
        else:
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        if cp_labels.shape[-1] == 6:
            crd_labels = cp_labels[:, 5:6].copy()
            labels = np.hstack((box_labels, cls_labels, crd_labels))
        else:
            labels = np.hstack((box_labels, cls_labels))
        if self.remove_outside_box:
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]

        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
            np.float32)

        return origin_img.astype(np.uint8), origin_labels 