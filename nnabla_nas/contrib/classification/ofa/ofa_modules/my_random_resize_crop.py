# Random resize crop opetation
# Original pytorch code from https://github.com/mit-han-lab/once-for-all

import os
import random
import math

import nnabla.functions as F


class MyResize(object):
    ACTIVE_SIZE = 224
    ACTIVE_SIZE_NUM = 0
    IMAGE_SIZE_LIST = [224]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = True
    IS_TRAINING = False

    EPOCH = 0
    BATCH = 0

    def __init__(self):
        self.max_size = max(MyResize.IMAGE_SIZE_LIST)

    def __call__(self, img):
        # [resize]
        if MyResize.IS_TRAINING:
            self.sample_image_size()
        return F.interpolate(
            img, output_size=(MyResize.ACTIVE_SIZE, MyResize.ACTIVE_SIZE), mode='linear')

    @staticmethod
    def get_candidate_image_size():
        if MyResize.CONTINUOUS:
            min_size = min(MyResize.IMAGE_SIZE_LIST)
            max_size = max(MyResize.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % MyResize.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyResize.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id=None):
        if batch_id is None:
            batch_id = MyResize.BATCH
        if MyResize.SYNC_DISTRIBUTED:
            _seed = int('%d%.3d' % (batch_id, MyResize.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = MyResize.get_candidate_image_size()
        # MyRandomResizedCrop.ACTIVE_SIZE = random.choices(candidate_sizes, weights=relative_probs)[0]
        idx = random.choices(range(len(candidate_sizes)), weights=relative_probs)[0]
        MyResize.ACTIVE_SIZE = candidate_sizes[idx]
        MyResize.ACTIVE_SIZE_NUM = idx


"""class MyRandomResizedCrop(object):
    ACTIVE_SIZE = 224
    ACTIVE_SIZE_NUM = 0
    IMAGE_SIZE_LIST = [224]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = True

    EPOCH = 0
    BATCH = 0

    def __init__(self, min_scale=0.08, max_scale=1.0, aspect_ratio=(4 / 3, 3 / 4)):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._aspect_ratio = aspect_ratio

    def __call__(self, img):
        self.sample_image_size()
        scale = random.uniform(self._min_scale, self._max_scale)
        aspect_ratio = random.uniform(self._aspect_ratio[0], self._aspect_ratio[1])
        nw = (img.shape[2] * img.shape[3] * scale * aspect_ratio) ** (1/2)
        nh = nw / aspect_ratio
        nw = img.shape[2] if nw > img.shape[2] else nw
        nh = img.shape[3] if nh > img.shape[3] else nh
        # [crop & reshape]
        img = F.random_crop(img, shape=(img.shape[0], img.shape[1], nw, nh))
        img = F.interpolate(
            img,
            output_size=(MyRandomResizedCrop.ACTIVE_SIZE, MyRandomResizedCrop.ACTIVE_SIZE),
            mode='linear')
        # img = F.image_augmentation(
        # img,
        # shape=(img.shape[0], img.shape[1], MyRandomResizedCrop.ACTIVE_SIZE, MyRandomResizedCrop.ACTIVE_SIZE),
        # )
        return img

    @staticmethod
    def get_candidate_image_size():
        if MyRandomResizedCrop.CONTINUOUS:
            min_size = min(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            max_size = max(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % MyRandomResizedCrop.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyRandomResizedCrop.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id=None):
        if batch_id is None:
            batch_id = MyRandomResizedCrop.BATCH
        if MyRandomResizedCrop.SYNC_DISTRIBUTED:
            _seed = int('%d%.3d' % (batch_id, MyRandomResizedCrop.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = MyRandomResizedCrop.get_candidate_image_size()
        idx = random.choices(range(len(candidate_sizes)), weights=relative_probs)[0]
        MyRandomResizedCrop.ACTIVE_SIZE = candidate_sizes[idx]
        MyRandomResizedCrop.ACTIVE_SIZE_NUM = idx


class MyValidationCrop(object):
    MAX_SIZE = 224
    ACTIVE_SIZE = 224

    def __init__(self, img_size=None):
        if img_size is None:
            img_size = MyValidationCrop.ACTIVE_SIZE
        MyValidationCrop.MAX_SIZE = img_size

    def __call__(self, img):
        new_shape = int(math.ceil(img.shape[2] / 0.875))
        img = F.image_augmentation(img, shape=(img.shape[0], img.shape[1], new_shape, new_shape))
        # center crop
        start = (new_shape - MyValidationCrop.ACTIVE_SIZE) // 2
        end = start + MyValidationCrop.ACTIVE_SIZE
        img = img[:, :, start:end, start:end]
        return img"""
