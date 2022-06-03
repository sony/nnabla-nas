# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import os
import random
import time

import nnabla.functions as F


class MyResize(object):
    """Resizes input image by interpolation"""

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
        # resize
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
        idx = random.choices(range(len(candidate_sizes)), weights=relative_probs)[0]
        MyResize.ACTIVE_SIZE = candidate_sizes[idx]
        MyResize.ACTIVE_SIZE_NUM = idx
