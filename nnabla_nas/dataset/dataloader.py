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

class DataLoader(object):
    r""" Dataloader class.

    Combines a data iterator and a transform (on numpy), and provides an
    iterable over the given dataset.  All subclasses should overwrite
    `__len__` and `next`.

    Args:
        data_iterator (iterator): Data iterator
        transform (object, optional): A transform to be applied on a sample.
            Defaults to None.
    """

    def __init__(self, data_iterator, transform=None):
        self.data_iterator = data_iterator
        self.transform = transform

    def next(self):
        x, t = self.data_iterator.next()
        if self.transform:
            x = self.transform(x)
        return x, t

    def __len__(self):
        return self.data_iterator.size
