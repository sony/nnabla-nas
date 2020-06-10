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

from abc import ABC
from abc import abstractmethod

from ..utils.data.transforms import Compose


class BaseDataLoader(ABC):
    r"""Dataloader is a base class to load your data.

    It provides an iterable over the given dataset.
    Your dataloader should overwrite `next`, `transform`, and `__len__`.
    """

    @abstractmethod
    def next(self):
        """Load the next minibatch.

        Returns:
            dict: A dictionary having the following structure:
                {
                    "inputs": <A list containing inputs (each must be `numpy.ndarray`)>,
                    "targets": <A list containing targets (each must be `numpy.ndarray`)>
                }
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def transform(self, key='train'):
        r"""Return a transform.

        Args:
            key (str, optional): Type of transform. Defaults to 'train'.
        """
        assert key in ('train', 'valid')

        return Compose([])
