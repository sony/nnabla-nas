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

from ... import module as Mo


class AuxiliaryHeadCIFAR(Mo.Module):
    r"""Auxiliary head used for CIFAR10 dataset.

    Args:
        channels (:obj:`int`): The number of input channels.
        num_classes (:obj:`int`): The number of classes.
    """

    def __init__(self, channels, num_classes):
        self._channels = channels
        self._num_classes = num_classes
        self._feature = Mo.Sequential(
            Mo.ReLU(),
            Mo.AvgPool(kernel=(5, 5), stride=(3, 3)),
            Mo.Conv(in_channels=channels, out_channels=128,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=128, n_dims=4),
            Mo.ReLU(),
            Mo.Conv(in_channels=128, out_channels=768,
                    kernel=(2, 2), with_bias=False),
            Mo.BatchNormalization(n_features=768, n_dims=4),
            Mo.ReLU()
        )
        self._classifier = Mo.Linear(in_features=768, out_features=num_classes)

    def call(self, input):
        out = self._feature(input)
        out = self._classifier(out)
        return out

    def extra_repr(self):
        return f'channels={self._channels}, num_classes={self._num_classes}'


class SepConv(Mo.DwConv):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        Mo.DwConv.__init__(self, in_channels=in_channels, *args, **kwargs)
        self._out_channels = out_channels
        self._conv_module_pw = Mo.Conv(self._in_channels, self._out_channels,
                                       kernel=(1, 1), pad=None, group=1,
                                       rng=self._rng, with_bias=False)

    def call(self, input):
        return self._conv_module_pw(Mo.DwConv.call(self, input))
