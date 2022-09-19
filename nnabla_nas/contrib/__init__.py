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

from .classification import darts
from .classification import mobilenet
from .classification import pnas
from .classification import random_wired
from .classification import zoph
from .classification import fairnas
from .classification.ofa.networks import ofa_mbv3, ofa_xception, ofa_resnet50


__all__ = ['darts', 'pnas', 'zoph', 'mobilenet', 'random_wired', 'fairnas', 'ofa_mbv3', 'ofa_xception', 'ofa_resnet50']
