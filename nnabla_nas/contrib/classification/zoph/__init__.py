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

from .zoph import TrainNet
from .zoph import SearchNet
from .zoph import SepConv
from .zoph import SepConvBN
from .zoph import SepConv3x3
from .zoph import SepConv5x5
from .zoph import DilSepConv3x3
from .zoph import DilSepConv5x5
from .zoph import MaxPool3x3
from .zoph import AveragePool3x3
from .zoph import ZOPH_CANDIDATES
from .zoph import ZophBlock
from .zoph import ZophCell


__all__ = ['SearchNet',
           'TrainNet',
           'SepConv',
           'SepConvBN',
           'SepConv3x3',
           'SepConv5x5',
           'DilSepConv3x3',
           'DilSepConv5x5',
           'MaxPool3x3',
           'AveragePool3x3',
           'ZOPH_CANDIDATES',
           'ZophBlock',
           'ZophCell']
