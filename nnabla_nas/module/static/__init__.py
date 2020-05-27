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

from .static_module import AvgPool
from .static_module import BatchNormalization
from .static_module import Conv
from .static_module import DwConv
from .static_module import GlobalAvgPool
from .static_module import Graph
from .static_module import Identity
from .static_module import Input
from .static_module import Join
from .static_module import MaxPool
from .static_module import Merging
from .static_module import Module
from .static_module import ReLU
from .static_module import Zero
from .static_module import Collapse
from .static_module import Linear
from .static_module import Dropout

__all__ = [
    'Module',
    'Graph',
    'Input',
    'Identity',
    'Zero',
    'Conv',
    'DwConv',
    'MaxPool',
    'AvgPool',
    'GlobalAvgPool',
    'ReLU',
    'BatchNormalization',
    'Merging',
    'Join',
    'Collapse',
    'Linear',
    'Dropout'
]
