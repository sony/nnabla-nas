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

from nnabla_nas.utils.helper import CommunicatorWrapper
from nnabla.ext_utils import get_extension_context

from nnabla_nas.dataset.cifar10 import DataLoader


def test_cifar10_dataloader():
    ctx = get_extension_context(
        'cpu',
        device_id='0'
    )

    # setup for distributed training
    comm = CommunicatorWrapper(ctx)

    loader = DataLoader(1, searching=False, training=True, train_portion=0.95, communicator=comm)
    assert len(loader) == 50000

    loader = DataLoader(1, searching=False, training=False, train_portion=0.95, communicator=comm)
    assert len(loader) == 10000

    loader = DataLoader(1, searching=True, training=True, train_portion=0.5, communicator=comm)
    assert len(loader) == 25000

    loader = DataLoader(1, searching=True, training=False, train_portion=0.5, communicator=comm)
    assert len(loader) == 25000
