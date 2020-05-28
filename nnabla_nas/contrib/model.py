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

from .. import module as Mo


class Model(Mo.Module):
    r"""This class is a base `Model`. Your model should be based on this
    class.
    """

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing all network parmeters of the model.

        Args:
            grad_only (bool, optional): If sets to `True`, then only
                parameters with `need_grad=True` will be retrieved. Defaults
                to `False`.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    def get_arch_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing all architecture parameters of
            the model.

        Args:
            grad_only (bool, optional): If sets to `True`, then only
                parameters with `need_grad=True` will be retrieved. Defaults
                to `False`.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def summary(self):
        r"""Returns a string summarizing the model."""
        return ''
