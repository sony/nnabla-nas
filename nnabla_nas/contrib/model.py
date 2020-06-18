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

    def loss(self, outputs, targets, loss_weights=None, *args):
        r"""Return a loss computed from a list of outputs and a list of targets.

        Args:
            outputs (list of nn.Variable): A list of output variables computed from the model.
            targets (list of nn.Variable): A list of target variables loaded from the data.
            loss_weights (list of float, optional): A list specifying scalar coefficients to weight the loss
                contributions of different model outputs. It is expected to have a 1:1 mapping to model outputs.
                Defaults to None.

        Returns:
            nn.Variable: A scalar NNabla Variable represents the loss.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def metrics(self, outputs, targets):
        r"""Return a dictionary of metrics to monitor during training.

        It is expected to have a 1:1 mapping between the model outputs and targets.

        Args:
            outputs (list of nn.Variable): A list of output variables (nn.Variable) computed from the model.
            targets (list of nn.Variable): A list of target variables (nn.Variable) loaded from the data.

        Returns:
            dict: A dictionary containing all metrics to monitor, e.g.,
                {
                    'accuracy': nn.Variable((1,)),
                    'F1': nn.Variable((1,))
                }

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError
