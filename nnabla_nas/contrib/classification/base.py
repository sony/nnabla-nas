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

import nnabla.functions as F

from ..model import Model


class ClassificationModel(Model):
    r"""This class is a base `Model` for classification task. Your model should be based on this class."""

    def loss(self, outputs, targets, loss_weights=None):
        r"""Return a loss computed from a list of outputs and a list of targets.

        Args:
            outputs (list of nn.Variable): A list of output variables computed from the model.
            targets (list of nn.Variable): A list of target variables loaded from the data.
            loss_weights (list of float, optional): A list specifying scalar coefficients to weight the loss
                contributions of different model outputs. It is expected to have a 1:1 mapping to model outputs.
                Defaults to None.

        Returns:
            nn.Variable: A scalar NNabla Variable represents the loss.

        """
        assert len(outputs) == 1 and len(targets) == 1

        return F.mean(F.softmax_cross_entropy(outputs[0], targets[0]))

    def metrics(self, outputs, targets):
        r"""Return a dictionary of metrics to monitor during training.

        It is expected to have a 1:1 mapping between the model outputs and targets variables.

        Args:
            outputs (list of nn.Variable): A list of output variables computed from the model.
            targets (list of nn.Variable): A list of target variables loaded from the data.

        Returns:
            dict: A dictionary containing all metrics (nn.Variable) to monitor.
                E.g., {'error': nn.Variable((1,)), 'F1': nn.Variable((1,))}
        """
        assert len(targets) == 1

        return {"error": F.mean(F.top_n_error(outputs[0], targets[0]))}
