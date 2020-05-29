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


class Estimator(object):
    """Estimator base class."""

    @property
    def memo(self):
        if '_memo' not in self.__dict__:
            self._memo = dict()
        return self._memo

    def get_estimation(self, module):
        """Returns the estimation of the whole module."""
        return sum(self.predict(m) for _, m in module.get_modules()
                   if len(m.modules) == 0 and m.need_grad)

    def reset(self):
        """Clear cache."""
        self.memo.clear()

    def predict(self, module):
        """Predicts the estimation for a module."""
        raise NotImplementedError
