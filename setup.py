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

import os

from setuptools import setup

if __name__ == '__main__':
    root_dir = os.path.realpath(os.path.dirname(__file__))
    setup(
        entry_points={"console_scripts": ["nnabla_nas=nnabla_nas.main:main"]},
        install_requires=['graphviz',
                          'h5py',
                          'mako',
                          'mypy',
                          'networkx',
                          'nnabla',
                          'sklearn',
                          'tensorboard',
                          'tqdm'
                          ],
        package_dir={'nnabla_nas': 'nnabla_nas'},
        packages=['nnabla_nas',
                  'nnabla_nas.contrib',
                  'nnabla_nas.contrib.classification.darts',
                  'nnabla_nas.contrib.classification.mobilenet',
                  'nnabla_nas.contrib.classification.pnas',
                  'nnabla_nas.contrib.classification.random_wired',
                  'nnabla_nas.contrib.classification.zoph',
                  'nnabla_nas.dataset',
                  'nnabla_nas.module',
                  'nnabla_nas.module.static',
                  'nnabla_nas.optimizer',
                  'nnabla_nas.runner',
                  'nnabla_nas.runner.searcher',
                  'nnabla_nas.runner.trainer',
                  'nnabla_nas.utils',
                  'nnabla_nas.utils.data',
                  'nnabla_nas.utils.estimator',
                  'nnabla_nas.utils.tensorboard'],
        name='nnabla_nas',
        description='Use NNC compute resource from NNabla',
        version='0.0.1',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Operating System :: POSIX :: Linux'
        ],
        keywords="deep learning artificial intelligence machine learning neural network",
        python_requires='>=3.6'
    )
