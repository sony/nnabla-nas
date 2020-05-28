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

from .data import TrainPipeline, ValPipeline
from nnabla_ext.cuda.experimental import dali_iterator


def get_data_iterators(batch_size,
                       dali_num_threads,
                       train_dir,
                       file_list,
                       dali_nvjpeg_memory_padding,
                       type_config,
                       channel_last,
                       comm,
                       stream_event_handler,
                       training=True):
    r'''Creates and returns DALI data iterators

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''
    cls_name = TrainPipeline if training else ValPipeline
    # Pipelines and Iterators for training
    train_pipe = cls_name(batch_size, dali_num_threads, comm.rank,
                          train_dir,
                          file_list,
                          dali_nvjpeg_memory_padding,
                          seed=comm.rank + 1,
                          num_shards=comm.n_procs,
                          channel_last=channel_last,
                          dtype=type_config)

    data = dali_iterator.DaliIterator(train_pipe)
    data.size = train_pipe.epoch_size("Reader") // comm.n_procs

    return data
