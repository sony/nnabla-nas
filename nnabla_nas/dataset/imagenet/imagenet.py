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
