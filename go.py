import argparse
import json

import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context

import nnabla_nas.contrib as contrib
from args import Configuration
from nnabla_nas import runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='1',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument('--search', '-s', action='store_true',
                        help='config file')
    parser.add_argument('--algorithm', '-a', type=str, default='DartsSeacher',
                        choices=runner.__all__, help='Algorithm used to run')
    parser.add_argument('--config-file', '-f', type=str, help='config file',
                        default=None)
    parser.add_argument('--output-path', '-o', type=str, help='config file',
                        default=None)

    options = parser.parse_args()

    config = json.load(open(options.config_file)) if options.config_file \
        else dict()
    config.update({k: v for k, v in vars(options).items() if v is not None})

    # setup context for nnabla
    ctx = get_extension_context(
        config['context'],
        device_id=config['device_id']
    )
    nn.set_default_context(ctx)

    # build the model
    attributes = config['network'].copy()
    algorithm = contrib.__dict__[attributes.pop('search_space')]

    model = algorithm.SearchNet(**attributes) if config['search'] else \
        algorithm.TrainNet(**attributes)

    # Get all arguments for the runner
    conf = Configuration(config)
    loader = conf.parse()

    runner.__dict__[config['algorithm']](
        model,
        placeholder=loader['placeholder'],
        optimizer=loader['optimizer'],
        dataloader=loader['dataloader'],
        transform=loader['transform'],
        regularizer=loader['regularizer'],
        criteria=lambda o, t: F.mean(F.softmax_cross_entropy(o, t)),
        evaluate=lambda o, t: F.mean(F.top_n_error(o, t)),
        args=conf
    ).run()


# # Communicator and Context
# from tqdm import tqdm
# import pdb
# import nnabla as nn
# from nnabla.ext_utils import get_extension_context
# from nnabla_nas.dataset.imagenet.imagenet import get_data_iterators
# from nnabla_ext.cuda import StreamEventHandler
# from nnabla_nas.utils import CommunicatorWrapper

# extension_module = "cudnn"
# ctx = get_extension_context(
#     extension_module, device_id=0,
#     type_config='float'
# )

# comm = CommunicatorWrapper(ctx)
# nn.set_default_context(comm.ctx)
# stream_event_handler = StreamEventHandler(int(comm.ctx.device_id))


# batch_size = 256
# tdata = get_data_iterators(
#     batch_size=batch_size,
#     dali_num_threads=16,
#     train_dir='/speech/db/Images/ILSVRC-2012/img_train',
#     dali_nvjpeg_memory_padding=64*(1 << 20),
#     type_config=float,
#     channel_last=False,
#     comm=comm,
#     stream_event_handler=stream_event_handler,
#     training=True
# )

# vdata = get_data_iterators(
#     batch_size=batch_size,
#     dali_num_threads=16,
#     train_dir='/speech/db/Images/ILSVRC-2012/img_val_folders',
#     dali_nvjpeg_memory_padding=64*(1 << 20),
#     type_config=float,
#     channel_last=False,
#     comm=comm,
#     stream_event_handler=stream_event_handler,
#     training=False
# )


# print(tdata.next())
# print(vdata.next())

# for _ in tqdm(range(1281167//batch_size)):
#     x, g = tdata.next()
#     assert x.shape[1] == 3
#     #print(x.shape, g.shape)
# #x, g = tdata.next()


# # pdb.set_trace()
