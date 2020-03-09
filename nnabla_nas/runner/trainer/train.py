import os

import nnabla as nn
import numpy as np
from tqdm import trange

from ...utils import helper
from ..runner import Runner


class Trainer(Runner):
    """Trainer class is a basic class for training a network.
    """

    def callback_on_start(self):
        r"""Builds the graphs and assigns parameters to the optimizers."""
        self.update_graph('train')
        params = self.model.get_parameters(grad_only=True)
        self.optimizer['train'].set_parameters(params)
        self.update_graph('valid')
        self._best_err = 1.0

        # loss and error
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))
        self.err = nn.NdArray.from_numpy_array(np.zeros((1,)))

        # calculate the model size
        model_size = helper.count_parameters(params)
        if hasattr(self.model, '_auxiliary_head'):
            model_size -= helper.count_parameters(
                self.model._auxiliary_head.get_parameters(grad_only=True))
        self.monitor.info('Model size = {:.6f} MB\n'.format(model_size*1e-6))

        # store a list of grads that will be synchronized
        if self.comm.n_procs > 1:
            self._grads = [x.grad for x in params.values()]

    def run(self):
        """Run the training process."""
        self.callback_on_start()

        for cur_epoch in range(self.args.epoch):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')

            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.args.print_frequency) == 0:
                    self.monitor.display(i, ['train_loss', 'train_err'])

            for i in trange(self.one_epoch_valid, disable=self.comm.rank > 0):
                self.valid_on_batch()

            self.callback_on_epoch_end()
            self.monitor.write(cur_epoch)

        self.callback_on_finish()
        self.monitor.close()

    def train_on_batch(self, key='train'):
        r"""Updates the model parameters."""
        bz, p = self.args.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(self.accum_train):
            self._load_data(p, self.dataloader['train'].next())
            p['loss'].forward(clear_no_need_grad=True)
            p['err'].forward(clear_buffer=True)
            p['loss'].backward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('train_loss', loss * self.accum_train, bz)
            self.monitor.update('train_err', err, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(self._grads, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Runs the validation."""
        bz, p = self.args.mbs_valid, self.placeholder['valid']

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(self.accum_valid):
            self._load_data(p, self.dataloader['valid'].next())
            p['loss'].forward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.loss.data += loss * self.accum_valid * bz
            self.err.data += err * bz

        if self.comm.n_procs > 1:
            self.event.add_default_stream_event()

    def callback_on_epoch_end(self):
        r"""Calculates the error and saves the best parameters."""
        if self.comm.n_procs > 1:
            self.comm.all_reduce([self.loss, self.err],
                                 division=True, inplace=False)

        self.loss.data /= len(self.dataloader['valid'])
        self.err.data /= len(self.dataloader['valid'])

        if self.comm.rank == 0:
            self.monitor.update('valid_loss', self.loss.data[0], 1)
            self.monitor.update('valid_err', self.err.data[0], 1)
            self.monitor.info(f'Error={self.err.data[0]:.4f}\n')
            if self._best_err > self.err.data[0]:
                self._best_err = self.err.data[0]
                path = os.path.join(self.args.output_path, 'weights.h5')
                self.model.save_parameters(path)

        # reset loss and error
        self.loss.zero()
        self.err.zero()

    def callback_on_finish(self):
        pass
