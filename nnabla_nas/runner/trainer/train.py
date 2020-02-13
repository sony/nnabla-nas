import os

import nnabla as nn
from tqdm import tqdm

from ... import utils as ut
from ..runner import Runner


class Trainer(Runner):
    """Trainer class is a basic class for training a network.
    """

    def callback_on_start(self):
        r"""Builds the graphs and assigns parameters to the optimizers."""
        self.update_graph('train')
        self.optimizer['train'].set_parameters(
            self.model.get_parameters(grad_only=True)
        )
        self.update_graph('valid')
        self._best_err = 1.0

        # calculate the model size
        model_size = ut.count_parameters(
            self.optimizer['train'].get_parameters()
        )
        if hasattr(self.model, '_auxiliary_head'):
            model_size -= ut.count_parameters(
                self.model._auxiliary_head.get_parameters(grad_only=True))
        self.monitor.info('Model size = {:.6f} MB\n'.format(model_size*1e-6))

        assert len(self.dataloader['valid']) % self.args.mbs_valid == 0

    def run(self):
        """Run the training process."""
        for cur_epoch in range(self.args.epoch):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')

            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.args.print_frequency) == 0:
                    self.monitor.display(i, ['train_loss', 'train_err'])

            for i in tqdm(range(self.one_epoch_valid)):
                self.valid_on_batch()

            self.callback_on_epoch_end()
            self.monitor.write(cur_epoch)

        self.callback_on_finish()
        self.monitor.close()

    def train_on_batch(self, key='train'):
        r"""Updates the model parameters."""
        bz, p = self.args.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()
        for _ in range(self.accum_train):
            p['input'].d, p['target'].d = self.dataloader['train'].next()
            p['loss'].forward(clear_no_need_grad=True)
            p['loss'].backward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('train_loss', loss * self.accum_train, bz)
            self.monitor.update('train_err', err, bz)
        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Runs the validation."""
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        for _ in range(self.accum_valid):
            p['input'].d, p['target'].d = self.dataloader['valid'].next()
            p['loss'].forward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('valid_loss', loss * self.accum_valid, bz)
            self.monitor.update('valid_err', err, bz)

    def callback_on_epoch_end(self):
        r"""Calculates the error and saves the best parameters.
        """
        err = self.monitor['valid_err'].avg
        self.monitor.info(f'Current error is {err:.4f}\n')
        if self._best_err > err:
            self._best_err = err
            nn.save_parameters(
                os.path.join(self.args.output_path, 'weights.h5'),
                self.model.get_parameters()
            )

    def callback_on_finish(self):
        pass

    def callback_on_sample_graph(self):
        pass
