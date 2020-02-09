from abc import ABC
from abc import abstractmethod

from .. import utils as ut
from ..utils import ProgressMeter


class Runner(ABC):
    r"""Searching the best architecture.

    Args:
        model (`nnabla_nas.contrib.model.Model`): The search model used to
            search the architecture.
        placeholder (dict): This stores `input` and `target` Variables for
            `train` and `valid` graphs.
        optimizer (dict): This stores optimizers for both `train` and `valid`
            graphs.
        dataloader (dict): This stores dataloaders for both `train` and `valid`
            graphs.
        criteria (function): Loss function used to train the network.
        evaluate (function): Evaluation criteria used log the output,
            e.g., top_1_err.
        args (Configuration): This stores all hyperparmeters used during
            training.
        regularizer (dict, optional): This stores contraints for the network.
            Defaults to None.
    """

    def __init__(self,
                 model,
                 placeholder,
                 optimizer,
                 dataloader,
                 criteria,
                 evaluate,
                 args,
                 regularizer=None):

        self.model = model
        self.criteria = criteria
        self.evaluate = evaluate
        self.dataloader = dataloader
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.placeholder = placeholder
        self.args = args

        # aditional argurments
        self.accum_train = self.args.bs_train // self.args.mbs_train
        self.accum_valid = self.args.bs_valid // self.args.mbs_valid
        self.one_epoch_train = len(self.dataloader['train']) // args.bs_train
        self.one_epoch_valid = len(self.dataloader['valid']) // args.bs_valid

        # monitor log info
        self.monitor = ProgressMeter(
            self.one_epoch_train,
            path=args.output_path
        )

        # initialize tasks
        self.callback_on_start()

    @abstractmethod
    def run(self):
        r"""Run the training process."""
        pass

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        self.callback_on_sample_graph()
        self.model.apply(training=key != 'valid')

        p = self.placeholder['valid' if key == 'valid' else 'train']

        image = p['input'] if key == 'valid' else ut.data_augment(p['input'])
        accum = self.accum_valid if key == 'valid' else self.accum_train

        output = self.model(image)

        # output features
        if isinstance(output, tuple):
            aux = output[1]
            w = self.args.aux_weight
            p['output'] = output[0]
        else:
            p['output'] = output
        # loss function
        p['loss'] = self.criteria(p['output'], p['target']) / accum
        if isinstance(output, tuple):
            p['loss'] += w * self.criteria(aux, p['target']) / accum
        # top_1_error
        p['err'] = self.evaluate(
            p['output'].get_unlinked_variable(),
            p['target']
        )
        # setup persistent flags.
        p['output'].apply(persistent=True)
        p['loss'].apply(persistent=True)
        p['err'].apply(persistent=True)

    @abstractmethod
    def train_on_batch(self, key='train'):
        r"""Runs the model update on a single batch of train data."""
        pass

    @abstractmethod
    def valid_on_batch(self):
        r"""Runs the model update on a single batch of valid data."""
        pass

    @abstractmethod
    def callback_on_epoch_end(self):
        r"""Calls this after one epoch."""
        pass

    @abstractmethod
    def callback_on_sample_graph(self):
        r"""Calls this before sample a graph."""
        pass

    @abstractmethod
    def callback_on_start(self):
        r"""Calls this on starting the training."""
        pass

    @abstractmethod
    def callback_on_finish(self):
        r"""Calls this on finishing the training."""
        pass
