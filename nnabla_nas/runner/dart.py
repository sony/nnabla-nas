from .searcher import Searcher


class DartsSearcher():

    def __init__(self, solver,
                 tinput=None, tlabel=None, tpred=None, tdata=None,
                 vinput=None, vlabel=None, vpred=None, vdata=None,
                 monitor_path=None, model_save_path=None,
                 max_epoch=1, iter_per_epoch=None,
                 val_iter=None):

        self.searcher = Searcher(

        )

    def fit(self):
        self.searcher().run()
