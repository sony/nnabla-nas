class Searcher(object):

    def __init__(self,
                 max_epoch,
                 updater_model,
                 updater_arch,
                 callbacks_begin,
                 callbacks_end,
                 callbacks_update_model_begin,
                 callbacks_update_model_end,
                 callbacks_update_arch_begin,
                 callbacks_update_arch_end):

        self.max_epoch = max_epoch
        self.updater_model = updater_model
        self.updater_arch = updater_arch
        self.callbacks_begin = callbacks_begin
        self.callbacks_end = callbacks_end
        self.callbacks_update_model_begin = callbacks_update_model_begin
        self.callbacks_update_model_end = callbacks_update_model_end
        self.callbacks_update_arch_begin = callbacks_update_arch_begin
        self.callbacks_update_arch_end = callbacks_update_arch_end

    def run(self):
        # begin
        for callback in self.callbacks_begin:
            callback()
        # training
        for epoch in range(self.max_epoch):
            for i in range(self.max_one_epoch):
                # model training
                for callback in self.callbacks_update_model_begin:
                    callback()
                for update in self.updater_model:
                    update(epoch)
                for callback in self.callbacks_update_model_end:
                    callback()
                # architecture training
                for callback in self.callbacks_update_arch_begin:
                    callback()
                for update in self.updater_arch:
                    update(epoch)
                for callback in self.callbacks_update_arch_end:
                    callback()
        # end
        for callback in self.callbacks_end:
            callback()
