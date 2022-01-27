from nnabla.utils.learning_rate_scheduler import CosineScheduler


class CosineSchedulerWarmup:
    def __init__(self, base_lr, max_iter, warmup_iter, warmup_lr=1e-5):
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_lr = warmup_lr
        self.cosine = CosineScheduler(
            self.base_lr, (self.max_iter - self.warmup_iter))

    def get_learning_rate(self, current_iter):
        # Warmup
        if current_iter < self.warmup_iter:
            return (self.base_lr - self.warmup_lr) * (current_iter + 1) / self.warmup_iter + self.warmup_lr
        else:
            return self.cosine.get_learning_rate(current_iter - self.warmup_iter)
