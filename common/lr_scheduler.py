import logging
import mxnet as mx
import math


class WarmupScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, steps, factor=0.1, warmup=False, start_lr=0., end_lr=0., warmup_step=0):
        super(WarmupScheduler, self).__init__()
        
        assert factor < 1, 'invalid decay factor for learn rate'
        assert warmup <= (start_lr > 0 and warmup_step > 0), 'invalid config for warmup'

        self.steps = steps
        self.factor = factor
        self.warmup = warmup
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_step = warmup_step
        self.learn_rate = start_lr
        self.prev_update = 0

    def __call__(self, num_update):
        if self.warmup and num_update <= self.warmup_step:
            self.learn_rate = self.start_lr + (self.end_lr - self.start_lr) * num_update / self.warmup_step

            if num_update % 50 == 0 and self.prev_update != num_update:
                logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.learn_rate)
                self.prev_update = num_update

            return self.learn_rate

        self.learn_rate = self.end_lr

        for step in self.steps:
            if num_update > step:
                self.learn_rate *= self.factor
            else:
                break

        return self.learn_rate


def warmup_scheduler(config):
    return WarmupScheduler(
            steps = [config.TRAIN.solver.epoch_size * step for step in config.TRAIN.solver.lr_step],
            factor = config.TRAIN.solver.lr_decay,
            warmup = config.TRAIN.solver.warmup,
            start_lr = config.TRAIN.solver.warmup_lr,
            end_lr = config.TRAIN.solver.lr,
            warmup_step = config.TRAIN.solver.warmup_epochs * config.TRAIN.solver.epoch_size)
