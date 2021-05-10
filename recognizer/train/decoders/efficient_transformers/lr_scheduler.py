import keras4torch as k4t

class NoamScheme(k4t.callbacks.Callback):
    def __init__(self, optimizer, d_model, warmup_steps=4000) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.i = 1

    def on_batch_begin(self, trainer):
        factor0 = self.d_model ** -0.5
        factor1 = min(self.i**-0.5, self.i*self.warmup_steps**-1.5)
        self.i += 1
        self.set_lr(factor0 * factor1)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr


class LinearWarmup(k4t.callbacks.Callback):
    def __init__(self, optimizer, warmup_epochs=1) -> None:
        self.optimizer = optimizer
        self.initial_lr = self.get_lr()
        self.warmup_epochs = warmup_epochs
        self.i = 1
        self.enabled = True

    def on_batch_begin(self, trainer):
        if not self.enabled:
            return

        warmup_steps = len(trainer.data_loaders[0]) * self.warmup_epochs
        
        self.i += 1
        self.set_lr(self.i / warmup_steps * self.initial_lr)

        if self.i >= warmup_steps:
            self.enabled = False


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr