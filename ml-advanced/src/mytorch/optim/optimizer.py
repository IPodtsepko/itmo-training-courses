__all__ = ['Optimizer']

class Optimizer:
    def __init__(self, optim_params, lr):
        self.optim_params = optim_params
        self.lr = lr

    def step(self):
        for param in self.optim_params:
            step = self.get_grad_step(param.grad)
            param.data -= step

    def get_grad_step(self, grad):
        raise NotImplementedError()

    def zero_grad(self):
        for param in self.optim_params:
            param.grad *= 0
