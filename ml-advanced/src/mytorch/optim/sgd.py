import numpy as np

from .optimizer import Optimizer

__all__ = [
    "SGD",
    "Nesterov",
]


class SGD(Optimizer):
    def get_grad_step(self, grad):
        return self.lr * grad


class Nesterov(Optimizer):
    def __init__(self, optim_params, lr, gamma=0.9):
        super().__init__(optim_params, lr)
        self.gamma = gamma
        self.u = {i: np.zeros_like(param.grad) for i, param in enumerate(optim_params)}

    def step(self):
        for i, param in enumerate(self.optim_params):
            self.u[i] = self.gamma * self.u[i] + self.lr * param.grad
            param.data -= self.u[i]

    def zero_grad(self):
        for param in self.optim_params:
            param.grad *= 0
