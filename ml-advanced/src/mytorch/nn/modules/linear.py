import numpy as np

from .module import Module
from ..parameter import Parameter

__all__ = ["Linear"]

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = Parameter(np.random.randn(out_features, in_features) * np.sqrt(2 / in_features))
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        self.cached_x = None

    def forward(self, x):
        self.cached_x = x
        out = x @ self.weight.data.T
        if self.bias is not None:
            out += self.bias.data
        return out

    def backward(self, grad):
        # grad - это dLoss/dLinear
        # на выход мы должны выдать dLoss/dCachedX
        # При этом мы должны запомнить у себя dLoss/dW и dL/db
        self.weight.grad += grad.T @ self.cached_x
        if self.bias is not None:
            self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.data

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
