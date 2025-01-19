import numpy as np
from scipy.special import erf

from ..parameter import Parameter
from .module import Module

__all__ = ["ReLU", "GELU", "Swish", "Softplus"]


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.cached_mask = None

    def forward(self, x):
        self.cached_mask = x > 0
        return x * self.cached_mask

    def backward(self, grad):
        return grad * self.cached_mask


def Phi(x):
    """
    Функция Лапласа (функция распределения стандартного нормального распределения)
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))


class GELU(Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.cached_x = None

    def forward(self, x):
        self.cached_x = x
        return x * Phi(x)

    def backward(self, grad):
        x = self.cached_x
        gelu_grad = Phi(x) + (x * np.exp(-0.5 * x**2)) / np.sqrt(2 * np.pi)
        return grad * gelu_grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Swish(Module):
    def __init__(self, beta_init=1.0):
        super(Swish, self).__init__()
        self.cached_x = None
        self.cached_sigmoid = None
        self.beta = Parameter(np.array([beta_init]))

    def forward(self, x):
        self.cached_x = x
        self.cached_sigmoid = sigmoid(self.beta.data * x)
        return x * self.cached_sigmoid

    def backward(self, grad):
        x = self.cached_x
        sigmoid = self.cached_sigmoid
        sigmoid_derivative = sigmoid * (1 - sigmoid)
        self.beta.grad = np.sum(grad * (x**2) * sigmoid_derivative)
        return grad * (sigmoid + x * self.beta.data * sigmoid_derivative)

    def parameters(self):
        return [self.beta]


class Softplus(Module):
    def __init__(self):
        super(Softplus, self).__init__()
        self.cached_x = None
        self.beta = Parameter(np.array([1.0]))

    def forward(self, x):
        self.cached_x = x
        return 1 / self.beta.data * np.log(1 + np.exp(self.beta.data * x))

    def backward(self, df_dy):
        x = self.cached_x

        beta = self.beta.data
        e_beta_x = np.exp(beta * x)
        log_term = np.log(1 + e_beta_x)
        dy_dbeta = (-log_term / beta**2) + (e_beta_x * x) / (beta * (1 + e_beta_x))

        self.beta.grad = np.sum(df_dy * dy_dbeta)

        return df_dy * e_beta_x / (1 + e_beta_x)

    def parameters(self):
        return [self.beta]
