import numpy as np

__all__ = ['Parameter']

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
