from .module import Module

__all__ = ['Module']

class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        self.cached_input = x
        self.cached_target = target
        out = (target - x) ** 2
        return out.mean()

    def backward(self):
        out = 2 * (self.cached_input - self.cached_target)
        return out / self.cached_input.size
