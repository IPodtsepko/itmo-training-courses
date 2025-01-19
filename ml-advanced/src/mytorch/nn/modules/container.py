from .module import Module

__all__ = ['Sequential']

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)

    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res
