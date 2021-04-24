from torch import set_grad_enabled
from abc import abstractmethod

set_grad_enabled(False)
''' Module class. Represents a PyTorch module.

'''
class Module:
    def __init__(self):
        self._parameters = []
        self._modules = []
        self._input = ()

    def forward(self, *input):
        return NotImplementedError

    def backward(self, *gradwrtoutput):
        return NotImplementedError

    def parameters(self):
        return self._parameters

    def modules(self):
        return self._modules

    def zero_grad(self):
        for _, grad in self._parameters:
            grad.zero_()

    def __call__(self, input):
        return self.forward(input)
