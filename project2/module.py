from torch import set_grad_enabled
from abc import abstractmethod

set_grad_enabled(False)

class Module:
    def __init__(self):
        self._parameters = []
        self._modules = []
        self._input = ()

    @abstractmethod
    def _activation_function(self, x):
        return NotImplementedError

    @abstractmethod
    def _activation_gradient(self, x):
        return NotImplementedError

    def forward(self, *input):
        self._input = input
        activ = self._activation_function
        if len(input) == 1:
            return activ(self._input[0])
        else:
            return tuple(activ(t) for t in self._input)

    def backward(self, *gradwrtoutput):
        grad = self._activation_gradient
        if len(gradwrtoutput) == 1:
            return grad(self._input[0]).mul(gradwrtoutput[0])
        else:
            in_and_gradout = zip(self._input, gradwrtoutput)
            return tuple(grad(x).mul(t) for x, t in in_and_gradout)

    def param(self):
        return NotImplementedError

    def modules(self):
        return self._modules