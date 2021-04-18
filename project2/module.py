from torch import set_grad_enabled
from abc import abstractmethod

set_grad_enabled(False)

class Module:
    def __init__(self):
        self._parameters = []
        self._modules = []
        self._input = ()

    def _activation_function(self, x):
        return NotImplementedError

    def _activation_gradient(self, x):
        return NotImplementedError

    def __call__(self, input):
        return self.forward(input)

    def forward(self, *input):
        activ = self._activation_function
        if len(input) == 1:
            self._input = input[0]
            return activ(self._input)
        else:
            self._input = input
            return tuple(activ(t) for t in self._input)

    def backward(self, *gradwrtoutput):
        grad = self._activation_gradient
        if len(gradwrtoutput) == 1:
            return grad(self._input[0]).mul(gradwrtoutput[0])
        else:
            in_and_gradout = zip(self._input, gradwrtoutput)
            return tuple(grad(x).mul(t) for x, t in in_and_gradout)

    def parameters(self):
        return self._parameters

    def modules(self):
        return self._modules

    def zero_grad(self):
        for p, grad in self._parameters:
            grad.zero_()
