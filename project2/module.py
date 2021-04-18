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

    def __call__(self, input):
        return self.forward(input)

    # def forward(self, *input):
    #     activ = self.forward
    #     if len(input) == 1:
    #         self._input = input[0]
    #         return activ(self._input)
    #     else:
    #         self._input = input
    #         return tuple(activ(t) for t in self._input)

    # def backward(self, *gradwrtoutput):
    #     grad = self.backward
    #     if len(gradwrtoutput) == 1:
    #         return grad(self._input).mul(gradwrtoutput[0])
    #     else:
    #         in_and_gradout = zip(self._input, gradwrtoutput)
    #         return tuple(grad(x).mul(t) for x, t in in_and_gradout)

    def parameters(self):
        return self._parameters

    def modules(self):
        return self._modules

    def zero_grad(self):
        for p, grad in self._parameters:
            grad.zero_()
